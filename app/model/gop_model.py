import warnings
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, Wav2Vec2Config, Wav2Vec2ForCTC
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from app.model.loss_function import OrdinalLogLoss


class GOPWav2Vec2Config(PretrainedConfig):
    model_type = "gop-wav2vec2"

    def __init__(
        self,
        base_model_name_or_path: Optional[str] = None,
        num_gop_labels: Optional[int] = 2,
        gop_head_labels: Optional[Dict[str, int]] = None,
        gop_embedding_dim: int = 128,
        gop_transformer_nhead: int = 4,
        gop_transformer_dim_feedforward: int = 512,
        gop_transformer_dropout: float = 0.1,
        gop_transformer_nlayers: int = 1,
        gop_loss_alpha: float = 0.5,
        gop_ce_weights: Optional[Union[List[float], Dict[str, List[float]]]] = None,
        pad_id: Optional[int] = None,
        unk_id: Optional[int] = None,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        word_boundary_id: Optional[int] = None,
        token_id_vocab: Optional[List[int]] = None,
        ctc_config: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        if gop_head_labels is not None:
            self.gop_head_labels = {str(k): int(v) for k, v in gop_head_labels.items()}
        elif num_gop_labels is not None:
            self.gop_head_labels = {"default": int(num_gop_labels)}
        else:
            self.gop_head_labels = None
        self.num_gop_labels = num_gop_labels
        self.gop_embedding_dim = gop_embedding_dim
        self.gop_transformer_nhead = gop_transformer_nhead
        self.gop_transformer_dim_feedforward = gop_transformer_dim_feedforward
        self.gop_transformer_dropout = gop_transformer_dropout
        self.gop_transformer_nlayers = gop_transformer_nlayers
        self.gop_loss_alpha = gop_loss_alpha
        self.gop_ce_weights = gop_ce_weights
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.word_boundary_id = word_boundary_id
        self.token_id_vocab = token_id_vocab
        self.ctc_config = ctc_config


class GOPPhonemeClassifier(PreTrainedModel):
    config_class = GOPWav2Vec2Config

    def __init__(
        self, config: GOPWav2Vec2Config, load_pretrained_backbone: bool = False
    ):
        super().__init__(config)

        if config.ctc_config is not None:
            backbone_config = Wav2Vec2Config.from_dict(config.ctc_config)
        elif config.base_model_name_or_path is not None:
            backbone_config = Wav2Vec2Config.from_pretrained(
                config.base_model_name_or_path
            )
        else:
            backbone_config = Wav2Vec2Config()

        self.ctc_model = Wav2Vec2ForCTC(backbone_config)
        self.config.ctc_config = backbone_config.to_dict()

        self.blank_id = config.pad_id
        self.unk_id = config.unk_id
        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.pad_id = self.blank_id
        self.word_boundary_id = config.word_boundary_id

        special_ids = {
            self.blank_id,
            self.unk_id,
            self.bos_id,
            self.eos_id,
            self.pad_id,
        }
        self.special_ids = {i for i in special_ids if i is not None}

        vocab_size = int(self.ctc_model.config.vocab_size)
        self.token_id_vocab = (
            config.token_id_vocab
            if config.token_id_vocab is not None
            else [i for i in range(vocab_size) if i not in self.special_ids]
        )

        self.gop_feature_dim = 1 + len(self.token_id_vocab) + 1
        self.embedding_dim = int(config.gop_embedding_dim)
        self.token_embedding = nn.Embedding(
            vocab_size,
            self.embedding_dim,
            padding_idx=self.pad_id if self.pad_id is not None else 0,
        )
        self.combined_feature_dim = self.gop_feature_dim + self.embedding_dim
        self.gop_part_dropout = nn.Dropout(config.gop_transformer_dropout)
        self.gop_part_norm = nn.LayerNorm(self.gop_feature_dim)

        self.lstm_hidden_size = int(config.gop_transformer_dim_feedforward)
        self.gop_rnn = nn.LSTM(
            input_size=self.combined_feature_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=config.gop_transformer_nlayers,
            dropout=(
                config.gop_transformer_dropout
                if config.gop_transformer_nlayers > 1
                else 0.0
            ),
            bidirectional=True,
            batch_first=True,
        )

        head_label_config = getattr(config, "gop_head_labels", None)
        if head_label_config is None:
            if config.num_gop_labels is None:
                raise ValueError(
                    "Config must provide gop_head_labels or num_gop_labels for the classifier."
                )
            head_label_config = {"default": int(config.num_gop_labels)}
        self.head_label_config = {str(k): int(v) for k, v in head_label_config.items()}
        self.head_hidden_dim = self.lstm_hidden_size * 2
        self.head_mlps = nn.ModuleDict(
            {
                head: nn.Sequential(
                    nn.Linear(self.head_hidden_dim, self.head_hidden_dim),
                    nn.LeakyReLU(),
                )
                for head in self.head_label_config.keys()
            }
        )
        self.classifiers = nn.ModuleDict(
            {
                head: nn.Linear(self.head_hidden_dim, num_labels)
                for head, num_labels in self.head_label_config.items()
            }
        )

        self._init_losses()

        self.post_init()

        if load_pretrained_backbone:
            self._load_pretrained_backbone()

    def _load_pretrained_backbone(self) -> None:
        if not self.config.base_model_name_or_path:
            raise ValueError(
                "Cannot load pretrained backbone without base_model_name_or_path in config."
            )
        pretrained_backbone = Wav2Vec2ForCTC.from_pretrained(
            self.config.base_model_name_or_path
        )
        missing_keys, unexpected_keys = self.ctc_model.load_state_dict(
            pretrained_backbone.state_dict(), strict=False
        )
        del pretrained_backbone
        if missing_keys:
            warnings.warn(
                f"Missing keys when loading pretrained backbone: {missing_keys}"
            )
        if unexpected_keys:
            warnings.warn(
                f"Unexpected keys when loading pretrained backbone: {unexpected_keys}"
            )

    def _init_losses(self) -> None:
        alpha = getattr(self.config, "gop_loss_alpha", 0.5)
        weights = getattr(self.config, "gop_ce_weights", None)
        loss_modules = {}
        for head, num_labels in self.head_label_config.items():
            head_weights = None
            if isinstance(weights, dict):
                head_weights = weights.get(head)
            elif weights is not None:
                head_weights = weights
            if head_weights is not None:
                head_weights = (
                    head_weights
                    if isinstance(head_weights, torch.Tensor)
                    else torch.tensor(head_weights, dtype=torch.float)
                )
            loss_modules[head] = OrdinalLogLoss(
                num_classes=int(num_labels),
                alpha=alpha,
                reduction="mean",
                class_weights=head_weights,
            )
        self.loss_fns = nn.ModuleDict(loss_modules)

    def _calculate_log_prob(
        self,
        log_probs_TNC: torch.Tensor,
        input_lengths: torch.Tensor,
        target_ids: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        target_ids_cpu = target_ids.cpu()
        target_lengths_cpu = target_lengths.cpu()
        log_probs_cpu = log_probs_TNC.cpu()
        input_lengths_cpu = input_lengths.cpu()

        targets_flat = []
        for i in range(target_ids_cpu.size(0)):
            valid_targets = target_ids_cpu[i, : target_lengths_cpu[i]]
            targets_flat.append(valid_targets)
        targets_cat = (
            torch.cat(targets_flat)
            if targets_flat
            else torch.tensor([], dtype=torch.long)
        )

        if target_lengths_cpu.sum() == 0:
            return torch.full(
                (log_probs_TNC.size(1),), -float("inf"), device=log_probs_TNC.device
            )

        ctc_loss_fn = torch.nn.CTCLoss(
            blank=self.blank_id, reduction="none", zero_infinity=True
        )
        try:
            loss_per_item = ctc_loss_fn(
                log_probs_cpu, targets_cat, input_lengths_cpu, target_lengths_cpu
            )
            return -loss_per_item.to(log_probs_TNC.device)
        except Exception as e:
            warnings.warn(f"CTCLoss calculation failed: {e}. Returning -inf for batch.")
            return torch.full(
                (log_probs_TNC.size(1),), -float("inf"), device=log_probs_TNC.device
            )

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        def _conv_out_length(input_length, kernel_size, stride):
            return (
                torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1
            )

        for kernel_size, stride in zip(
            self.ctc_model.config.conv_kernel, self.ctc_model.config.conv_stride
        ):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        return input_lengths

    def _prepare_labels(
        self, labels: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Optional[Dict[str, torch.Tensor]]:
        if labels is None:
            return None
        if isinstance(labels, torch.Tensor):
            if len(self.head_label_config) != 1:
                raise ValueError(
                    "Multi-head setup requires `labels` to be a dict keyed by head name."
                )
            head_name = next(iter(self.head_label_config))
            return {head_name: labels}
        if not isinstance(labels, dict):
            raise ValueError(
                "`labels` must be a Tensor for single-head setups or a dict for multi-head setups."
            )
        return labels

    def _validate_inputs(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        canonical_token_ids: torch.Tensor,
        token_lengths: torch.Tensor,
        token_mask: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if input_values.dim() != 2:
            raise ValueError(
                f"`input_values` must be 2D (batch, time); got shape {tuple(input_values.shape)}."
            )
        if attention_mask is None or attention_mask.shape != input_values.shape:
            raise ValueError(
                "`attention_mask` must be provided and match the shape of `input_values`."
            )

        if canonical_token_ids is None or token_lengths is None or token_mask is None:
            raise ValueError(
                "`canonical_token_ids`, `token_lengths`, and `token_mask` are required."
            )

        if canonical_token_ids.dim() != 2:
            raise ValueError(
                f"`canonical_token_ids` must be 2D (batch, tokens); got shape {tuple(canonical_token_ids.shape)}."
            )
        batch_size, max_tokens = canonical_token_ids.shape
        if batch_size != input_values.shape[0]:
            raise ValueError(
                "Batch size mismatch between `input_values` and `canonical_token_ids`."
            )

        if token_mask.dim() != 2 or token_mask.shape != canonical_token_ids.shape:
            raise ValueError(
                "`token_mask` must be the same shape as `canonical_token_ids`."
            )

        if token_lengths.dim() != 1 or token_lengths.shape[0] != batch_size:
            raise ValueError(
                "`token_lengths` must be 1D with length equal to batch size."
            )
        if torch.any(token_lengths < 0):
            raise ValueError("`token_lengths` must be non-negative.")
        if torch.any(token_lengths > max_tokens):
            raise ValueError(
                "`token_lengths` cannot exceed the number of provided tokens."
            )

        token_mask_bool = token_mask.to(
            device=canonical_token_ids.device, dtype=torch.bool
        )
        arange_positions = torch.arange(max_tokens, device=canonical_token_ids.device)
        padded_active = token_mask_bool & (
            arange_positions.unsqueeze(0) >= token_lengths.unsqueeze(1)
        )
        if torch.any(padded_active):
            raise ValueError(
                "`token_mask` marks padded positions as valid (indices >= token_lengths)."
            )
        if torch.any(token_mask_bool.sum(dim=1) > token_lengths):
            raise ValueError(
                "`token_mask` has more active positions than `token_lengths` for some batch items."
            )

        if labels is not None:
            if not isinstance(labels, dict):
                raise ValueError(
                    "`labels` must be a dict keyed by head name after normalization."
                )
            expected_heads = set(self.head_label_config.keys())
            label_heads = set(labels.keys())
            unknown_heads = label_heads - expected_heads
            missing_heads = expected_heads - label_heads
            if unknown_heads:
                raise ValueError(
                    f"Unexpected label heads provided: {sorted(unknown_heads)}."
                )
            if missing_heads:
                raise ValueError(f"Missing label heads: {sorted(missing_heads)}.")
            for head, head_labels in labels.items():
                if head_labels.shape != canonical_token_ids.shape:
                    raise ValueError(
                        f"Labels for head '{head}' must match `canonical_token_ids` shape "
                        f"{tuple(canonical_token_ids.shape)}; got {tuple(head_labels.shape)}."
                    )
                if head_labels.dtype not in (torch.int64, torch.long):
                    raise ValueError(
                        f"Labels for head '{head}' must be integer tensors; got dtype {head_labels.dtype}."
                    )
                masked_positions = token_mask_bool.logical_not()
                bad_mask = masked_positions & (head_labels != -100)
                if torch.any(bad_mask):
                    bad_count = int(bad_mask.sum().item())
                    raise ValueError(
                        f"Labels for head '{head}' must be -100 at masked positions; found {bad_count} mismatches."
                    )

        return token_mask_bool

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        canonical_token_ids: torch.Tensor,
        token_lengths: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        device = input_values.device

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        labels = self._prepare_labels(labels)

        if self.training or labels is not None:
            if (
                canonical_token_ids is None
                or token_lengths is None
                or token_mask is None
            ):
                raise ValueError(
                    "`canonical_token_ids`, `token_lengths`, and `token_mask` are required during training."
                )

        if token_mask is None:
            raise ValueError(
                "`token_mask` must be provided to GOPPhonemeClassifier.forward."
            )

        token_mask_bool = self._validate_inputs(
            input_values=input_values,
            attention_mask=attention_mask,
            canonical_token_ids=canonical_token_ids,
            token_lengths=token_lengths,
            token_mask=token_mask,
            labels=labels,
        ).to(device=device)

        outputs = self.ctc_model.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state

        logits_ctc = self.ctc_model.lm_head(hidden_states)
        log_probs_ctc = F.log_softmax(logits_ctc, dim=-1)
        log_probs_TNC = log_probs_ctc.permute(1, 0, 2).contiguous()

        batch_size = input_values.size(0)
        input_lengths_samples = attention_mask.sum(dim=-1)
        input_lengths_frames = self._get_feat_extract_output_lengths(
            input_lengths_samples
        )
        input_lengths_frames = torch.clamp(
            input_lengths_frames, max=log_probs_TNC.size(0)
        )

        max_token_len = canonical_token_ids.size(1) if canonical_token_ids is not None else 0
        batch_combined_features_list = []

        token_embeddings = self.token_embedding(canonical_token_ids)

        lpp_log_prob_batch = self._calculate_log_prob(
            log_probs_TNC, input_lengths_frames, canonical_token_ids, token_lengths
        )

        for token_idx in range(max_token_len):
            current_token_embeddings = token_embeddings[:, token_idx, :]
            active_mask = token_mask_bool[:, token_idx]
            skip_mask = ~active_mask

            all_sub_log_probs = []
            for sub_token_id in self.token_id_vocab:
                sub_ids_batch = canonical_token_ids.clone()
                sub_ids_batch[active_mask, token_idx] = sub_token_id
                log_prob_sub_batch = self._calculate_log_prob(
                    log_probs_TNC, input_lengths_frames, sub_ids_batch, token_lengths
                )
                all_sub_log_probs.append(log_prob_sub_batch)

            sub_log_probs_batch = torch.stack(all_sub_log_probs, dim=1)
            sub_log_probs_batch = F.log_softmax(sub_log_probs_batch, dim=1)
            sub_log_probs_batch = torch.nan_to_num(
                sub_log_probs_batch, nan=0.0, posinf=1e10, neginf=-1e10
            )
            if skip_mask.any():
                sub_log_probs_batch[skip_mask, :] = 0.0

            del_lpr_list = []
            for b_idx in range(batch_size):
                if skip_mask[b_idx]:
                    del_lpr_list.append(torch.tensor(-1e10, device=device))
                else:
                    item_tokens = canonical_token_ids[
                        b_idx, : token_lengths[b_idx]
                    ].tolist()
                    del_tokens_list = item_tokens[:token_idx] + item_tokens[token_idx + 1 :]
                    if not del_tokens_list:
                        log_prob_del_item = torch.tensor(-float("inf"), device=device)
                    else:
                        del_ids_tensor = torch.tensor(
                            [del_tokens_list],
                            dtype=torch.long,
                            device=canonical_token_ids.device,
                        )
                        del_len_tensor = torch.tensor(
                            [len(del_tokens_list)],
                            dtype=torch.long,
                            device=canonical_token_ids.device,
                        )
                        log_probs_item_TNC = log_probs_TNC[:, b_idx : b_idx + 1, :]
                        input_len_item = input_lengths_frames[b_idx : b_idx + 1]
                        log_prob_del_item = self._calculate_log_prob(
                            log_probs_item_TNC,
                            input_len_item,
                            del_ids_tensor,
                            del_len_tensor,
                        )
                        if log_prob_del_item.dim() > 0:
                            log_prob_del_item = log_prob_del_item[0]
                    lpr_del_item = lpp_log_prob_batch[b_idx] - log_prob_del_item
                    lpr_del_item = torch.nan_to_num(
                        lpr_del_item, nan=0.0, posinf=1e10, neginf=-1e10
                    )
                    del_lpr_list.append(lpr_del_item)
            del_lpr_batch = torch.stack(del_lpr_list)

            gop_part = torch.cat(
                [
                    lpp_log_prob_batch.unsqueeze(1),
                    sub_log_probs_batch,
                    del_lpr_batch.unsqueeze(1),
                ],
                dim=1,
            )
            gop_part = self.gop_part_norm(gop_part)
            gop_part = self.gop_part_dropout(gop_part)
            combined_features = torch.cat([gop_part, current_token_embeddings], dim=1)
            batch_combined_features_list.append(combined_features)

        transformer_input = torch.stack(batch_combined_features_list, dim=1)

        gop_rnn_output, _ = self.gop_rnn(transformer_input)

        final_logits = {}
        for head, classifier in self.classifiers.items():
            head_features = self.head_mlps[head](gop_rnn_output)
            final_logits[head] = classifier(head_features)

        loss = None
        if labels is not None:
            loss = 0.0
            for head, head_logits in final_logits.items():
                head_labels = labels[head]
                head_loss = self.loss_fns[head](head_logits, head_labels)
                loss += head_loss

        if not return_dict:
            output = (final_logits,)
            if loss is not None:
                output = (loss,) + output
            return output

        return SequenceClassifierOutput(
            loss=loss,
            logits=final_logits,
            hidden_states=None,
            attentions=None,
        )
