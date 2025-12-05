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
    """
    Configuration for GOP-enhanced model that wraps a Wav2Vec2ForCTC backbone.
    """

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
        self.token_id_vocab = token_id_vocab
        self.ctc_config = ctc_config


class GOPPhonemeClassifier(PreTrainedModel):
    """
    GOP classifier that wraps a pretrained Wav2Vec2ForCTC backbone.
    Computes per-phoneme scores using GOP-derived features + a small Transformer + classifier head.
    """

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

        # Special ids
        self.blank_id = config.pad_id
        self.unk_id = config.unk_id
        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.pad_id = self.blank_id

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

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.combined_feature_dim,
            nhead=config.gop_transformer_nhead,
            dim_feedforward=config.gop_transformer_dim_feedforward,
            dropout=config.gop_transformer_dropout,
            activation=F.relu,
            batch_first=True,
        )
        self.gop_transformer_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=config.gop_transformer_nlayers
        )

        head_label_config = getattr(config, "gop_head_labels", None)
        if head_label_config is None:
            if config.num_gop_labels is None:
                raise ValueError(
                    "Config must provide gop_head_labels or num_gop_labels for the classifier."
                )
            head_label_config = {"default": int(config.num_gop_labels)}
        self.head_label_config = {str(k): int(v) for k, v in head_label_config.items()}
        self.classifiers = nn.ModuleDict(
            {
                head: nn.Linear(self.combined_feature_dim, num_labels)
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
        """CTC log p(target|input) per item for a batch."""
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
        """Compute time dimension after backbone feature extractor."""

        def _conv_out_length(input_length, kernel_size, stride):
            return (
                torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1
            )

        for kernel_size, stride in zip(
            self.ctc_model.config.conv_kernel, self.ctc_model.config.conv_stride
        ):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        return input_lengths

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

        # 1) Backbone forward to get hidden states
        outputs = self.ctc_model.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state

        # 2) Frame-level logits for CTC
        logits_ctc = self.ctc_model.lm_head(hidden_states)
        log_probs_ctc = F.log_softmax(logits_ctc, dim=-1)
        log_probs_TNC = log_probs_ctc.permute(1, 0, 2).contiguous()

        # 3) Frame lengths
        batch_size = input_values.size(0)
        if attention_mask is None:
            input_lengths_frames = torch.full(
                (batch_size,), log_probs_TNC.size(0), dtype=torch.long, device=device
            )
        else:
            input_lengths_samples = attention_mask.sum(dim=-1)
            input_lengths_frames = self._get_feat_extract_output_lengths(
                input_lengths_samples
            )
            input_lengths_frames = torch.clamp(
                input_lengths_frames, max=log_probs_TNC.size(0)
            )

        # 4) GOP feature calculation over tokens
        max_token_len = (
            canonical_token_ids.size(1) if canonical_token_ids is not None else 0
        )
        batch_combined_features_list = [[] for _ in range(batch_size)]
        token_mask_bool = token_mask.to(device=device).bool()

        lpp_log_prob_batch = self._calculate_log_prob(
            log_probs_TNC, input_lengths_frames, canonical_token_ids, token_lengths
        )

        for token_idx in range(max_token_len):
            current_token_ids = canonical_token_ids[:, token_idx]
            current_token_embeddings = self.token_embedding(current_token_ids)
            token_out_of_bounds_mask = token_idx >= token_lengths
            mask_column = (
                token_mask_bool[:, token_idx]
                if token_mask_bool.dim() == 2
                else token_mask_bool
            )
            skip_mask = token_out_of_bounds_mask | ~mask_column

            if skip_mask.all():
                continue

            active_mask = ~skip_mask

            all_sub_log_probs = []
            if self.token_id_vocab:
                for sub_token_id in self.token_id_vocab:
                    sub_ids_batch = canonical_token_ids.clone()
                    sub_ids_batch[active_mask, token_idx] = sub_token_id
                    log_prob_sub_batch = self._calculate_log_prob(
                        log_probs_TNC,
                        input_lengths_frames,
                        sub_ids_batch,
                        token_lengths,
                    )
                    all_sub_log_probs.append(log_prob_sub_batch)

            if all_sub_log_probs:
                sub_lpr_batch = lpp_log_prob_batch.unsqueeze(1) - torch.stack(
                    all_sub_log_probs, dim=1
                )
                sub_lpr_batch = torch.nan_to_num(
                    sub_lpr_batch, nan=0.0, posinf=1e10, neginf=-1e10
                )
                if skip_mask.any():
                    sub_lpr_batch[skip_mask, :] = 0.0
            else:
                sub_lpr_batch = torch.zeros((batch_size, 0), device=device)

            # Deletion GOP component
            del_lpr_list = []
            for b_idx in range(batch_size):
                if skip_mask[b_idx]:
                    del_lpr_list.append(torch.tensor(-1e10, device=device))
                    continue
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
                    sub_lpr_batch,
                    del_lpr_batch.unsqueeze(1),
                ],
                dim=1,
            )
            combined_features = torch.cat([gop_part, current_token_embeddings], dim=1)
            for b_idx in range(batch_size):
                if active_mask[b_idx]:
                    batch_combined_features_list[b_idx].append(combined_features[b_idx])

        # 5) Pad phoneme feature sequences and mask
        feature_lengths_list = [
            len(seq_list) for seq_list in batch_combined_features_list
        ]
        if feature_lengths_list:
            feature_lengths = torch.tensor(
                feature_lengths_list, dtype=torch.long, device=device
            )
            target_pad_len = int(feature_lengths.max().item())
        else:
            feature_lengths = torch.zeros(
                (batch_size,), dtype=torch.long, device=device
            )
            target_pad_len = 0

        padded_sequences = []
        for seq_list in batch_combined_features_list:
            if seq_list:
                seq_tensor = torch.stack(seq_list, dim=0)
                pad_len = target_pad_len - seq_tensor.size(0)
                if pad_len > 0:
                    seq_tensor = F.pad(seq_tensor, (0, 0, 0, pad_len))
                padded_sequences.append(seq_tensor)
            else:
                padded_sequences.append(
                    torch.zeros(
                        (target_pad_len, self.combined_feature_dim), device=device
                    )
                )
        transformer_input = (
            torch.stack(padded_sequences, dim=0)
            if padded_sequences
            else torch.zeros((0, 0, self.combined_feature_dim), device=device)
        )
        transformer_padding_mask = (
            torch.arange(target_pad_len, device=device)[None, :]
            >= feature_lengths[:, None]
        )

        # 6) GOP transformer
        gop_transformer_output = self.gop_transformer_encoder(
            transformer_input, src_key_padding_mask=transformer_padding_mask
        )

        # 7) Per-phoneme classifier (multi-head)
        final_logits = {
            head: classifier(gop_transformer_output)
            for head, classifier in self.classifiers.items()
        }

        # 8) Loss
        loss = None
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                label_map = {next(iter(final_logits.keys())): labels}
            elif isinstance(labels, dict):
                label_map = labels
            else:
                raise TypeError(
                    "labels must be a Tensor or a dict of Tensors when provided."
                )

            active_mask = ~transformer_padding_mask.view(-1)
            for head, head_logits in final_logits.items():
                head_labels = label_map.get(head)
                if head_labels is None:
                    continue
                logits_flat = head_logits.view(-1, head_logits.size(-1))
                labels_flat = head_labels.view(-1)
                active_logits = logits_flat[active_mask]
                # breakpoint()
                active_labels = labels_flat[active_mask]
                if active_labels.numel() == 0:
                    continue
                head_loss = self.loss_fns[head](active_logits, active_labels)
                loss = head_loss if loss is None else loss + head_loss
            if loss is None:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

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
