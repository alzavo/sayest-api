import argparse
import json
from pathlib import Path


DEFAULT_OUTPUT = Path("app/constants/phonemes.py")
SPECIAL_TOKEN_KEYS = ("pad_token", "unk_token", "bos_token", "eos_token")
EXCLUDED_TOKENS = set()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_vocab_path(model_path: Path) -> Path:
    return model_path / "vocab.json" if model_path.is_dir() else model_path


def collect_special_tokens(model_dir: Path) -> set[str]:
    special_tokens = set(EXCLUDED_TOKENS)

    special_tokens_map_path = model_dir / "special_tokens_map.json"
    if special_tokens_map_path.exists():
        special_tokens_map = load_json(special_tokens_map_path)
        for key in SPECIAL_TOKEN_KEYS:
            token_info = special_tokens_map.get(key)
            if isinstance(token_info, dict):
                content = token_info.get("content")
                if content:
                    special_tokens.add(content)
            elif isinstance(token_info, str):
                special_tokens.add(token_info)

    added_tokens_path = model_dir / "added_tokens.json"
    if added_tokens_path.exists():
        special_tokens.update(load_json(added_tokens_path).keys())

    return special_tokens


def extract_phonemes(vocab_path: Path) -> list[str]:
    vocab = load_json(vocab_path)
    model_dir = vocab_path.parent
    excluded_tokens = collect_special_tokens(model_dir)

    phoneme_items = [
        (token, token_id)
        for token, token_id in vocab.items()
        if token not in excluded_tokens and not token.startswith("[")
    ]
    phoneme_items.sort(key=lambda item: item[1])
    return [token for token, _ in phoneme_items]


def render_phonemes_module(phonemes: list[str]) -> str:
    lines = ["ALL_PHONEMES = {"]
    lines.extend(f'    "{phoneme}",' for phoneme in phonemes)
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate app/constants/phonemes.py from a model vocab.json."
    )
    parser.add_argument(
        "model_path",
        help="Path to a model directory containing vocab.json, or directly to vocab.json.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to the generated phonemes.py file.",
    )
    args = parser.parse_args()

    vocab_path = resolve_vocab_path(Path(args.model_path))
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    phonemes = extract_phonemes(vocab_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_phonemes_module(phonemes), encoding="utf-8")


if __name__ == "__main__":
    main()
