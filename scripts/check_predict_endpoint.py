#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send an audio file to the local /predict endpoint."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/predict",
        help="Predict endpoint URL.",
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to the audio file to upload.",
    )
    parser.add_argument(
        "--phonemes",
        required=True,
        help="Space-separated phonemes, for example: 'a i'.",
    )
    parser.add_argument(
        "--word",
        default=None,
        help="Optional word label sent with the request.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise SystemExit(f"Audio file not found: {audio_path}")

    data = {"phonemes": args.phonemes}
    if args.word is not None:
        data["word"] = args.word

    with audio_path.open("rb") as audio_file:
        response = requests.post(
            args.url,
            data=data,
            files={"audio": (audio_path.name, audio_file, "audio/wav")},
            timeout=args.timeout,
        )

    print(f"HTTP {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2, ensure_ascii=True))
    except ValueError:
        print(response.text)

    return 0 if response.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
