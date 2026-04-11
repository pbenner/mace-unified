from __future__ import annotations

import argparse
from pathlib import Path

from .foundation import (
    SUPPORTED_FOUNDATION_SOURCES,
    download_foundation_model,
    save_foundation_model,
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mace-model-foundation",
        description="Download an upstream MACE foundation model and convert it to the local Torch or JAX backend.",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=("torch", "jax"),
        help="Target backend for the exported local model.",
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=SUPPORTED_FOUNDATION_SOURCES,
        help="Foundation source to download from.",
    )
    parser.add_argument(
        "--model",
        help="Optional foundation variant or model path, depending on the source.",
    )
    parser.add_argument(
        "--head",
        help="Optional head to select from a multi-head foundation model before conversion.",
    )
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to '<source>-<model>-<backend>'.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used for the upstream Torch model download/conversion.",
    )
    parser.add_argument(
        "--dtype",
        help="Optional upstream loader dtype override, for example 'float32' or 'float64'.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)

    result = download_foundation_model(
        backend=args.backend,
        source=args.source,
        model=args.model,
        head=args.head,
        device=args.device,
        default_dtype=args.dtype,
    )
    written = save_foundation_model(
        result,
        args.output,
        source=args.source,
        model=args.model,
    )

    print(f"backend: {result.backend}")
    print(f"foundation_source: {args.source}")
    print(f"foundation_model: {args.model or 'default'}")
    print(f"model_class: {result.model_class}")
    print(f"atomic_numbers: {result.normalized_model_config.get('atomic_numbers', [])}")
    print(f"num_interactions: {result.normalized_model_config.get('num_interactions')}")
    print("written:")
    for path in written:
        print(f"  {Path(path)}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
