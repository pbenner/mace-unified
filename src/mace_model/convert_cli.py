from __future__ import annotations

import argparse
from pathlib import Path

from .conversion import (
    convert_torch_model,
    load_serialized_torch_model,
    save_converted_model,
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mace-model-convert",
        description="Convert a Torch MACE model checkpoint or bundle to a local JAX model bundle.",
    )
    parser.add_argument(
        "torch_model",
        help="Path to a Torch model file or a local Torch model directory containing config.json and state_dict.pt.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path for the converted JAX bundle. Defaults to '<input>-jax'.",
    )
    parser.add_argument(
        "--head",
        help="Optional head to select from a multi-head Torch model before conversion.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used if an intermediate legacy Torch conversion is required.",
    )
    return parser


def _default_output_path(torch_model_arg: str) -> Path:
    path = Path(torch_model_arg).expanduser().resolve()
    if path.is_dir():
        return path.with_name(f"{path.name}-jax")
    if path.suffix:
        return path.with_suffix(".json")
    return path.with_name(f"{path.name}-jax")


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)

    torch_model, normalized = load_serialized_torch_model(args.torch_model)
    result = convert_torch_model(
        torch_model,
        backend="jax",
        head=args.head,
        device=args.device,
        config=normalized,
    )
    output = (
        args.output
        if args.output is not None
        else _default_output_path(args.torch_model)
    )
    written = save_converted_model(result, output)

    print("backend: jax")
    print(f"model_class: {result.model_class}")
    print(f"atomic_numbers: {result.normalized_model_config.get('atomic_numbers', [])}")
    print(f"num_interactions: {result.normalized_model_config.get('num_interactions')}")
    print("written:")
    for path in written:
        print(f"  {Path(path)}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
