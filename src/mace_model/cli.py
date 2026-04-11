from __future__ import annotations

import argparse
from pathlib import Path

from .build import build_initial_model, save_initialized_model, summarize_build
from .config import DEFAULT_CONFIG_TOML, load_build_request


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mace-model-init",
        description="Construct an initialized MACE model from a config file.",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to a TOML or JSON model config.",
    )
    parser.add_argument(
        "-b",
        "--backend",
        choices=("torch", "jax"),
        help="Optional backend override. By default the backend comes from the config.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output path override. By default the output comes from the config.",
    )
    parser.add_argument(
        "--print-example-config",
        action="store_true",
        help="Print an example TOML config and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)

    if args.print_example_config:
        print(DEFAULT_CONFIG_TOML.rstrip())
        return 0

    if not args.config:
        parser.error("--config is required unless --print-example-config is used.")

    request = load_build_request(
        args.config,
        backend_override=args.backend,
        output_override=args.output,
    )
    result = build_initial_model(request)
    summary = summarize_build(result)

    print(f"backend: {summary['backend']}")
    print(f"model_class: {summary['model_class']}")
    print(f"parameters: {summary['parameters']}")
    print(f"atomic_numbers: {summary['atomic_numbers']}")
    print(f"num_interactions: {summary['num_interactions']}")

    if request.output:
        written = save_initialized_model(result, request.output)
        print("written:")
        for path in written:
            print(f"  {Path(path)}")
    else:
        print("written: none")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
