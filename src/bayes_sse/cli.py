from __future__ import annotations

import argparse
from collections.abc import Sequence

from bayes_sse.simulations.cli import main as simulations_main


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bayes-sse",
        description="Command line tools for BayesSSE",
        usage="bayes-sse <COMMAND>",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        required=True,
    )

    subparsers.add_parser(
        "simulate",
        help="Run simulation scenarios.",
        description="Run BayesSSE simulation scenarios.",
    )

    args, remaining = parser.parse_known_args(argv)

    if args.command == "simulate":
        return simulations_main(remaining)

    parser.error(f"unknown command: {args.command}")
