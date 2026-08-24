"""Thin CLI wrappers. All logic lives in cq/."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from cq.data.gate import main as archive_gate_main
from cq.data.ingest_store import main as archive_store_main


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run data ingest/gate helpers")
    parser.add_argument("command", choices=["gate", "store"])
    # REMAINDER hands every later token to the subcommand untouched, so
    # `ingest.py store --help` reaches the store parser instead of this one.
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to the subcommand",
    )
    parsed = parser.parse_args(argv)
    if parsed.command == "gate":
        return archive_gate_main(parsed.args)
    return archive_store_main(parsed.args)


if __name__ == "__main__":
    raise SystemExit(main())
