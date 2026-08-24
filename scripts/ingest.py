"""Thin CLI wrappers. All logic lives in cq/."""

from __future__ import annotations

import argparse
from pathlib import Path

from cq.data.gate import main as archive_gate_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data ingest/gate helpers")
    parser.add_argument("command", choices=["gate"])
    args = parser.parse_args()
    if args.command == "gate":
        archive_gate_main()
        return
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
