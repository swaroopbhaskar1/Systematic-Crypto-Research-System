"""Live execution is outside this research system's scope."""

from typing import NoReturn


def execute_live(*_args: object, **_kwargs: object) -> NoReturn:
    """Reject every attempt to perform live execution."""
    raise NotImplementedError("live execution is intentionally not implemented")
