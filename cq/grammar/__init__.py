"""Safe, point-in-time research grammar."""

import sys

from cq.grammar import compile as _compiler
from cq.grammar.ast import Expression, GrammarError
from cq.grammar.compile import CompiledSignal, compile_signal
from cq.grammar.ops import evaluate
from cq.grammar.parser import parse_expression

sys.modules[f"{__name__}.compiler"] = _compiler

__all__ = [
    "CompiledSignal",
    "Expression",
    "GrammarError",
    "compile_signal",
    "evaluate",
    "parse_expression",
]
