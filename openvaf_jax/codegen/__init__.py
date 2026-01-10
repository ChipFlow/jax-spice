"""Code generation for OpenVAF to JAX translation.

This module provides:
- context: Code generation context (variable tracking, constants)
- instruction: Single instruction translation
- function_builder: Complete function assembly (init/eval)
"""

from .context import CodeGenContext
from .function_builder import EvalFunctionBuilder, InitFunctionBuilder
from .instruction import InstructionTranslator

__all__ = [
    "CodeGenContext",
    "InstructionTranslator",
    "InitFunctionBuilder",
    "EvalFunctionBuilder",
]
