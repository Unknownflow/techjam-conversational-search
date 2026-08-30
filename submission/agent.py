"""Official entry point exporting the submission Agent."""

try:  # Imported as ``submission.agent`` from the repository root.
    from .src.agent import Agent
except ImportError:  # Imported as top-level ``agent`` from the extracted bundle.
    from src.agent import Agent

__all__ = ["Agent"]
