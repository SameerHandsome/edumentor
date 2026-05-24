"""
Evaluation conftest — registers the 'eval' marker so pytest doesn't warn.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "eval: LLM-as-judge evaluation tests (require Groq API key for live runs)",
    )
