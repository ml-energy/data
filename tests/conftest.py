from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--run-bulk",
        action="store_true",
        default=False,
        help="Run doc snippets that download bulk raw files from HF Hub.",
    )
