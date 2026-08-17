"""Pytest configuration for the classify-twse-query test suite.

The production ``.env`` carries a real FinMind API token. The unit test
``test_finmind_client.py::test_init_with_defaults`` instantiates
``FinMindClient()`` with no arguments and asserts the resolved token equals the
placeholder ``"test_token_for_testing"``.

Instead of hard-coding the real token or editing ``.env``, we monkeypatch the
resolved ``settings`` singleton so every test runs against the placeholder
token. This isolates the suite from the secret (the real token is never read by
the test process) and keeps the assertion meaningful.
"""

import pytest

from classifier.config import settings


@pytest.fixture(autouse=True)
def _isolate_finmind_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``settings.finmind_api_token`` to the test placeholder.

    ``FinMindClient()`` falls back to ``settings.finmind_api_token`` when no
    explicit token is passed; this makes that fallback deterministic for the
    suite. Function-scoped, so the patch is restored after each test.
    """
    monkeypatch.setattr(settings, "finmind_api_token", "test_token_for_testing")
