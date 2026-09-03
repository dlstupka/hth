"""Run legacy function-style tests under the project's unittest command.

The project standard is ``python -m unittest discover -s tests -v``.  A small
number of older test modules use plain ``test_*`` functions, which unittest
does not discover.  This adapter keeps those behavioral tests active while
they are migrated naturally when their modules next change.

Only the fixture shapes already present in this suite are supported.  The
audit test deliberately fails if a new implicit fixture or unsupported test
signature is introduced.
"""

from __future__ import annotations

import importlib
import inspect
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


TEST_ROOT = Path(__file__).resolve().parent
THIS_MODULE = Path(__file__).stem
SUPPORTED_FIXTURES = frozenset({"tmp_path", "monkeypatch"})


class _MonkeyPatch:
    """Minimal unittest-backed replacement for the fixture used by two tests."""

    def __init__(self) -> None:
        self._patchers: list[object] = []

    def setattr(self, target: str, value: object) -> None:
        patcher = patch(target, value)
        patcher.start()
        self._patchers.append(patcher)

    def close(self) -> None:
        for patcher in reversed(self._patchers):
            patcher.stop()
        self._patchers.clear()


def _function_tests() -> list[tuple[ModuleType, object]]:
    discovered: list[tuple[ModuleType, object]] = []
    for path in sorted(TEST_ROOT.glob("test_*.py")):
        if path.stem == THIS_MODULE:
            continue
        module = importlib.import_module(f"tests.{path.stem}")
        for name, candidate in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_") and candidate.__module__ == module.__name__:
                discovered.append((module, candidate))
    return discovered


FUNCTION_TESTS = _function_tests()


def _run_function_test(function: object) -> None:
    parameters = tuple(inspect.signature(function).parameters)
    unsupported = set(parameters) - SUPPORTED_FIXTURES
    if unsupported:
        raise TypeError(
            f"Unsupported unittest function fixture(s) for {function.__module__}."
            f"{function.__name__}: {sorted(unsupported)}"
        )

    with tempfile.TemporaryDirectory() as temporary:
        monkeypatch = _MonkeyPatch()
        fixtures = {
            "tmp_path": Path(temporary),
            "monkeypatch": monkeypatch,
        }
        try:
            function(**{name: fixtures[name] for name in parameters})
        finally:
            monkeypatch.close()


class FunctionStyleUnittestTests(unittest.TestCase):
    def test_all_function_style_tests_use_supported_fixtures(self) -> None:
        unsupported = []
        for _, function in FUNCTION_TESTS:
            parameters = set(inspect.signature(function).parameters)
            if parameters - SUPPORTED_FIXTURES:
                unsupported.append(
                    f"{function.__module__}.{function.__name__}: "
                    f"{sorted(parameters - SUPPORTED_FIXTURES)}"
                )
        self.assertEqual(unsupported, [])


def _add_function_test(module: ModuleType, function: object) -> None:
    def test(self: unittest.TestCase) -> None:
        _run_function_test(function)

    test.__name__ = f"test_{module.__name__}__{function.__name__}"
    test.__qualname__ = f"FunctionStyleUnittestTests.{test.__name__}"
    test.__doc__ = function.__doc__
    setattr(FunctionStyleUnittestTests, test.__name__, test)


for _module, _function in FUNCTION_TESTS:
    _add_function_test(_module, _function)


if __name__ == "__main__":
    unittest.main()
