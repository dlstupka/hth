import contextlib
import sys
import types
import unittest
from unittest import mock

from hth.doc_ufcn_compat import use_modern_torch_autocast


class DocUFCNAutocastCompatTests(unittest.TestCase):
    def test_routes_doc_ufcn_alias_to_torch_amp_autocast_cuda(self):
        calls = []

        @contextlib.contextmanager
        def modern_autocast(device_type, *args, **kwargs):
            calls.append((device_type, args, kwargs))
            yield

        fake_torch = types.ModuleType("torch")
        fake_torch.amp = types.SimpleNamespace(autocast=modern_autocast)

        fake_package = types.ModuleType("doc_ufcn")
        fake_package.__path__ = []
        fake_model = types.ModuleType("doc_ufcn.model")

        def deprecated_autocast(*args, **kwargs):
            raise AssertionError("deprecated torch.cuda.amp.autocast alias was used")

        fake_model.autocast = deprecated_autocast

        modules = {
            "torch": fake_torch,
            "doc_ufcn": fake_package,
            "doc_ufcn.model": fake_model,
        }
        with mock.patch.dict(sys.modules, modules, clear=False):
            self.assertTrue(use_modern_torch_autocast())
            with fake_model.autocast(enabled=False):
                pass

        self.assertEqual(calls, [("cuda", (), {"enabled": False})])

    def test_requires_current_torch_autocast_api(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.amp = types.SimpleNamespace()
        with mock.patch.dict(sys.modules, {"torch": fake_torch}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "torch.amp.autocast"):
                use_modern_torch_autocast()


if __name__ == "__main__":
    unittest.main()
