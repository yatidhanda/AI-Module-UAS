import sys
import types

from garuda_edge.perception.detector import resolve_device


def _fake_torch(cuda: bool, mps: bool):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
    )


def test_explicit_device_is_kept():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch):
    for cuda, mps, expected in ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda, mps))
        assert resolve_device("auto") == expected
        assert resolve_device(None) == expected
