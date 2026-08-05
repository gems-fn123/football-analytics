import pytest

from footy.stages.detect import Detector, resolve_device


def test_cpu_device_passes_through():
    assert resolve_device("cpu") == "cpu"


def test_cuda_falls_back_when_unavailable(monkeypatch):
    """configs/pipeline.yaml ships device: cuda. A CPU-only torch build must not crash."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda") == "cpu"


def test_cuda_honoured_when_available(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == "cuda"


def test_agpl_backend_is_refused_in_src():
    """docs/licensing.md: importing ultralytics into src/footy/ would relicense the package."""
    with pytest.raises(RuntimeError, match="AGPL-3.0"):
        Detector({"name": "ultralytics_yolo"}).setup()


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown detector backend"):
        Detector({"name": "detectron9"}).setup()


def test_missing_local_weights_gives_actionable_error():
    with pytest.raises(FileNotFoundError, match="rtdetr_coco"):
        Detector({"name": "rfdetr", "weights": "models/weights/does_not_exist.pt"}).setup()
