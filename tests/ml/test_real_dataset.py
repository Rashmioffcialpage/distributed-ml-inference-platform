import torch
from model import PerceptionCNN
from real_dataset import CLASS_NAMES, _label_for


def test_label_for_known_prefixes():
    assert _label_for("regulatory-prohibitory-B1_123.jpg") == "regulatory"
    assert _label_for("NIC-backwards-additional_sign_123.jpg") == "NIC"
    assert _label_for("danger-slippery-A1_123.jpg") == "danger"


def test_label_for_unknown_prefix_returns_none():
    assert _label_for("unrecognized_prefix_123.jpg") is None


def test_perception_cnn_handles_rgb_64px_input():
    model = PerceptionCNN(num_classes=len(CLASS_NAMES), in_channels=3, image_size=64)
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    assert out.shape == (4, len(CLASS_NAMES))


def test_perception_cnn_default_still_matches_synthetic_task():
    # Backward-compat check: no-arg construction must still match the
    # original 1-channel/32px/3-class synthetic perception task.
    model = PerceptionCNN()
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    assert out.shape == (2, 3)
