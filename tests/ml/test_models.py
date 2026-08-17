import torch
from event_classifier import EventClassifier, NUM_CLASSES, NUM_FEATURES


def test_event_classifier_output_shape():
    model = EventClassifier()
    x = torch.randn(8, NUM_FEATURES)
    out = model(x)
    assert out.shape == (8, NUM_CLASSES)


def test_event_classifier_is_trainable():
    model = EventClassifier()
    x = torch.randn(4, NUM_FEATURES)
    y = torch.tensor([0, 1, 2, 3])
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    before = [p.clone() for p in model.parameters()]
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    after = list(model.parameters())

    assert any(not torch.equal(b, a) for b, a in zip(before, after))
