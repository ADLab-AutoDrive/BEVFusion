import pytest
import torch

from mmdet.models import Accuracy, build_loss


def test_ce_loss():
    # use_mask and use_sigmoid cannot be true at the same time
    with pytest.raises(AssertionError):
        loss_cfg = dict(
            type='CrossEntropyLoss',
            use_mask=True,
            use_sigmoid=True,
            loss_weight=1.0)
        build_loss(loss_cfg)

    # test loss with class weights
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        class_weight=[0.8, 0.2],
        loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    fake_pred = torch.Tensor([[100, -100]])
    fake_label = torch.Tensor([1]).long()
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(40.))

    loss_cls_cfg = dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(200.))


def test_varifocal_loss():
    # only sigmoid version of VarifocalLoss is implemented
    with pytest.raises(AssertionError):
        loss_cfg = dict(
            type='VarifocalLoss', use_sigmoid=False, loss_weight=1.0)
        build_loss(loss_cfg)

    # test that alpha should be greater than 0
    with pytest.raises(AssertionError):
        loss_cfg = dict(
            type='VarifocalLoss',
            alpha=-0.75,
            gamma=2.0,
            use_sigmoid=True,
            loss_weight=1.0)
        build_loss(loss_cfg)

    # test that pred and target should be of the same size
    loss_cls_cfg = dict(
        type='VarifocalLoss',
        use_sigmoid=True,
        alpha=0.75,
        gamma=2.0,
        iou_weighted=True,
        reduction='mean',
        loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    with pytest.raises(AssertionError):
        fake_pred = torch.Tensor([[100.0, -100.0]])
        fake_target = torch.Tensor([[1.0]])
        loss_cls(fake_pred, fake_target)

    # test the calculation
    loss_cls = build_loss(loss_cls_cfg)
    fake_pred = torch.Tensor([[100.0, -100.0]])
    fake_target = torch.Tensor([[1.0, 0.0]])
    assert torch.allclose(loss_cls(fake_pred, fake_target), torch.tensor(0.0))

    # test the loss with weights
    loss_cls = build_loss(loss_cls_cfg)
    fake_pred = torch.Tensor([[0.0, 100.0]])
    fake_target = torch.Tensor([[1.0, 1.0]])
    fake_weight = torch.Tensor([0.0, 1.0])
    assert torch.allclose(
        loss_cls(fake_pred, fake_target, fake_weight), torch.tensor(0.0))


def test_kd_loss():
    # test that temeprature should be greater than 1
    with pytest.raises(AssertionError):
        loss_cfg = dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=1.0, T=0.5)
        build_loss(loss_cfg)

    # test that pred and target should be of the same size
    loss_cls_cfg = dict(
        type='KnowledgeDistillationKLDivLoss', loss_weight=1.0, T=1)
    loss_cls = build_loss(loss_cls_cfg)
    with pytest.raises(AssertionError):
        fake_pred = torch.Tensor([[100, -100]])
        fake_label = torch.Tensor([1]).long()
        loss_cls(fake_pred, fake_label)

    # test the calculation
    loss_cls = build_loss(loss_cls_cfg)
    fake_pred = torch.Tensor([[100.0, 100.0]])
    fake_target = torch.Tensor([[1.0, 1.0]])
    assert torch.allclose(loss_cls(fake_pred, fake_target), torch.tensor(0.0))

    # test the loss with weights
    loss_cls = build_loss(loss_cls_cfg)
    fake_pred = torch.Tensor([[100.0, -100.0], [100.0, 100.0]])
    fake_target = torch.Tensor([[1.0, 0.0], [1.0, 1.0]])
    fake_weight = torch.Tensor([0.0, 1.0])
    assert torch.allclose(
        loss_cls(fake_pred, fake_target, fake_weight), torch.tensor(0.0))


def test_accuracy():
    # test for empty pred
    pred = torch.empty(0, 4)
    label = torch.empty(0)
    accuracy = Accuracy(topk=1)
    acc = accuracy(pred, label)
    assert acc.item() == 0

    pred = torch.Tensor([[0.2, 0.3, 0.6, 0.5], [0.1, 0.1, 0.2, 0.6],
                         [0.9, 0.0, 0.0, 0.1], [0.4, 0.7, 0.1, 0.1],
                         [0.0, 0.0, 0.99, 0]])
    # test for top1
    true_label = torch.Tensor([2, 3, 0, 1, 2]).long()
    accuracy = Accuracy(topk=1)
    acc = accuracy(pred, true_label)
    assert acc.item() == 100

    # test for top1 with score thresh=0.8
    true_label = torch.Tensor([2, 3, 0, 1, 2]).long()
    accuracy = Accuracy(topk=1, thresh=0.8)
    acc = accuracy(pred, true_label)
    assert acc.item() == 40

    # test for top2
    accuracy = Accuracy(topk=2)
    label = torch.Tensor([3, 2, 0, 0, 2]).long()
    acc = accuracy(pred, label)
    assert acc.item() == 100

    # test for both top1 and top2
    accuracy = Accuracy(topk=(1, 2))
    true_label = torch.Tensor([2, 3, 0, 1, 2]).long()
    acc = accuracy(pred, true_label)
    for a in acc:
        assert a.item() == 100

    # topk is larger than pred class number
    with pytest.raises(AssertionError):
        accuracy = Accuracy(topk=5)
        accuracy(pred, true_label)

    # wrong topk type
    with pytest.raises(AssertionError):
        accuracy = Accuracy(topk='wrong type')
        accuracy(pred, true_label)

    # label size is larger than required
    with pytest.raises(AssertionError):
        label = torch.Tensor([2, 3, 0, 1, 2, 0]).long()  # size mismatch
        accuracy = Accuracy()
        accuracy(pred, label)

    # wrong pred dimension
    with pytest.raises(AssertionError):
        accuracy = Accuracy()
        accuracy(pred[:, :, None], true_label)
