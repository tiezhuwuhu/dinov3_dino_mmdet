# Copyright (c) OpenMMLab. All rights reserved.
"""Pure PyTorch target tests; runnable even without MMCV compiled operators."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# The auxiliary implementation deliberately has no MMDetection dependencies.
# Load this file alone so target tests do not require unrelated MMCV ops.
ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    'point_fidt_aux_head',
    ROOT / 'mmdet/models/dense_heads/point_fidt_aux_head.py')
FIDT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIDT)
PointFIDTAuxHead = FIDT.PointFIDTAuxHead
generate = FIDT.generate_fidt_target


def test_stage4_config():
    mmengine = pytest.importorskip('mmengine')
    stage3 = mmengine.Config.fromfile(
        str(ROOT / 'configs/dino/point_dino_stage3_lam020_cost20.py'))
    stage4 = mmengine.Config.fromfile(str(
        ROOT / 'configs/dino/'
        'point_dino_r50_shanghaitech_stage4_fidt_native_12e.py'))
    assert stage4.model.bbox_head.point_euclidean_weight == 0.20
    assert stage4.model.bbox_head.loss_bbox.loss_weight == 5.0
    assert stage4.model.train_cfg.assigner.match_costs == [
        dict(type='FocalLossCost', weight=2.0),
        dict(type='PointL1Cost', weight=20.0)]
    assert stage4.model.dn_cfg.point_noise_scale == 0.01
    fidt = stage4.model.point_fidt_head
    assert fidt.enabled and fidt.fidt_loss_weight == 1.0
    assert (fidt.gamma, fidt.phi, fidt.xi) == (0.02, 0.75, 1.0)
    assert fidt.upsample_factor == 2
    # Apart from the new branch, the complete resolved model stays Stage 3.
    original_model = stage4.model.copy()
    original_model.pop('point_fidt_head')
    assert original_model == stage3.model
    for field in ('train_dataloader', 'val_dataloader', 'test_dataloader',
                  'val_evaluator', 'test_evaluator', 'optim_wrapper'):
        assert stage4[field] == stage3[field]
    assert stage4.train_pipeline[2].scale == (1024, 768)
    print('lambda_euc=0.20; PointL1 training weight=5; matcher=2/20; '
          'DN noise=0.01; FIDT:', dict(fidt))


def test_formula_and_single_gt():
    distance = torch.tensor([0., 0.125, 1., 4., 8., 50., 100., 1500.],
                            requires_grad=True)
    actual = FIDT.distance_to_fidt(distance)
    expected = 1 / (distance.double().pow(
        0.02 * distance.double() + 0.75) + 1)
    torch.testing.assert_close(actual, expected.float(), rtol=1e-5, atol=1e-7)
    assert actual[0] == 1
    assert torch.all(actual[:-1] >= actual[1:])
    assert torch.isfinite(actual).all() and not actual.requires_grad

    # Grid centers 4,12,...,100,... give an exact D=0 at (100,100).
    points = torch.tensor([[100., 100.]], requires_grad=True)
    target, valid = generate(points, (50, 50), (400, 400), (400, 400))
    assert valid.all() and target[12, 12] == target.max() == 1
    assert target[12, 12] > target[12, 13] > target[12, 14]
    assert not target.requires_grad and points.grad is None


def test_two_gt_use_nearest_not_sum():
    points = torch.tensor([[100., 100.], [200., 100.]])
    args = ((50, 50), (400, 400), (400, 400))
    combined, _ = generate(points, *args)
    first, _ = generate(points[:1], *args)
    second, _ = generate(points[1:], *args)
    torch.testing.assert_close(combined, torch.maximum(first, second))
    assert not torch.allclose(combined, first + second)
    duplicate, _ = generate(points.repeat(2, 1), *args)
    torch.testing.assert_close(combined, duplicate)


def test_subpixel_anisotropic_grid_and_bounded_chunks(monkeypatch):
    torch.manual_seed(4)
    points = torch.rand(300, 2) * torch.tensor([20., 9.])
    points[0] = torch.tensor([2.25, 1.25])
    shapes = []
    original_cdist = torch.cdist

    def tracked_cdist(x, y, **kwargs):
        shapes.append((len(x), len(y)))
        return original_cdist(x, y, **kwargs)

    monkeypatch.setattr(torch, 'cdist', tracked_cdist)
    target, valid = generate(points, (3, 5), (9, 20), (8, 17),
                             chunk_size=4, point_chunk_size=32)
    ys, xs = torch.meshgrid(torch.tensor([1.5, 4.5, 7.5]),
                            torch.tensor([2., 6., 10., 14., 18.]),
                            indexing='ij')
    grid = torch.stack((xs, ys), -1)
    # Independent broadcast reference, safe at this tiny test size.
    distance = ((grid[:, :, None] - points).square().sum(-1)).sqrt().min(-1)[0]
    expected = 1 / (distance.pow(0.02 * distance + 0.75) + 1)
    torch.testing.assert_close(target[valid], expected[valid])
    assert shapes and max(x for x, _ in shapes) <= 4
    assert max(y for _, y in shapes) <= 32
    assert not valid[:, -1].any() and valid[:, :-1].all()
    fractional, _ = generate(points[:1], (3, 5), (9, 20), (9, 20))
    rounded, _ = generate(points[:1].round(), (3, 5), (9, 20), (9, 20))
    assert not torch.allclose(fractional, rounded)


def test_empty_gt_and_padding_loss():
    empty = torch.empty(0, 2)
    target, valid = generate(empty, (3, 5), (9, 20), (6, 13))
    assert torch.count_nonzero(target) == 0
    assert valid.sum() == 6  # x=2,6,10; y=1.5,4.5
    prediction = torch.full_like(target, 0.5, requires_grad=True)
    raw = PointFIDTAuxHead.masked_mse(prediction, target, valid)
    assert torch.isfinite(raw) and raw == 0.25
    changed_padding = prediction.detach().clone()
    changed_padding[~valid] = 10000
    assert PointFIDTAuxHead.masked_mse(changed_padding, target, valid) == raw
    raw.backward()
    assert prediction.grad[~valid].count_nonzero() == 0
    assert prediction.grad[valid].count_nonzero() == valid.sum()
    zero_valid = torch.zeros_like(valid)
    assert PointFIDTAuxHead.masked_mse(prediction, target, zero_valid) == 0


@pytest.mark.parametrize('upsample_factor', [1, 2])
def test_auxiliary_forward_backward_and_debug(upsample_factor):
    torch.manual_seed(3)
    head = PointFIDTAuxHead(
        in_channels=8, hidden_channels=8, num_groups=2,
        upsample_factor=upsample_factor, fidt_loss_weight=0.3, debug=True)
    feature = torch.randn(2, 8, 4, 5, requires_grad=True)
    original = feature.detach().clone()
    samples = [
        SimpleNamespace(
            metainfo=dict(img_shape=(24, 31), batch_input_shape=(32, 40),
                          pad_shape=(24, 31)),
            gt_instances=SimpleNamespace(points=torch.tensor([[10.25, 8.5]]))),
        SimpleNamespace(
            metainfo=dict(img_shape=(32, 40), batch_input_shape=(32, 40),
                          pad_shape=(32, 40)),
            gt_instances=SimpleNamespace(points=torch.empty(0, 2)))
    ]
    prediction = head(feature)
    assert prediction.shape == (2, 1, 4 * upsample_factor, 5 * upsample_factor)
    assert ((prediction > 0) & (prediction < 1)).all()
    losses = head.loss(feature, samples, (32, 40))
    assert set(losses) == {'loss_fidt', 'fidt_mse'}
    assert not losses['fidt_mse'].requires_grad
    torch.testing.assert_close(losses['loss_fidt'], losses['fidt_mse'] * 0.3)
    losses['loss_fidt'].backward()
    assert torch.isfinite(feature.grad).all() and feature.grad.abs().sum() > 0
    assert head.output.weight.grad.abs().sum() > 0
    torch.testing.assert_close(feature.detach(), original, rtol=0, atol=0)
    head.debug = False
    assert set(head.loss(feature, samples, (32, 40))) == {'loss_fidt'}
    samples[0].metainfo['batch_input_shape'] = (31, 40)
    with pytest.raises(ValueError, match='batch_input_shape'):
        head.loss(feature, samples, (32, 40))


def test_native_auxiliary_output_shape():
    head = PointFIDTAuxHead(upsample_factor=2)
    with torch.no_grad():
        output = head(torch.randn(1, 256, 96, 128))
    assert output.shape == (1, 1, 192, 256)


def test_target_under_autocast():
    points = torch.tensor([[100.125, 100.25]])
    expected, _ = generate(points, (50, 50), (400, 400), (400, 400))
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        actual, _ = generate(points, (50, 50), (400, 400), (400, 400))
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_local_config_keeps_full_baseline_and_stage2_init():
    mmengine = pytest.importorskip('mmengine')
    folder = ROOT / 'configs/dino'
    full = mmengine.Config.fromfile(str(
        folder / 'point_dino_r50_shanghaitech_stage4_fidt_native_12e.py'))
    local = mmengine.Config.fromfile(str(
        folder / 'point_dino_r50_shanghaitech_stage4_local_fidt_native_12e.py'))
    assert local.model.point_fidt_head.local_radius_px == 16.0
    assert local.model.point_fidt_head.fidt_loss_weight == 1.0
    assert local.model.point_fidt_head.debug
    assert local.load_from == full.load_from
    assert local.load_from.endswith('/point_dino_stage2_step2_init.pth')
    resolved = local.to_dict()
    resolved['model']['point_fidt_head'].pop('local_radius_px')
    resolved['work_dir'] = full.work_dir
    assert resolved == full.to_dict()


def _pixel_grid(map_shape, padded_shape):
    h, w = map_shape
    pad_h, pad_w = padded_shape
    y, x = torch.meshgrid((torch.arange(h) + 0.5) * pad_h / h,
                          (torch.arange(w) + 0.5) * pad_w / w,
                          indexing='ij')
    return torch.stack((x, y), dim=-1)


def test_local_single_gt_radius_in_pixels_and_unchanged_target():
    points = torch.tensor([[100., 100.]])
    args = ((50, 50), (400, 400), (400, 400))
    full_target, valid = generate(points, *args)
    target, mask = generate(points, *args, local_radius_px=16)
    distance = (_pixel_grid(*args[:2]) - points[0]).norm(dim=-1)
    assert torch.equal(mask, valid & (distance <= 16))
    assert mask[12, 14]  # center (116,100): boundary D=16 is included
    assert not mask[12, 15]  # center (124,100): D=24 is excluded
    torch.testing.assert_close(target, full_target, rtol=0, atol=0)
    assert target[12, 15] > 0  # Outside is ignored, not zeroed/rescaled.


@pytest.mark.parametrize('second_x', [200., 116.])
def test_local_multiple_gt_union_and_overlap_once(second_x):
    points = torch.tensor([[100., 100.], [second_x, 100.]])
    args = ((50, 50), (400, 400), (400, 400))
    target, mask = generate(points, *args, local_radius_px=16)
    grid = _pixel_grid(*args[:2])
    distances = (grid[:, :, None] - points).norm(dim=-1)
    circles = distances <= 16
    assert mask.dtype == torch.bool
    assert torch.equal(mask, circles.any(dim=-1))
    if second_x == 116:
        assert mask.sum() < circles.sum()  # Overlap counted only once.
    nearest = distances.min(dim=-1).values
    expected = 1 / (nearest.pow(0.02 * nearest + 0.75) + 1)
    torch.testing.assert_close(target, expected, rtol=1e-5, atol=1e-7)
    duplicated, duplicate_mask = generate(
        points.repeat(2, 1), *args, local_radius_px=16)
    torch.testing.assert_close(duplicated, target, rtol=0, atol=0)
    assert torch.equal(duplicate_mask, mask)


def test_local_padding_and_no_extra_cdist(monkeypatch):
    points = torch.tensor([[5.75, 3.75], [1.25, 1.25]])
    args = ((4, 4), (8, 8), (6, 6))
    calls = []
    original = torch.cdist

    def tracked(x, y, **kwargs):
        calls.append((len(x), len(y)))
        return original(x, y, **kwargs)

    monkeypatch.setattr(torch, 'cdist', tracked)
    full, valid = generate(points, *args, chunk_size=3, point_chunk_size=1)
    full_calls = list(calls)
    calls.clear()
    target, mask, reported_valid = generate(
        points, *args, chunk_size=3, point_chunk_size=1,
        local_radius_px=4, return_valid_mask=True)
    assert calls == full_calls
    grid = _pixel_grid(*args[:2])
    nearest = (grid[:, :, None] - points).norm(dim=-1).min(-1).values
    assert ((nearest <= 4) & ~valid).any()  # Padding near GT really exists.
    assert torch.equal(mask, valid & (nearest <= 4))
    assert torch.equal(reported_valid, valid)
    assert not mask[~valid].any()
    torch.testing.assert_close(target, full, rtol=0, atol=0)


def test_local_denominator_is_global_selected_count():
    prediction = torch.tensor([[[[1., 99., 99., 99.]]],
                               [[[2., 3., 4., 99.]]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    selected = torch.tensor([[[[True, False, False, False]]],
                             [[[True, True, True, False]]]])
    raw = PointFIDTAuxHead.masked_mse(prediction, target, selected)
    assert raw == (1 + 4 + 9 + 16) / 4
    assert raw != 30 / 8  # Wrong: divide selected error by all image cells.
    assert raw != (1 + 29 / 3) / 2  # Wrong: average per-image means.
    raw.backward()
    assert prediction.grad[~selected].count_nonzero() == 0
    torch.testing.assert_close(prediction.grad[selected],
                               torch.tensor([0.5, 1., 1.5, 2.]))


def _local_samples():
    return [SimpleNamespace(
        metainfo=dict(img_shape=(16, 16), batch_input_shape=(16, 16)),
        gt_instances=SimpleNamespace(points=points))
        for points in (torch.tensor([[4.25, 4.75]]), torch.empty(0, 2))]


def test_local_mixed_empty_debug_and_full_map_compatibility():
    torch.manual_seed(51)
    head = PointFIDTAuxHead(in_channels=8, hidden_channels=8, num_groups=2,
                            fidt_loss_weight=0.7, debug=True)
    feature = torch.randn(2, 8, 4, 4)
    samples = _local_samples()
    pred = head(feature)
    full_losses = head.loss(feature, samples, (16, 16))
    # Independent old full-map reference; empty image has background MSE.
    distance = (_pixel_grid((4, 4), (16, 16))
                - samples[0].gt_instances.points[0]).norm(dim=-1)
    target = torch.stack((1 / (distance.pow(0.02 * distance + 0.75) + 1),
                          torch.zeros_like(distance))).unsqueeze(1)
    expected = (pred - target).square().mean()
    torch.testing.assert_close(full_losses['fidt_mse'], expected)
    torch.testing.assert_close(full_losses['loss_fidt'], expected * 0.7)
    assert set(full_losses) == {'loss_fidt', 'fidt_mse'}
    assert head.local_radius_px is None

    head.local_radius_px = 4
    losses = head.loss(feature, samples, (16, 16))
    selected = distance <= 4
    assert losses['fidt_local_count'] == selected.sum()
    assert losses['fidt_local_ratio'] == selected.sum() / 32
    expected_local = (pred[0, 0] - target[0, 0]).square()[selected].mean()
    torch.testing.assert_close(losses['fidt_mse'], expected_local)
    single = head.loss(feature[:1], samples[:1], (16, 16))
    torch.testing.assert_close(single['loss_fidt'], losses['loss_fidt'])
    assert all(not value.requires_grad for key, value in losses.items()
               if key != 'loss_fidt')
    head.debug = False
    assert set(head.loss(feature, samples, (16, 16))) == {'loss_fidt'}


@pytest.mark.parametrize('no_gt', [True, False])
def test_local_zero_selected_cells_connected_backward(no_gt):
    head = PointFIDTAuxHead(in_channels=8, hidden_channels=8, num_groups=2,
                            fidt_loss_weight=1, local_radius_px=0.01,
                            debug=True)
    feature = torch.randn(1, 8, 4, 4, requires_grad=True)
    sample = _local_samples()[1 if no_gt else 0]
    target, mask = generate(sample.gt_instances.points, (4, 4), (16, 16),
                            (16, 16), local_radius_px=0.01)
    assert mask.sum() == 0
    if no_gt:
        assert target.count_nonzero() == 0
    losses = head.loss(feature, [sample], (16, 16))
    assert all(torch.isfinite(x) and x == 0 for x in losses.values())
    assert losses['loss_fidt'].requires_grad
    losses['loss_fidt'].backward()
    assert feature.grad is not None and feature.grad.count_nonzero() == 0
    assert all(p.grad is not None and p.grad.count_nonzero() == 0
               for p in head.parameters())


@pytest.mark.parametrize('radius', [0, -1, float('nan'), float('inf')])
def test_invalid_local_radius(radius):
    with pytest.raises(ValueError, match='local_radius_px'):
        PointFIDTAuxHead(local_radius_px=radius)
    with pytest.raises(ValueError, match='local_radius_px'):
        generate(torch.empty(0, 2), (4, 4), (16, 16), (16, 16),
                 local_radius_px=radius)
