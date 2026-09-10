# Copyright (c) OpenMMLab. All rights reserved.
"""Real Point-DINO integration smoke; no mocked transformer or MMCV ops.

Run from the repository root with::

    python -m pytest -o addopts= -s \
        tests/test_models/test_detectors/test_point_dino_stage4.py

Synthetic training uses one 256x256 image, 30 matching queries, two encoder
layers and a DN budget of 10 (20 positive/negative queries for two GTs).
R50, width 256, four feature levels, all six decoder
layers, 2D references, the real matcher and Point DN remain enabled. Keeping
six decoder layers verifies all 39 existing Stage 3 loss keys. Full-map and
local-radius-16 supervision share the same two detector instances and initial
auxiliary weights. Native 768x1024 is separately exercised through the real
backbone/neck and FIDT head only. This is a functional smoke, not a
ShanghaiTech training or accuracy result.
"""
from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('mmengine')
pytest.importorskip(
    'mmcv.ops', reason='Real detector smoke requires compiled MMCV operators.')

from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.utils import register_all_modules

ROOT = Path(__file__).resolve().parents[3]
STAGE3 = ROOT / 'configs/dino/point_dino_stage3_lam020_cost20.py'
STAGE4 = (ROOT / 'configs/dino/'
          'point_dino_r50_shanghaitech_stage4_fidt_native_12e.py')
SEED = 73


def _build(stage4=False, disabled=False, zero_weight=False):
    config = deepcopy(Config.fromfile(str(STAGE4 if stage4 else STAGE3)).model)
    config.backbone.init_cfg = None  # No pretrained download or checkpoint.
    config.num_queries = 30
    config.test_cfg.max_per_img = 30
    config.encoder.num_layers = 2
    config.dn_cfg.group_cfg.num_dn_queries = 10
    if disabled:
        config.point_fidt_head.enabled = False
    if zero_weight:
        config.point_fidt_head.fidt_loss_weight = 0.0
    torch.manual_seed(SEED)
    model = MODELS.build(config)
    # Avoid thousands of routine per-parameter initialization log lines.
    logger = MMLogger.get_current_instance()
    old_level = logger.level
    logger.setLevel('WARNING')
    try:
        model.init_weights()
    finally:
        logger.setLevel(old_level)
    return model


def _raw_batch():
    # These are the actual PackDetInputs fields: points and labels, no boxes.
    sample = DetDataSample(metainfo=dict(
        img_shape=(256, 256), ori_shape=(128, 128), scale_factor=(2., 2.)))
    sample.gt_instances = InstanceData(
        points=torch.tensor([[63.25, 81.5], [177.75, 189.125]]),
        labels=torch.zeros(2, dtype=torch.long))
    generator = torch.Generator().manual_seed(123)
    return dict(inputs=[torch.rand(3, 256, 256, generator=generator) * 255],
                data_samples=[sample])


def _stage3_loss_keys():
    components = ('cls', 'point', 'point_euclidean')
    keys = {f'enc_loss_{name}' for name in components}
    for prefix in ('', 'd0.', 'd1.', 'd2.', 'd3.', 'd4.'):
        for name in components:
            keys.add(f'{prefix}loss_{name}')
            keys.add(f'{prefix}dn_loss_{name}')
    assert len(keys) == 39
    return keys


def _has_nonzero_grad(module):
    return any(p.grad is not None and torch.isfinite(p.grad).all()
               and p.grad.abs().sum() > 0 for p in module.parameters())


def _assert_predictions_equal(expected, actual):
    assert len(expected) == len(actual) == 1
    left, right = expected[0].pred_instances, actual[0].pred_instances
    assert set(left.keys()) == set(right.keys()) == {'points', 'scores', 'labels'}
    assert right.points.shape == (30, 2)
    for key in ('points', 'scores', 'labels'):
        assert torch.equal(left[key], right[key]), key


@pytest.fixture(scope='module')
def models():
    register_all_modules()
    old_threads = torch.get_num_threads()
    # Tiny CPU operations can be much slower with a large host thread pool.
    torch.set_num_threads(min(old_threads, 4))
    baseline = _build()
    baseline_rng = torch.get_rng_state().clone()
    active = _build(stage4=True)
    incompatible = active.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    expected_missing = {
        f'point_fidt_head.{name}' for name in active.point_fidt_head.state_dict()}
    assert set(incompatible.missing_keys) == expected_missing
    print('Stage 3 state_dict load: only auxiliary keys missing:',
          sorted(incompatible.missing_keys))
    yield baseline, active, baseline_rng
    torch.set_num_threads(old_threads)


@pytest.fixture(params=[None, 16.0], ids=['full', 'local16'])
def fidt_mode(models, request):
    """Exercise the real mode constructor without rebuilding two detectors."""
    baseline, active, _ = models
    original_head = active.point_fidt_head
    head_config = deepcopy(Config.fromfile(str(STAGE4)).model.point_fidt_head)
    head_config.pop('enabled')
    head_config.update(local_radius_px=request.param, fidt_loss_weight=1.0,
                       debug=True)
    active.point_fidt_head = type(original_head)(**head_config)
    active.point_fidt_head.load_state_dict(original_head.state_dict())
    try:
        yield baseline, active, request.param
    finally:
        # Each mode starts with identical core and auxiliary weights, including
        # after the real optimizer step in the training smoke.
        active.load_state_dict(baseline.state_dict(), strict=False)
        active.point_fidt_head = original_head
        active.zero_grad(set_to_none=True)


def test_real_training_losses_and_shared_feature_gradients(fidt_mode):
    baseline, active, radius = fidt_mode
    baseline.train()
    active.train()
    active.zero_grad(set_to_none=True)
    batch = active.data_preprocessor(deepcopy(_raw_batch()), training=True)
    assert batch['data_samples'][0].batch_input_shape == (256, 256)

    # Observe the actual shared tensor, without replacing any computation.
    neck_outputs, aux_inputs = [], []
    handles = [
        active.neck.register_forward_hook(
            lambda module, args, output: neck_outputs.append(output)),
        active.point_fidt_head.register_forward_pre_hook(
            lambda module, args: aux_inputs.append(args[0]))]
    try:
        torch.manual_seed(19)  # Match the real stochastic Point DN samples.
        losses = active.loss(batch['inputs'], batch['data_samples'])
    finally:
        for handle in handles:
            handle.remove()
    assert len(aux_inputs) == len(neck_outputs) == 1
    assert aux_inputs[0] is neck_outputs[0][0]
    assert [tuple(x.shape) for x in neck_outputs[0]] == [
        (1, 256, 32, 32), (1, 256, 16, 16),
        (1, 256, 8, 8), (1, 256, 4, 4)]

    original_keys = _stage3_loss_keys()
    debug_keys = {'fidt_mse'}
    if radius is not None:
        debug_keys |= {'fidt_local_ratio', 'fidt_local_count'}
    assert set(losses) == original_keys | {'loss_fidt'} | debug_keys
    assert all(torch.isfinite(value).all() for value in losses.values())
    assert losses['loss_fidt'] > 0
    for key in debug_keys:
        assert losses[key].ndim == 0
        assert not losses[key].requires_grad
    if radius is not None:
        # Count selected cells independently in image pixels; the preprocessor
        # produced a fully valid 256x256 input and the auxiliary map is 64x64.
        centers = (torch.arange(64, dtype=torch.float32) + 0.5) * 4
        grid_y, grid_x = torch.meshgrid(centers, centers, indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
        points = batch['data_samples'][0].gt_instances.points
        nearest = (grid[:, None] - points[None]).norm(dim=-1).amin(dim=1)
        expected_count = (nearest <= radius).sum()
        assert 0 < expected_count < 64 * 64
        torch.testing.assert_close(losses['fidt_local_count'],
                                   expected_count.float())
        torch.testing.assert_close(losses['fidt_local_ratio'],
                                   expected_count.float() / (64 * 64))
    with torch.no_grad():
        torch.manual_seed(19)
        stage3_losses = baseline.loss(batch['inputs'], batch['data_samples'])
    assert set(stage3_losses) == original_keys
    for key in original_keys:
        assert torch.equal(stage3_losses[key], losses[key]), key

    total, log_vars = active.parse_losses(losses)
    manual_total = sum(value.mean() for key, value in losses.items()
                       if 'loss' in key)
    torch.testing.assert_close(total, manual_total)
    without_debug, _ = active.parse_losses(
        {key: value for key, value in losses.items() if key not in debug_keys})
    torch.testing.assert_close(total, without_debug, rtol=0, atol=0)
    assert debug_keys <= set(log_vars)
    torch.testing.assert_close(
        losses['loss_fidt'],
        losses['fidt_mse'] * active.point_fidt_head.fidt_loss_weight)

    losses['loss_fidt'].backward(retain_graph=True)
    assert _has_nonzero_grad(active.point_fidt_head)
    assert _has_nonzero_grad(active.neck)
    assert _has_nonzero_grad(active.backbone.layer2)
    # FIDT alone must never produce query/transformer/head/DN gradients.
    for module in (active.encoder, active.decoder, active.bbox_head,
                   active.query_embedding, active.dn_query_generator):
        assert all(p.grad is None for p in module.parameters())
    assert all(p.grad is None for p in active.backbone.parameters()
               if not p.requires_grad)
    active.zero_grad(set_to_none=True)
    total.backward()
    assert _has_nonzero_grad(active.decoder)
    assert _has_nonzero_grad(active.bbox_head)
    assert _has_nonzero_grad(active.dn_query_generator)
    assert all(torch.isfinite(p.grad).all() for p in active.parameters()
               if p.grad is not None)
    before_step = active.point_fidt_head.output.weight.detach().clone()
    # One real update checks optimizer usability; this smoke SGD is not the
    # experiment optimizer and does not modify the inherited AdamW config.
    torch.optim.SGD(active.parameters(), lr=1e-4).step()
    assert not torch.equal(before_step, active.point_fidt_head.output.weight)
    print('Training smoke: Stage 3 keys preserved:', sorted(original_keys))
    print('Training smoke: mode=', 'full' if radius is None else 'local16',
          ';', {key: losses[key].item() for key in (
              'loss_fidt', 'fidt_mse', 'fidt_local_ratio', 'fidt_local_count',
              'loss_point', 'loss_point_euclidean') if key in losses},
          '; Stage 3 total=', sum(stage3_losses.values()).item(),
          '; parsed total loss=', total.item(),
          '; detached debug keys excluded=', sorted(debug_keys),
          '; backward and optimizer.step passed')
    print('FIDT-only gradients: auxiliary/neck/backbone.layer2 nonzero; '
          'query/encoder/decoder/Point-DN gradients absent')


def test_predict_and_test_step_are_bitwise_stage3(fidt_mode):
    baseline, active, radius = fidt_mode
    baseline.eval()
    active.eval()

    def forbid_auxiliary(module, args):
        pytest.fail('Inference evaluated the training-only FIDT head.')

    handle = active.point_fidt_head.register_forward_pre_hook(forbid_auxiliary)
    try:
        with torch.no_grad():
            batch = baseline.data_preprocessor(_raw_batch(), training=False)
            expected = baseline.predict(batch['inputs'],
                                        deepcopy(batch['data_samples']))
            predicted = active.predict(batch['inputs'],
                                       deepcopy(batch['data_samples']))
            _assert_predictions_equal(expected, predicted)
            _assert_predictions_equal(expected, active.test_step(_raw_batch()))
            _assert_predictions_equal(expected,
                                      baseline.test_step(_raw_batch()))
    finally:
        handle.remove()
    print('predict/test_step:', 'full' if radius is None else 'local16',
          '; zero FIDT calls; Stage 3 scores/points/labels bitwise equal; '
          'prediction points shape=(30, 2)')


@pytest.mark.parametrize('off_mode', ['zero_weight', 'disabled'])
def test_disabled_stage4_preserves_rng_parameters_and_predictions(models,
                                                               off_mode):
    baseline, _, baseline_rng = models
    off = _build(stage4=True, **{off_mode: True})
    assert off.point_fidt_head is None
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    expected_state = baseline.state_dict()
    actual_state = off.state_dict()
    assert set(actual_state) == set(expected_state)
    for key in expected_state:
        assert actual_state[key].shape == expected_state[key].shape, key
        assert torch.equal(actual_state[key], expected_state[key]), key
    baseline.eval()
    off.eval()
    with torch.no_grad():
        _assert_predictions_equal(baseline.test_step(_raw_batch()),
                                  off.test_step(_raw_batch()))
    if off_mode == 'zero_weight':
        baseline.train()
        off.train()
        batch = baseline.data_preprocessor(_raw_batch(), training=True)
        with torch.no_grad():
            torch.manual_seed(31)
            original_losses = baseline.loss(batch['inputs'],
                                             batch['data_samples'])
            torch.manual_seed(31)
            off_losses = off.loss(batch['inputs'], batch['data_samples'])
        assert set(off_losses) == set(original_losses) == _stage3_loss_keys()
        for key in original_losses:
            assert torch.equal(original_losses[key], off_losses[key]), key
        print('zero_weight: all 39 training losses bitwise equal Stage 3; '
              'no loss_fidt or fidt_mse')
    print(off_mode, ': no auxiliary parameters; construction/init RNG, '
          'all parameter values/shapes and prediction outputs equal Stage 3')


def test_native_backbone_neck_and_auxiliary_shapes(models):
    _, active, _ = models
    active.eval()
    backbone_shapes = []
    handle = active.backbone.register_forward_hook(
        lambda module, args, output: backbone_shapes.extend(
            tuple(feature.shape) for feature in output))
    try:
        with torch.no_grad():
            feats = active.extract_feat(torch.zeros(1, 3, 768, 1024))
    finally:
        handle.remove()
    assert backbone_shapes == [(1, 512, 96, 128), (1, 1024, 48, 64),
                               (1, 2048, 24, 32)]
    with torch.no_grad():
        shapes = [tuple(feature.shape) for feature in feats]
        assert shapes == [(1, 256, 96, 128), (1, 256, 48, 64),
                          (1, 256, 24, 32), (1, 256, 12, 16)]
        fidt = active.point_fidt_head(feats[0])
        assert fidt.shape == (1, 1, 192, 256)
    print('Native actual backbone output:', backbone_shapes,
          '; neck output:', shapes,
          '; strides=8/16/32/64; FIDT output=', tuple(fidt.shape),
          '; no native transformer/training benchmark performed')
