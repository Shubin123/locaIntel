import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from pipeline import mlx_facing as M
    from pipeline.train_drivebc_facing import group_log_probs, select_backend
    from pipeline.train_facing import FastFacing
    AVAILABLE = M.available()
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'needs MLX with Metal, PyTorch and timm (Apple silicon)')
class MlxFacingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.reference = FastFacing()
        # Non-trivial running statistics, as in a trained checkpoint.
        for name, buffer in cls.reference.named_buffers():
            if name.endswith('running_mean'):
                buffer.uniform_(-.1, .1)
            elif name.endswith('running_var'):
                buffer.uniform_(.5, 1.5)
        cls.reference.eval()
        cls.rng = np.random.default_rng(0)

    def test_weights_round_trip_and_eval_outputs_match_pytorch(self):
        student = M.Student(self.reference); student.eval()
        state = self.reference.state_dict()
        back = student.torch_state(state)
        self.assertTrue(all(torch.equal(back[k], state[k]) for k in state))
        x = self.rng.random((3, 160, 160, 3), dtype=np.float32)
        with torch.no_grad():
            logits, vector = self.reference(torch.from_numpy(x).permute(0, 3, 1, 2))
        mine, direction = student(mx.array(x))
        self.assertLess(float(np.abs(np.array(mine) - logits.numpy()).max()), 1e-3)
        self.assertLess(float(np.abs(np.array(direction) - vector.numpy()).max()), 1e-3)

    def test_resize_matches_torch_bilinear(self):
        x = self.rng.random((2, 160, 160, 3), dtype=np.float32)
        for size in (24, 56, 112):
            expected = F.interpolate(torch.from_numpy(x).permute(0, 3, 1, 2), (size, size), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).numpy()
            self.assertLess(float(np.abs(np.array(M.resize(mx.array(x), size)) - expected).max()), 1e-4)

    def test_strided_depthwise_gradients_match_pytorch(self):
        conv = torch.nn.Conv2d(24, 24, 5, stride=2, padding=2, groups=24, bias=False)
        x = self.rng.standard_normal((2, 24, 21, 21)).astype(np.float32)
        tx = torch.tensor(x, requires_grad=True); out = conv(tx)
        upstream = self.rng.standard_normal(out.shape).astype(np.float32); out.backward(torch.from_numpy(upstream))
        layer = M.Conv(conv); layer.weight = mx.array(conv.weight.detach().numpy().transpose(0, 2, 3, 1))
        grad = mx.grad(lambda v: (layer(v) * mx.array(upstream.transpose(0, 2, 3, 1))).sum())(mx.array(x.transpose(0, 2, 3, 1)))
        self.assertLess(float(np.abs(np.array(grad).transpose(0, 3, 1, 2) - tx.grad.numpy()).max()), 1e-4)

    def test_training_loss_and_gradients_match_pytorch(self):
        model = FastFacing(); model.load_state_dict(self.reference.state_dict()); model.train()
        student = M.Student(model); student.train()
        x, dx, px = (self.rng.random((n, 160, 160, 3), dtype=np.float32) for n in (4, 3, 3))
        y, dy = self.rng.integers(0, 8, 4), self.rng.integers(0, 3, 3)
        soft = self.rng.dirichlet([1, 1, 1], 3).astype(np.float32); angle = self.rng.uniform(0, 360, 4).astype(np.float32)
        circle = np.stack([np.sin(np.radians(angle)), np.cos(np.radians(angle))], 1).astype(np.float32)
        nchw = lambda v: torch.from_numpy(v).permute(0, 3, 1, 2)
        logits, direction = model(nchw(x))
        loss = F.cross_entropy(logits, torch.from_numpy(y), label_smoothing=.05) + .6 * (1 - (F.normalize(direction, dim=1) * torch.from_numpy(circle)).sum(1)).mean()
        loss = loss + 2 * F.nll_loss(group_log_probs(model(nchw(dx))[0]), torch.from_numpy(dy)) - 1.5 * (torch.from_numpy(soft) * group_log_probs(model(nchw(px))[0])).sum(1).mean()
        loss.backward()
        value, grads = nn.value_and_grad(student, lambda *b: M.loss_fn(student, *b))(*map(mx.array, (x, y, circle, dx, dy, px, soft)))
        self.assertAlmostEqual(float(value), float(loss), places=3)
        from mlx.utils import tree_flatten
        mine = {k: np.array(v) for k, v in tree_flatten(grads)}
        theirs = {k[len('backbone.'):]: p.grad.numpy() for k, p in model.named_parameters()}
        mine = {k: v.transpose(0, 3, 1, 2) if v.ndim == 4 else v for k, v in mine.items()}
        error = np.sqrt(sum(((mine[k] - theirs[k]) ** 2).sum() for k in theirs)) / np.sqrt(sum((v ** 2).sum() for v in theirs.values()))
        self.assertLess(error, 1e-2)
        state = student.torch_state(model.state_dict())
        for name in ('backbone.bn1.running_mean', 'backbone.blocks.5.2.bn2.running_var'):
            self.assertLess(float((state[name] - model.state_dict()[name]).abs().max()), 1e-3)

    def test_fit_trains_and_returns_a_pytorch_model(self):
        frame = Image.fromarray(self.rng.integers(0, 255, (120, 200, 3), dtype=np.uint8))
        rows = [{'file': 'f.jpg', 'box': [.1 + .1 * i, .2, .25 + .1 * i, .6], 'group': g} for i, g in enumerate(['front', 'rear', 'side', 'front'])]
        cfv = (self.rng.integers(0, 255, (16, 160, 160, 3), dtype=np.uint8), self.rng.uniform(0, 360, 16), np.arange(16))
        with tempfile.TemporaryDirectory() as directory:
            init = Path(directory) / 'init.pt'; torch.save(self.reference.state_dict(), init)
            for pseudo in (None, (rows, np.full((4, 3), 1 / 3, np.float32))):
                trained = M.fit(rows, {'f.jpg': frame}, cfv, 'cpu', 1, 2, 5, init, pseudo)
                self.assertIsInstance(trained, FastFacing)
                moved = max(float((a - b).abs().max()) for a, b in zip(trained.state_dict().values(), self.reference.state_dict().values()) if a.is_floating_point())
                self.assertGreater(moved, 0)
                with torch.no_grad():
                    self.assertTrue(torch.isfinite(trained.eval()(torch.rand(2, 3, 160, 160))[0]).all())

    def test_backend_option(self):
        self.assertIs(select_backend('mlx'), M.fit)
