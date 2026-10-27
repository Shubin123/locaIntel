"""Train the DriveBC facing student with MLX on the Apple-silicon GPU.

`python -m pipeline.train_drivebc_facing --backend mlx` runs the student's
training steps here instead of in PyTorch. MLX executes on the GPU through
Metal. Apple's Neural Engine is not exposed to any training framework, so no
option here uses it; it can only run finished Core ML models. On an M4 this
depthwise-heavy MobileNet trains ≈2.7× slower in MLX than with PyTorch's MPS
backend, so PyTorch stays the default.

The network mirrors timm's MobileNetV3-Large module for module, weights convert
exactly in both directions, and the losses, sampling, augmentation, AdamW,
cosine schedule and gradient clipping follow train_drivebc_facing.fit. The
trained student comes back as the usual PyTorch FastFacing, so cross-validation
predictions and ONNX export are unchanged.
"""
import math
import random
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

MEAN = mx.array([.485, .456, .406])
STD = mx.array([.229, .224, .225])
GROUP_VIEWS = [mx.array(v) for v in ([0, 1, 7], [3, 4, 5], [2, 6])]
ACTIVATIONS = {'ReLU': nn.relu, 'Hardswish': nn.hardswish, 'Hardsigmoid': lambda x: mx.clip(x + 3, 0, 6) / 6, 'Identity': lambda x: x}


def activation(module):
    return ACTIVATIONS[type(module).__name__]


class Conv(nn.Module):
    """A torch Conv2d in NHWC layout (weights O×kh×kw×I/groups)."""

    def __init__(self, conv):
        super().__init__()
        out, per_group, kh, kw = conv.weight.shape
        self.weight = mx.zeros((out, kh, kw, per_group))
        if conv.bias is not None:
            self.bias = mx.zeros((out,))
        self.stride, self.padding, self.groups = conv.stride[0], conv.padding[0], conv.groups

    def __call__(self, x):
        if self.groups > 1 and self.stride > 1:
            # MLX before 0.32 returns wrong input gradients for strided grouped convolutions
            # (0.29.3, the last release for Python 3.9, has it). A stride-1 convolution sampled
            # at every stride-th pixel is the same function with correct gradients.
            y = mx.conv2d(x, self.weight, padding=self.padding, groups=self.groups)[:, ::self.stride, ::self.stride, :]
        else:
            y = mx.conv2d(x, self.weight, stride=self.stride, padding=self.padding, groups=self.groups)
        return y + self.bias if 'bias' in self else y


class BatchNormAct(nn.BatchNorm):
    """timm BatchNormAct2d with PyTorch's running statistics (unbiased running variance)."""

    def __init__(self, bn):
        super().__init__(bn.num_features, eps=bn.eps, momentum=bn.momentum)
        self.act = activation(getattr(bn, 'act', None) or nn.Identity())

    def __call__(self, x):
        if self.training:
            mean, var = x.mean(axis=(0, 1, 2)), x.var(axis=(0, 1, 2))
            n = x.size // x.shape[-1]
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var * n / max(1, n - 1)
        else:
            mean, var = self.running_mean, self.running_var
        return self.act((x - mean) * mx.rsqrt(var + self.eps) * self.weight + self.bias)


class SqueezeExcite(nn.Module):
    def __init__(self, se):
        super().__init__()
        self.conv_reduce, self.conv_expand = Conv(se.conv_reduce), Conv(se.conv_expand)
        self.act1, self.gate = activation(se.act1), activation(se.gate)

    def __call__(self, x):
        return x * self.gate(self.conv_expand(self.act1(self.conv_reduce(x.mean(axis=(1, 2), keepdims=True)))))


# Layer order of timm's MobileNetV3 blocks (aa and drop_path are identities in this model).
ORDER = {'DepthwiseSeparableConv': ('conv_dw', 'bn1', 'se', 'conv_pw', 'bn2'),
         'InvertedResidual': ('conv_pw', 'bn1', 'conv_dw', 'bn2', 'se', 'conv_pwl', 'bn3'),
         'ConvBnAct': ('conv', 'bn1')}


class Block(nn.Module):
    def __init__(self, block):
        super().__init__()
        import torch
        names = [n for n in ORDER[type(block).__name__] if not isinstance(getattr(block, n), torch.nn.Identity)]
        for name in names:
            layer = getattr(block, name)
            setattr(self, name, Conv(layer) if isinstance(layer, torch.nn.Conv2d) else SqueezeExcite(layer) if name == 'se' else BatchNormAct(layer))
        self.names, self.skip = ' '.join(names), bool(block.has_skip)

    def __call__(self, x):
        y = x
        for name in self.names.split():
            y = self[name](y)
        return x + y if self.skip else y


class Student(nn.Module):
    """FastFacing (timm mobilenetv3_large_100, 10 outputs) on 160×160 NHWC images in [0, 1]."""

    def __init__(self, torch_model):
        super().__init__()
        b = torch_model.backbone
        self.conv_stem, self.bn1 = Conv(b.conv_stem), BatchNormAct(b.bn1)
        self.blocks = [[Block(block) for block in stage] for stage in b.blocks]
        self.conv_head, self.act2 = Conv(b.conv_head), activation(b.act2)
        self.classifier = nn.Linear(b.classifier.in_features, b.classifier.out_features)
        self.load_torch(torch_model.state_dict())

    def __call__(self, x):
        x = self.bn1(self.conv_stem((x - MEAN) / STD))
        for stage in self.blocks:
            for block in stage:
                x = block(x)
        x = self.act2(self.conv_head(x.mean(axis=(1, 2), keepdims=True))).reshape(x.shape[0], -1)
        out = self.classifier(x)
        return out[:, :8], out[:, 8:]

    def load_torch(self, state):
        weights = {k[len('backbone.'):]: v.detach().cpu().numpy() for k, v in state.items()
                   if k.startswith('backbone.') and not k.endswith('num_batches_tracked')}
        mine = {k for k, _ in tree_flatten(self.parameters())}
        if mine != set(weights):
            raise ValueError('MLX and PyTorch facing models differ: ' + ', '.join(sorted(mine ^ set(weights))[:6]))
        self.update(tree_unflatten([(k, mx.array(v.transpose(0, 2, 3, 1) if v.ndim == 4 else v)) for k, v in weights.items()]))

    def torch_state(self, template):
        """A PyTorch state dict for FastFacing; buffers MLX does not train come from template."""
        import torch
        state = {k: v.clone() for k, v in template.items()}
        for k, v in tree_flatten(self.parameters()):
            v = np.array(v)
            state['backbone.' + k] = torch.from_numpy(np.ascontiguousarray(v.transpose(0, 3, 1, 2) if v.ndim == 4 else v))
        return state


def resize_matrix(source, target):
    """Rows of torch's bilinear interpolation (align_corners=False, no antialias) as a matrix."""
    m = np.zeros((target, source), np.float32)
    for i in range(target):
        x = max(0.0, (i + .5) * source / target - .5)
        lo = min(int(x), source - 1); hi = min(lo + 1, source - 1); w = x - lo
        m[i, lo] += 1 - w; m[i, hi] += w
    return mx.array(m)


def resize(x, size):
    """Bilinear resize of NHWC images to size×size."""
    rows, cols = resize_matrix(x.shape[1], size), resize_matrix(x.shape[2], size)
    return mx.einsum('ih,nhwc,jw->nijc', rows, x, cols)


def cfv_batch(images, angles, indices, augment=True):
    x = mx.array(images[indices]).astype(mx.float32) / 255
    a = mx.array(angles[indices].astype(np.float32))
    if augment:
        flips = mx.random.uniform(shape=(len(indices),)) < .5
        x = mx.where(flips[:, None, None, None], x[:, :, ::-1, :], x); a = mx.where(flips, (360 - a) % 360, a)
        x = mx.clip(x * mx.random.uniform(.6, 1.4, (len(indices), 1, 1, 1)), 0, 1)
        x = mx.clip((x - .5) * mx.random.uniform(.65, 1.35, (len(indices), 1, 1, 1)) + .5, 0, 1)
    if augment and random.random() < .7:
        x = resize(resize(x, random.choice([24, 32, 40, 56, 80, 112])), 160)
    y = mx.remainder(mx.ceil((a - 22.5) / 45).astype(mx.int32), 8)
    radians = a * math.pi / 180
    return x, y, mx.stack([mx.sin(radians), mx.cos(radians)], axis=1)


def drivebc_batch(rows, images, augment):
    from .train_drivebc_facing import GROUPS, drivebc_crop
    crops = []
    for r in rows:
        c = drivebc_crop(images[r['file']], r['box'], jitter=.08 if augment else 0)
        crops.append(c[:, ::-1] if augment and random.random() < .5 else c)
    x = mx.array(np.stack(crops)).astype(mx.float32) / 255
    if augment:
        x = mx.clip(x * mx.random.uniform(.7, 1.3, (len(rows), 1, 1, 1)), 0, 1)
    return x, mx.array([GROUPS.get(r.get('group'), 0) for r in rows])


def group_log_probs(logits):
    logp = logits - mx.logsumexp(logits, axis=1, keepdims=True)
    return mx.stack([mx.logsumexp(logp[:, v], axis=1) for v in GROUP_VIEWS], axis=1)


def loss_fn(model, x, y, circle, dx, dy, px=None, soft=None):
    logits, direction = model(x)
    direction = direction / mx.maximum(mx.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    loss = nn.losses.cross_entropy(logits, y, label_smoothing=.05, reduction='mean') + .6 * (1 - (direction * circle).sum(1)).mean()
    loss = loss - 2 * mx.take_along_axis(group_log_probs(model(dx)[0]), dy[:, None], axis=1).mean()
    if px is not None:
        loss = loss - 1.5 * (soft * group_log_probs(model(px)[0])).sum(1).mean()
    return loss


def fit(train_rows, images, cfv, device, epochs, steps, seed, init, pseudo=None):
    """train_drivebc_facing.fit on MLX; returns a PyTorch FastFacing on `device`."""
    import torch
    from .train_facing import FastFacing
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed); mx.random.seed(seed)
    template = torch.load(init, map_location='cpu', weights_only=True)
    reference = FastFacing(); reference.load_state_dict(template)
    model = Student(reference); model.train()
    total = epochs * steps
    optimizer = optim.AdamW(learning_rate=optim.cosine_decay(3e-4, total, 1e-5), weight_decay=.02, bias_correction=True)
    gradients = nn.value_and_grad(model, lambda *batch: loss_fn(model, *batch))
    state = [model.state, optimizer.state]

    # One compiled graph per step; BatchNorm running statistics and optimizer state flow through `state`.
    @partial(mx.compile, inputs=state, outputs=state)
    def step(*batch):
        loss, grads = gradients(*batch)
        grads, _ = optim.clip_grad_norm(grads, 5)
        optimizer.update(model, grads)
        return loss
    # MLX keeps freed buffers for reuse; uncapped, that cache reached 15 GB on a 24 GB Mac and
    # pushed the system into swap (≈30× slower steps). A step needs ≈7.6 GB at its peak.
    previous_cache = mx.set_cache_limit(1 << 30)
    cfv_images, cfv_angles, cfv_train = cfv
    # Class-balanced DriveBC sampling: rear views are rarer in the labeled set.
    counts = {g: sum(r['group'] == g for r in train_rows) for g in {r['group'] for r in train_rows}}
    weights = [1 / counts[r['group']] for r in train_rows]
    try:
        for _ in range(total):
            x, y, circle = cfv_batch(cfv_images, cfv_angles, np.random.choice(cfv_train, 48))
            dx, dy = drivebc_batch(random.choices(train_rows, weights=weights, k=48), images, True)
            batch = (x, y, circle, dx, dy)
            if pseudo:
                rows, targets = pseudo
                pick = np.random.choice(len(rows), 64)
                px, _ = drivebc_batch([rows[i] for i in pick], images, True)
                batch += (px, mx.array(targets[pick].astype(np.float32)))
            mx.eval(step(*batch), state)
    finally:
        mx.set_cache_limit(previous_cache)
    trained = FastFacing(); trained.load_state_dict(model.torch_state(template)); trained.to(device)
    return trained


def available():
    try:
        return mx.metal.is_available()
    except Exception:
        return False
