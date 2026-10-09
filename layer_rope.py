"""LayerRoPE (https://github.com/shikhar-srivastava/layer-rope) for the timm ViT blocks.

Every block's two LayerNorms take their gains from one shared gamma per site, and the attention and MLP outputs
are scaled by shared gains generated the same way, both conditioned on the block's depth l (u = log(l + 1)):

    gamma_l = exp(alpha + beta u) * rotate(gamma, theta_j),  theta_j = exp(alpha_rot + beta_rot u) * base_freq^(-2j/D)

where rotate multiplies the pairs (gamma[2j], gamma[2j+1]) by exp(i theta_j). Each LayerNorm keeps a bias of its
own. All arithmetic is in fp32.
"""

import math

import torch
from torch import nn, Tensor
from timm.models.vision_transformer import Block


def _as_fp32(v) -> Tensor:
    return v.to(torch.float32) if isinstance(v, Tensor) else torch.tensor(v, dtype=torch.float32)


def _apply_complex_rotation(x: Tensor, mag, theta) -> Tensor:
    """Multiply the pairs (x[2j] + i x[2j+1]) by exp(mag + i theta[j]), in fp32."""
    original_dtype = x.dtype
    x_f = x.to(torch.float32)
    u = x_f[..., 0::2]
    v = x_f[..., 1::2]
    R = torch.exp(_as_fp32(mag))[..., None]
    cos_t = torch.cos(theta.to(torch.float32))
    sin_t = torch.sin(theta.to(torch.float32))
    u_out = R * (u * cos_t - v * sin_t)
    v_out = R * (u * sin_t + v * cos_t)
    return torch.stack([u_out, v_out], dim=-1).flatten(-2).to(original_dtype)


class LayerRopeGates(nn.Module):
    """The shared LayerRoPE gains and depth scalars. The model owns one instance; every block holds a
    non-registered reference to it, so its parameters are counted and checkpointed once."""

    _BRANCHES = ("attn", "mlp")
    eps = 1e-6  # the timm ViT LayerNorm eps

    def __init__(self, dim: int, alpha: float, beta: float, alpha_rot: float, beta_rot: float, base_freq: float) -> None:
        super().__init__()
        assert dim % 2 == 0, f"LayerRoPE requires an even width, got {dim}"

        self.gamma_input = nn.Parameter(torch.ones(dim, dtype=torch.float32))  # attention input
        self.gamma_post = nn.Parameter(torch.ones(dim, dtype=torch.float32))  # MLP input
        self.gamma_residual_attn = nn.Parameter(torch.ones(dim, dtype=torch.float32))  # attention output
        self.gamma_residual_mlp = nn.Parameter(torch.ones(dim, dtype=torch.float32))  # MLP output

        def scalar(v: float) -> nn.Parameter:
            return nn.Parameter(torch.tensor(float(v), dtype=torch.float32))

        self.gates = nn.ParameterDict()
        for stage in ("input", "residual"):
            for branch in self._BRANCHES:
                self.gates[f"{stage}_alpha_{branch}"] = scalar(alpha)
                self.gates[f"{stage}_beta_{branch}"] = scalar(beta)
                self.gates[f"{stage}_alpha_rot_{branch}"] = scalar(alpha_rot)
                self.gates[f"{stage}_beta_rot_{branch}"] = scalar(beta_rot)

        j = torch.arange(dim // 2, dtype=torch.float32)
        rope_freqs = torch.pow(torch.tensor(float(base_freq), dtype=torch.float32), -2.0 * j / dim)
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

    def _rotated_gamma(self, gamma: Tensor, stage: str, branch: str, u: float) -> Tensor:
        g = self.gates
        mag = _as_fp32(g[f"{stage}_alpha_{branch}"]) + _as_fp32(g[f"{stage}_beta_{branch}"]) * _as_fp32(u)
        theta_scalar = torch.exp(
            _as_fp32(g[f"{stage}_alpha_rot_{branch}"]) + _as_fp32(g[f"{stage}_beta_rot_{branch}"]) * _as_fp32(u)
        )
        theta = theta_scalar.to(torch.float32)[..., None] * self.rope_freqs
        return _apply_complex_rotation(gamma, mag, theta)

    def input_norm(self, x: Tensor, u: float, branch: str, bias: Tensor) -> Tensor:
        """LayerNorm of a block's attention (branch="attn") or MLP input, with the depth-conditioned gain."""
        gamma = self.gamma_input if branch == "attn" else self.gamma_post
        gamma_eff = self._rotated_gamma(gamma, "input", branch, u)
        x_f = x.to(torch.float32)
        xc = x_f - x_f.mean(-1, keepdim=True)
        xhat = xc * torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + self.eps)
        return (gamma_eff * xhat + bias.to(torch.float32)).to(x.dtype)

    def residual_gate(self, y: Tensor, u: float, branch: str) -> Tensor:
        """Depth-conditioned gain on a block's attention or MLP output, before the residual add."""
        gamma = self.gamma_residual_attn if branch == "attn" else self.gamma_residual_mlp
        return self._rotated_gamma(gamma, "residual", branch, u).to(y.dtype) * y


class LayerRopeBlock(Block):
    """A timm ViT block whose norms and output gains come from the model's LayerRopeGates (see apply_layer_rope)."""

    def forward(self, x):
        g = self._gates
        a = self.attn(g.input_norm(x, self.log_l, "attn", self.ln_bias_attn))
        x = x + self.drop_path(g.residual_gate(a, self.log_l, "attn"))
        m = self.mlp(g.input_norm(x, self.log_l, "mlp", self.ln_bias_mlp))
        x = x + self.drop_path(g.residual_gate(m, self.log_l, "mlp"))
        return x


def apply_layer_rope(model, alpha: float, beta: float, alpha_rot: float, beta_rot: float, base_freq: float) -> None:
    """Convert every block of a timm VisionTransformer into a LayerRopeBlock, in place. The model's final norm is
    unchanged."""
    dim = model.embed_dim
    model.layer_rope_gates = LayerRopeGates(dim, alpha, beta, alpha_rot, beta_rot, base_freq)
    for idx, blk in enumerate(model.blocks):
        blk.log_l = math.log(idx + 1)
        object.__setattr__(blk, "_gates", model.layer_rope_gates)
        del blk.norm1
        del blk.norm2
        blk.__class__ = LayerRopeBlock
        blk.ln_bias_attn = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        blk.ln_bias_mlp = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
