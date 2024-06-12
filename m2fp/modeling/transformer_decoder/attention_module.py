import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from .prior_module import PriorModule


class AttentionModule(nn.Module):

    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 384):
        super(AttentionModule, self).__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        hidden_dim = max(dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )
        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, slots, num_slots = None):
        B, N, D, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots
        slots = rearrange(slots, "n b d -> b n d")
        inputs = self.norm_input(inputs)        
        k = self.to_k(inputs)
        v = self.to_v(inputs)
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim = 1) + self.eps
            attn = attn / attn.sum(dim = -1, keepdim = True)
            updates = torch.einsum('bjd,bij->bid', v, attn)
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            )
            slots = slots.reshape(B, -1, D)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        slots = rearrange(slots, "b n d -> n b d")
        return slots