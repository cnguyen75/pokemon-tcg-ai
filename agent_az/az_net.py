"""AlphaZero-style joint policy+value network for pokemon-tcg-ai.

Phase 3 SCALE-UP: the Phase-1/2 net (d_model=128, single-layer torso, ~68k
params) UNDERFIT the 176k soft self-play visit targets (SP top-1 agreement
capped ~0.62). This enlarged architecture raises capacity ~10x while keeping
per-leaf inference cheap (<0.15 ms/forward on this Mac; the MCTS bottleneck is
the engine step, not the net):

  d_model         256   (was 128)
  torso           input proj -> N_BLOCKS residual MLP blocks (was 1 hidden layer)
  value head      3-layer MLP  (was 2)
  policy head     4-layer MLP scorer over [option feats | torso]  (was 3)

~700k params at d_model=256, n_blocks=3. Architecture hyperparameters are stored
in the checkpoint (d_model, n_blocks) so both a freshly BC-bootstrapped big net
and the legacy small nets load through the same class. n_blocks=0 reproduces the
old single-hidden-layer torso for backward compatibility.

Both training (torch on the full batch) and inference (torch, single forward per
MCTS leaf) use this exact module. torch ships in the Kaggle sample notebook.

D_STATE = 101  (value_features.N_NET_FEATURES: 100 base + hc_tanh)
D_OPT   = 38   (extract_policy option_features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


D_STATE = 101
D_OPT = 38


class _ResBlock(nn.Module):
    """Pre-activation-ish residual MLP block: x + LN(W2 relu drop(LN(W1 x)))."""

    def __init__(self, d_model, p_drop):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = F.relu(self.ln1(self.fc1(x)))
        h = self.drop(h)
        h = self.ln2(self.fc2(h))
        return F.relu(x + h)


class AZNet(nn.Module):
    def __init__(self, d_state=D_STATE, d_opt=D_OPT, d_model=256, n_blocks=3, p_drop=0.1):
        super().__init__()
        self.d_state = d_state
        self.d_opt = d_opt
        self.d_model = d_model
        self.n_blocks = n_blocks

        if n_blocks > 0:
            # SCALED torso: input projection + residual MLP blocks.
            self.inp = nn.Sequential(
                nn.Linear(d_state, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
            )
            self.blocks = nn.ModuleList([_ResBlock(d_model, p_drop) for _ in range(n_blocks)])
            self.enc = None  # marker: use scaled path
        else:
            # LEGACY torso (single hidden layer) for loading old 68k checkpoints.
            self.enc = nn.Sequential(
                nn.Linear(d_state, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
            )
            self.inp = None
            self.blocks = None

        # value head: torso -> scalar (logit; sigmoid outside for prob)
        if n_blocks > 0:
            self.value = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
            )
            # policy head: [option feats | torso] -> per-option logit (deeper)
            self.policy = nn.Sequential(
                nn.Linear(d_opt + d_model, d_model),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
            )
        else:
            self.value = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
            )
            self.policy = nn.Sequential(
                nn.Linear(d_opt + d_model, d_model),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
            )

    def encode(self, x_state):
        if self.n_blocks > 0:
            h = self.inp(x_state)
            for b in self.blocks:
                h = b(h)
            return h
        return self.enc(x_state)

    def value_from_h(self, h):
        # clamp keeps the win-prob logit in a sane range (stops BCE blow-up)
        return self.value(h).squeeze(-1).clamp(-12.0, 12.0)  # logit

    def value_logit(self, x_state):
        return self.value_from_h(self.encode(x_state))

    def policy_logits_from_h(self, h, x_opt, mask=None):
        """h: (N, d_model)  x_opt: (N, M, d_opt)  mask: (N, M) 1=real 0=pad
        returns per-option logits (N, M) with pads set to -inf."""
        N, M, _ = x_opt.shape
        h_b = h.unsqueeze(1).expand(N, M, self.d_model)
        z = torch.cat([x_opt, h_b], dim=-1)
        logits = self.policy(z).squeeze(-1)  # (N, M)
        if mask is not None:
            logits = logits + (mask - 1.0) * 1e9
        return logits

    def forward(self, x_state, x_opt, mask=None):
        """Joint forward. Returns (policy_logits (N,M), value_logit (N,))."""
        h = self.encode(x_state)
        v = self.value_from_h(h)
        p = self.policy_logits_from_h(h, x_opt, mask)
        return p, v

    # ---- single-decision inference helpers (used by MCTS) ----
    @torch.no_grad()
    def eval_leaf(self, x_state_1, x_opt_1):
        """x_state_1: (D_STATE,)  x_opt_1: (K, D_OPT) for K legal options.
        Returns (policy_probs (K,) numpy, value_prob float in [0,1])."""
        self.eval()
        xs = x_state_1.unsqueeze(0)  # (1, D_STATE)
        h = self.encode(xs)  # (1, d_model)
        v = torch.sigmoid(self.value_from_h(h)).item()
        xo = x_opt_1.unsqueeze(0)  # (1, K, D_OPT)
        logits = self.policy_logits_from_h(h, xo)  # (1, K)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        return probs, v


def param_count(net):
    return sum(p.numel() for p in net.parameters())
