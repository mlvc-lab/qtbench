# -*- coding: utf-8 -*-
"""Timestep grouping and progressive scheduling for FastDM calibration.

Ported from ``Fast_DM_PTQ/quant/utils.py`` with logging adapted to qtbench.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

__all__ = [
    "make_widths_from_adjacent_dist",
    "make_widths_fixedK",
    "plan_loop_sizes",
    "steps_per_epoch_like",
    "dp_distance_segments_from_adj",
]


def dp_distance_segments_from_adj(adj: np.ndarray, K: int) -> List[Tuple[int, int]]:
    """Partition a 1D adjacent-distance vector into K contiguous segments.

    Solves a dynamic program that minimises the squared deviation of each
    segment's accumulated distance from the global average ``total / K``.
    """
    assert adj.ndim == 1
    U = adj.shape[0] + 1
    assert 1 <= K <= U

    prefix_edges = np.zeros(U, dtype=np.float64)
    for i in range(U - 1):
        prefix_edges[i + 1] = prefix_edges[i] + adj[i]

    total_dist = prefix_edges[U - 1]
    target = total_dist / K if K > 0 else 0.0

    INF = 1e30
    dp = np.full((K + 1, U + 1), INF, dtype=np.float64)
    choice = np.full((K + 1, U + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0

    def seg_dist(s: int, i: int) -> float:
        if i - s <= 1:
            return 0.0
        return float(prefix_edges[i - 1] - prefix_edges[s])

    for k in range(1, K + 1):
        for i in range(k, U + 1):
            best = INF
            best_j = -1
            for j in range(k - 1, i):
                d = seg_dist(j, i)
                cost = dp[k - 1, j] + (d - target) ** 2
                if cost < best:
                    best = cost
                    best_j = j
            dp[k, i] = best
            choice[k, i] = best_j

    bounds = [U]
    cur_i = U
    for k in range(K, 0, -1):
        j = int(choice[k, cur_i])
        if j < 0:
            raise RuntimeError(f"DP backtrack failed at k={k}, i={cur_i}")
        bounds.append(j)
        cur_i = j
    bounds.sort()

    segments: List[Tuple[int, int]] = []
    for m in range(K):
        s = bounds[m]
        e = bounds[m + 1] - 1
        segments.append((s, e))

    assert len(segments) == K
    assert segments[0][0] == 0 and segments[-1][1] == U - 1
    return segments


def make_widths_from_adjacent_dist(
    adj_1d: torch.Tensor | np.ndarray,
    K: int,
    mode: str = "adaptive",
) -> torch.Tensor:
    """Compute per-bin widths over the U=adj_1d.shape[0]+1 unique timesteps.

    With ``mode="adaptive"`` the widths come from the DP segmentation of
    the supplied adjacent-distance vector. With ``mode="uniform"`` the
    distance vector is ignored and U is split as evenly as possible.

    Returned widths are flipped so that index 0 corresponds to the
    high-timestep (noisy) end -- consistent with the source repo.
    """
    if isinstance(adj_1d, torch.Tensor):
        U = int(adj_1d.shape[0]) + 1
    else:
        U = int(adj_1d.shape[0]) + 1

    if mode == "uniform":
        assert 1 <= K <= U
        q, r = divmod(U, K)
        widths = [q + 1] * r + [q] * (K - r)
        widths = widths[::-1]
        return torch.tensor(widths, dtype=torch.int64)

    if isinstance(adj_1d, torch.Tensor):
        adj_np = adj_1d.detach().cpu().numpy().astype(np.float64)
    else:
        adj_np = adj_1d.astype(np.float64)

    segments = dp_distance_segments_from_adj(adj_np, K)
    widths = [e - s + 1 for (s, e) in segments]
    widths = widths[::-1]
    return torch.tensor(widths, dtype=torch.int64)


def make_widths_fixedK(U: int, K: int, mode: str = "gradual") -> torch.Tensor:
    """Construct a fixed-K width vector using one of three shape profiles.

    ``mode="steep"`` skews bin widths so smaller bins land near the
    noisy end; ``"flat"`` keeps the widths nearly equal; ``"gradual"`` is
    the in-between default used by the source repo.
    """
    assert 1 <= K <= U
    q, r = divmod(U, K)

    if q == 1:
        L, M = r, K - r
        widths = [2] * L + [1] * M
        return torch.tensor(widths, dtype=torch.int64)

    if mode == "steep":
        N = (K - r) // 2
    elif mode == "flat":
        N = 0
    else:
        N = min(K // 3, (K - r) // 2)

    L = N + r
    M = K - L - N
    assert L >= 0 and M >= 0 and N >= 0, "invalid L/M/N"
    widths = [q + 1] * L + [q] * M + [q - 1] * N
    assert sum(widths) == U, "widths sum mismatch"
    return torch.tensor(widths, dtype=torch.int64)


def steps_per_epoch_like(N: int, bs: int) -> int:
    """Compute the number of optimiser steps per epoch for batch size ``bs``."""
    if N <= 0:
        return 0
    drop_last = N >= bs
    return (N // bs) if drop_last else ((N + bs - 1) // bs)


def plan_loop_sizes(
    uniq: torch.Tensor,
    counts: torch.Tensor,
    N0: int,
    progressive_direction: str,
    widths: torch.Tensor,
) -> list[int]:
    """Plan the cumulative calibration-set size for each progressive loop.

    For ``progressive_direction == "reverse"`` the k-th loop sees all
    samples whose timestep falls in the top-k bins (noisy end first). For
    ``"forward"`` the order is flipped (clean end first).
    """
    U = int(uniq.numel())
    widths = widths.to(torch.int64)

    assert widths.numel() >= 1, "widths must be non-empty"
    assert int(widths.sum().item()) == U, (
        f"sum(widths)={int(widths.sum().item())} != U={U}"
    )

    starts = torch.cat(
        [torch.tensor([0], dtype=counts.dtype), torch.cumsum(counts, dim=0)[:-1]]
    )

    K = int(widths.numel())
    n_list: list[int] = []

    if progressive_direction == "reverse":
        edges = torch.empty(K + 1, dtype=torch.int64)
        edges[-1] = U
        edges[:-1] = U - torch.cumsum(widths, dim=0)
        for k in range(K):
            u_lo = int(edges[k].item())
            pos = int(starts[u_lo].item())
            N = int(N0 - pos)
            if N > 0:
                n_list.append(N)
    elif progressive_direction == "forward":
        widths_f = torch.flip(widths, dims=[0])
        edges = torch.empty(K + 1, dtype=torch.int64)
        edges[0] = 0
        edges[1:] = torch.cumsum(widths_f, dim=0)
        for k in range(K):
            u_hi = int(edges[k + 1].item())
            if u_hi >= U:
                N = int(N0)
            else:
                pos = int(starts[u_hi].item())
                N = int(pos)
            if N > 0:
                n_list.append(N)
    else:
        raise ValueError(
            f"progressive_direction must be 'reverse' or 'forward', got {progressive_direction!r}"
        )

    return n_list
