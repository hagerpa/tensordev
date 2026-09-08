"""Shared JAX backend hooks for bidegree layouts."""

from __future__ import annotations


class _JaxBigradedBackend:
    """JAX array updates shared by both bidegree layouts."""

    def _placement_scatter(self, output, target_ranks, values):
        return output.at[..., target_ranks, :, :].set(values)

    def _placement_scatter_add(self, output, target_ranks, values):
        return output.at[..., target_ranks, :, :].add(values)

    def _coordinate_scatter(self, output, target_indices, values):
        return output.at[..., target_indices].set(values)

    def make_sequential_core(self):
        from tensordev.core.jax import JaxSequentialCore

        sequential_core = getattr(self, "_sequential_core", None)
        if sequential_core is None:
            sequential_core = JaxSequentialCore()
            self._sequential_core = sequential_core
        return sequential_core


class _JaxPartiallySymmetrizedBigradedBackend(_JaxBigradedBackend):
    """Rank-axis updates required by partial symmetrization."""

    def _fmexp_first_level(self, g, z, *, trunc):
        from tensordev.core.bigraded.symmetrized._cpu_horner import (
            try_fused_horner,
        )

        fused = try_fused_horner(self, g, z, trunc)
        if fused is not None:
            return fused
        return super()._fmexp_first_level(g, z, trunc=trunc)

    def _rank_scatter(self, output, target_ranks, values):
        return output.at[..., target_ranks, :].set(values)

    def _rank_scatter_add(self, output, target_ranks, values):
        return output.at[..., target_ranks, :].add(values)


__all__ = []
