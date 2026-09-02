"""Shared JAX backend hooks for bidegree representations."""

from __future__ import annotations


class _JaxBigradedBackend:
    """JAX array updates shared by both bidegree representations."""

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

    def _rank_scatter(self, output, target_ranks, values):
        return output.at[..., target_ranks, :].set(values)

    def _rank_scatter_add(self, output, target_ranks, values):
        return output.at[..., target_ranks, :].add(values)


__all__ = []
