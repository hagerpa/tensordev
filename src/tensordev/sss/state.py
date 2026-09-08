"""StateSpaceSignature — composition wrapper around FSSK."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp

from tensordev.sss.kernel import FSSK
from tensordev.sss.state_update import (
    _maximum_order,
    _resolve_fssk_core_pair,
    _validate_fssk_core,
    _validate_state,
    _zero_state,
    fssk_readout,
    fssk_state,
    fssk_vsig,
)

Array = jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class StateSpaceSignature:
    """
    Thin wrapper around :class:`FSSK` that binds a truncation, algebra core,
    sequential core, and optional hidden recursion state.

    Parameters
    ----------
    kernel:
        The underlying finite-state-space Volterra kernel.
    trunc:
        Positive total-degree level or nonzero bidegree rectangle. Static
        changes cause retracing.
    state:
        Hidden recursion seed in first-on format. A total-degree core uses a
        tuple without degree zero; a bidegree core uses a scalar-omitting
        :class:`~tensordev.core.BigradedTensor`. Defaults to ``None``, in
        which case a zero state is created.
    core, seq_core:
        Algebra and sequential cores. Both default to the configured core
        pair and remain bound to the returned object.
    """

    kernel: FSSK
    trunc: Any = field(metadata={"static": True})
    state: Any = field(default=None)
    core: Any = field(default=None, metadata={"static": True})
    seq_core: Any = field(default=None, repr=False, metadata={"static": True})

    def __post_init__(self) -> None:
        core, seq_core = _resolve_fssk_core_pair(self.core, self.seq_core)
        trunc = core.normalize_truncation(self.trunc)
        _validate_fssk_core(
            core,
            q=self.kernel.q,
            m=self.kernel.m,
            feature="StateSpaceSignature",
        )
        if _maximum_order(trunc) <= 0:
            raise ValueError(f"trunc must be positive, got {trunc}.")
        object.__setattr__(self, "core", core)
        object.__setattr__(self, "seq_core", seq_core)
        object.__setattr__(self, "trunc", trunc)

        if self.state is None:
            object.__setattr__(
                self,
                "state",
                _zero_state(
                    core=core,
                    trunc=trunc,
                    q=self.kernel.q,
                    R=self.kernel.state_dim,
                    m=self.kernel.m,
                    dtype=self.kernel.b.dtype,
                ),
            )
        else:
            _validate_state(
                self.state,
                core=core,
                trunc=trunc,
                q=self.kernel.q,
                R=self.kernel.state_dim,
                m=self.kernel.m,
                name="state",
            )

    # ------------------------------------------------------------------
    # Convenience constructors — thin wrappers around FSSK factories.
    # ------------------------------------------------------------------

    @classmethod
    def from_matrix(
            cls,
            *,
            trunc: Any,
            state: Any = None,
            core: Any = None,
            seq_core: Any = None,
            **kwargs,
    ) -> StateSpaceSignature:
        """Construct from a dense Lambda matrix. Forwards all kwargs to :meth:`FSSK.from_matrix`."""
        return cls(
            kernel=FSSK.from_matrix(**kwargs),
            trunc=trunc,
            state=state,
            core=core,
            seq_core=seq_core,
        )

    @classmethod
    def from_jordan(
            cls,
            *,
            trunc: Any,
            state: Any = None,
            core: Any = None,
            seq_core: Any = None,
            **kwargs,
    ) -> StateSpaceSignature:
        """Construct from Jordan block data. Forwards all kwargs to :meth:`FSSK.from_jordan`."""
        return cls(
            kernel=FSSK.from_jordan(**kwargs),
            trunc=trunc,
            state=state,
            core=core,
            seq_core=seq_core,
        )

    @classmethod
    def from_prony(
            cls,
            *,
            trunc: Any,
            state: Any = None,
            core: Any = None,
            seq_core: Any = None,
            **kwargs,
    ) -> StateSpaceSignature:
        """Construct from Prony coefficients. Forwards all kwargs to :meth:`FSSK.from_prony`."""
        return cls(
            kernel=FSSK.from_prony(**kwargs),
            trunc=trunc,
            state=state,
            core=core,
            seq_core=seq_core,
        )

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update_with_path(
            self,
            X: Array,
            *,
            dt: Array | float,
            axis: int = -2,
            increment_input: bool = False,
    ) -> StateSpaceSignature:
        """Process a multi-step path ``X`` and return a new instance with the terminal hidden state.

        Takes the current ``state`` as the seed, runs the FSSK recursion over
        all steps of ``X``, and stores the resulting terminal state in the
        returned instance. The kernel, truncation, and cores are preserved.

        Parameters
        ----------
        X:
            Path nodes or increments, shape ``(..., S, d)`` by default.
            The step axis is selected by ``axis`` and the trailing axis must
            match ``self.kernel.path_dim``.
        dt:
            Step size(s). Scalar, ``(S,)`` array, or matching batch/step axes
            of ``X`` without the trailing coordinate axis.
        axis:
            Step axis of ``X`` (default ``-2``).
        increment_input:
            Pass ``True`` if ``X`` already contains increments rather than
            path nodes (skips the internal :func:`jnp.diff`).

        Returns
        -------
        StateSpaceSignature
            New instance identical to ``self`` except ``state`` is replaced by
            the terminal hidden state after processing ``X``.
        """
        terminal = fssk_state(
            X,
            kernel=self.kernel,
            dt=dt,
            trunc=self.trunc,
            axis=axis,
            block_size=None,
            accumulate=True,
            initial_state=self.state,
            output_starting_state=False,
            increment_input=increment_input,
            core=self.core,
            seq_core=self.seq_core,
        )
        return replace(self, state=terminal)

    def update_with_increment(
            self,
            dx: Array,
            *,
            dt: Array | float,
    ) -> StateSpaceSignature:
        """Advance the state by a single increment ``dx``.

        Convenience wrapper around :meth:`update_with_path` for the common
        online case where increments arrive one at a time.

        Parameters
        ----------
        dx:
            A single path increment with shape ``(..., d)`` where ``d`` must
            match ``self.kernel.path_dim``. No step axis is expected.
        dt:
            Scalar step size for this increment.

        Returns
        -------
        StateSpaceSignature
            New instance with ``state`` advanced by one step.
        """
        # Insert a singleton step axis: (..., d) -> (..., 1, d).
        dx = jnp.asarray(dx)
        return self.update_with_path(
            dx[..., None, :],
            dt=dt,
            axis=-2,
            increment_input=True,
        )

    # ------------------------------------------------------------------
    # Readout
    # ------------------------------------------------------------------

    def readout(self, *, tau_dt: Array | float = 0.0) -> Any:
        """Read out the truncated Volterra signature from the current hidden state.

        Evaluates the linear readout

            ``1 + sum_p Z^p · exp(-Lambda · tau_dt) · b_p``

        where ``Z^p`` are the stored hidden state levels.

        Parameters
        ----------
        tau_dt:
            Non-negative readout lag ``tau - t``. Scalar or arbitrary batch
            shape; broadcasts against the state leading axes.

        Returns
        -------
        tuple or BigradedTensor
            Truncated Volterra signature in the bound core's native layout and
            coordinates. The scalar block is the unit ``1``.
        """

        return fssk_readout(
            self.state,
            kernel=self.kernel,
            tau_dt=tau_dt,
            core=self.core,
        )

    def reset(self, new_state: Any = None) -> StateSpaceSignature:
        """Return a copy with the hidden state reset.

        Parameters
        ----------
        new_state:
            Replacement state in first-on format. When ``None`` (default)
            the state is reset to zeros via :meth:`__post_init__`.

        Returns
        -------
        StateSpaceSignature
            New instance identical to ``self`` except ``state`` is replaced.
        """
        return replace(self, state=new_state)

    # ------------------------------------------------------------------
    # Direct computation (read-only — self.state is not modified)
    # ------------------------------------------------------------------

    def states(
            self,
            X: Array,
            *,
            dt: Array | float,
            axis: int = -2,
            block_size: int | None = None,
            accumulate: bool = True,
            initial_state: Any = None,
            output_starting_state: bool = True,
            increment_input: bool = False,
    ) -> Any:
        """Return the hidden-state trajectory over ``X``.

        A thin, read-only wrapper around :func:`fssk_state`. ``self.state`` is
        used as the recursion seed by default but is **never** modified.

        Parameters
        ----------
        X:
            Path nodes or increments ``(..., S, d)``; step axis ``axis``.
        dt:
            Step size(s), same conventions as :meth:`update_with_path`.
        axis:
            Step axis of ``X`` (default ``-2``).
        block_size:
            Steps per emitted state block. ``None`` (default) emits a single
            block covering the full sequence.
        accumulate:
            Carry hidden state across blocks (default ``True``).
        initial_state:
            Explicit seed in first-on format. Defaults to ``self.state``.
        output_starting_state:
            Prepend the seed state to the output (default ``True``).
        increment_input:
            ``True`` if ``X`` already contains increments.

        Returns
        -------
        tuple or BigradedTensor
            First-on state trajectory in the bound core's native layout and
            coordinates, with the emitted block axis placed at ``axis``.
        """
        return fssk_state(
            X,
            kernel=self.kernel,
            dt=dt,
            trunc=self.trunc,
            axis=axis,
            block_size=block_size,
            accumulate=accumulate,
            initial_state=self.state if initial_state is None else initial_state,
            output_starting_state=output_starting_state,
            increment_input=increment_input,
            core=self.core,
            seq_core=self.seq_core,
        )

    def vsig(
            self,
            X: Array,
            *,
            dt: Array | float,
            axis: int = -2,
            block_size: int | None = None,
            accumulate: bool = True,
            initial_state: Any = None,
            output_starting_state: bool = False,
            tau_dt: Array | float = 0.0,
            increment_input: bool = False,
    ) -> Any:
        """Compute the Volterra signature of ``X``.

        Runs the FSSK recursion and applies the linear readout. ``self.state``
        is **never** modified.

        When ``block_size`` is ``None`` (default) and
        ``output_starting_state=False``, returns the signature at the single
        terminal time. Set ``block_size=1`` and ``output_starting_state=True``
        to obtain a full per-step signature trajectory.

        Parameters
        ----------
        X:
            Path nodes or increments ``(..., S, d)``; step axis ``axis``.
        dt:
            Step size(s), same conventions as :meth:`update_with_path`.
        axis:
            Step axis of ``X`` (default ``-2``).
        block_size:
            Steps per emitted block. ``None`` = full sequence.
        accumulate:
            Carry hidden state across blocks (default ``True``).
        initial_state:
            Explicit seed in first-on format. Defaults to ``self.state``.
        output_starting_state:
            Include the readout of the seed state (default ``False``).
        tau_dt:
            Non-negative readout lag ``tau - t``; broadcasts against batch axes.
        increment_input:
            ``True`` if ``X`` already contains increments.

        Returns
        -------
        tuple or BigradedTensor
            Volterra signature in the bound core's native layout and
            coordinates. With blocking, an emitted block axis appears at
            ``axis``.
        """
        return fssk_vsig(
            X,
            kernel=self.kernel,
            dt=dt,
            trunc=self.trunc,
            axis=axis,
            block_size=block_size,
            accumulate=accumulate,
            initial_state=self.state if initial_state is None else initial_state,
            output_starting_state=output_starting_state,
            tau_dt=tau_dt,
            increment_input=increment_input,
            core=self.core,
            seq_core=self.seq_core,
        )


__all__ = ["StateSpaceSignature"]
