"""StateSpaceSignature — composition wrapper around FSSK."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp

from tensordev.core.universal import DenseElem, DenseElemFirstOn
from tensordev.sss.kernel import FSSK
from tensordev.sss.state_update import fssk_readout, fssk_state

Array = jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class StateSpaceSignature:
    """
    Thin wrapper around :class:`FSSK` that adds a truncation level and an
    optional hidden recursion state.

    Parameters
    ----------
    kernel:
        The underlying finite-state-space Volterra kernel.
    trunc:
        Tensor truncation level. Required. Static (changes cause retracing).
    state:
        Hidden recursion seed in **first-on format**: ``trunc`` levels where
        level ``r`` carries degree ``r+1`` and has trailing shape
        ``(q, 1, R, m**(r+1))``. Type: :data:`DenseElemFirstOn`.
        Defaults to ``None``, in which case a zero state is created.
    """

    kernel: FSSK
    trunc: int = field(metadata={"static": True})
    state: DenseElemFirstOn | None = field(default=None)

    def __post_init__(self) -> None:
        if self.trunc < 0:
            raise ValueError(f"trunc must be non-negative, got {self.trunc}.")

        if self.state is None:
            # Auto-initialise zero state from kernel dimensions.
            q, m, R = self.kernel.q, self.kernel.m, self.kernel.state_dim
            dtype = self.kernel.b.dtype
            zero_state: DenseElemFirstOn = tuple(
                jnp.zeros((q, 1, R, m ** (r + 1)), dtype=dtype)
                for r in range(self.trunc)
            )
            object.__setattr__(self, "state", zero_state)
        else:
            # Validate an explicitly supplied state.
            levels = tuple(self.state)
            if not levels and self.trunc > 0:
                raise ValueError("state must not be empty when trunc > 0.")
            if len(levels) != self.trunc:
                raise ValueError(
                    f"state must have exactly trunc={self.trunc} levels; "
                    f"got {len(levels)}."
                )

            q = self.kernel.q
            R = self.kernel.state_dim
            m = self.kernel.m
            for r, z in enumerate(levels):
                expected_tail = (q, 1, R, m ** (r + 1))
                if z.shape[-4:] != expected_tail:
                    raise ValueError(
                        f"state[{r}] has incompatible trailing shape: "
                        f"expected {expected_tail}, got {z.shape[-4:]}."
                    )

    # ------------------------------------------------------------------
    # Convenience constructors — thin wrappers around FSSK factories.
    # ------------------------------------------------------------------

    @classmethod
    def from_matrix(
            cls,
            *,
            trunc: int,
            state: DenseElemFirstOn | None = None,
            **kwargs,
    ) -> StateSpaceSignature:
        """Construct from a dense Lambda matrix. Forwards all kwargs to :meth:`FSSK.from_matrix`."""
        return cls(
            kernel=FSSK.from_matrix(**kwargs),
            trunc=trunc,
            state=state,
        )

    @classmethod
    def from_jordan(
            cls,
            *,
            trunc: int,
            state: DenseElemFirstOn | None = None,
            **kwargs,
    ) -> StateSpaceSignature:
        """Construct from Jordan block data. Forwards all kwargs to :meth:`FSSK.from_jordan`."""
        return cls(
            kernel=FSSK.from_jordan(**kwargs),
            trunc=trunc,
            state=state,
        )

    @classmethod
    def from_prony(
            cls,
            *,
            trunc: int,
            state: DenseElemFirstOn | None = None,
            **kwargs,
    ) -> StateSpaceSignature:
        """Construct from Prony coefficients. Forwards all kwargs to :meth:`FSSK.from_prony`."""
        return cls(
            kernel=FSSK.from_prony(**kwargs),
            trunc=trunc,
            state=state,
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
        returned instance. All other kernel parameters (``kernel``, ``trunc``
        are preserved unchanged.

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

    def readout(self, *, tau_dt: Array | float = 0.0) -> DenseElem:
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
        DenseElem
            Truncated Volterra signature. Level ``r`` has trailing shape
            ``(m**r,)``; level 0 is always the scalar unit ``1``.
        """

        return fssk_readout(self.state, kernel=self.kernel, tau_dt=tau_dt)

    def reset(self, new_state: DenseElemFirstOn | None = None) -> StateSpaceSignature:
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
            initial_state: DenseElemFirstOn | None = None,
            output_starting_state: bool = True,
            increment_input: bool = False,
    ) -> DenseElemFirstOn:
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
        DenseElemFirstOn
            State trajectory with a time axis of size ``n_blocks`` (or
            ``n_blocks + 1`` when ``output_starting_state=True``) at ``axis``.
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
        )

    def vsig(
            self,
            X: Array,
            *,
            dt: Array | float,
            axis: int = -2,
            block_size: int | None = None,
            accumulate: bool = True,
            initial_state: DenseElemFirstOn | None = None,
            output_starting_state: bool = False,
            tau_dt: Array | float = 0.0,
            increment_input: bool = False,
    ) -> DenseElem:
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
        DenseElem
            Volterra signature. Without blocking, level ``r`` has trailing
            shape ``(m**r,)``. With blocking, an extra block/time axis
            appears at ``axis``.
        """
        hidden = fssk_state(
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
        )
        return fssk_readout(hidden, kernel=self.kernel, tau_dt=tau_dt)


__all__ = ["StateSpaceSignature"]
