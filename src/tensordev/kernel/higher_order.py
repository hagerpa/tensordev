from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import jax.numpy as jnp

from tensordev import Jax
from tensordev.core.jax import JaxSequentialCore
from tensordev.core.universal import DenseElemFirstOn
from tensordev.development.free import free_development
from tensordev.kernel.free import free_kernel
from tensordev.kernel.base_kernel import BaseKernel
from tensordev.util.path_preprocessing import DyadicOrder

_CORE = Jax()
_SEQ_CORE = JaxSequentialCore()
Array = jnp.ndarray
PathInput = Union[Array, DenseElemFirstOn]


def higher_order_kernel(
        X: PathInput,
        Y: PathInput,
        *,
        log_steps: tuple[int, int],
        log_degree: tuple[int, int],
        evaluate: Literal["terminal", "grid"] = "terminal",
        return_fg: bool = False,
        pairwise: bool = False,
        backend: Literal["scan", "wavefront"] = "scan",
        dyadic_order: DyadicOrder = 0,
        increment_input: bool = False,
        num_devices: int = 1,
):
    """
    Higher-order signature kernel via piecewise log-linear approximation.

    Input convention
    ----------------
    ``X`` and ``Y`` are paths whose positive tensor levels are packed as

        (X_1, ..., X_N),   (Y_1, ..., Y_M),

    with the time / steps axis on ``-2`` and flat tensor width on the last axis.

    A plain array is interpreted as a level-1 path, i.e. as ``(X_1,)``.

    If ``increment_input=False``, the inputs are interpreted as paths and are first
    converted to interval increments by differencing along axis ``-2``.

    If ``increment_input=True``, the inputs are interpreted directly as interval
    increments.

    Higher-order approximation
    --------------------------
    For each input separately:

    - split the interval sequence into contiguous blocks of size
      ``log_steps_x`` / ``log_steps_y``,
    - compute the block signatures truncated at degree
      ``log_degree_x`` / ``log_degree_y``,
    - take their tensor logarithms,
    - interpret these blockwise log-signatures as the increments of a
      piecewise log-linear path.

    These two piecewise log-linear paths are then passed to ``free_kernel``.

    Parameters
    ----------
    X, Y :
        Path inputs in packed positive-level form, with steps on axis ``-2``.

    log_steps :
        Tuple ``(log_steps_x, log_steps_y)`` of block sizes used to form the
        piecewise log-linear approximations of ``X`` and ``Y``.
        Each block size must divide the corresponding number of input intervals.

    log_degree :
        Tuple ``(log_degree_x, log_degree_y)`` of truncation degrees used for the
        block signatures / block log-signatures of ``X`` and ``Y``.

    evaluate, return_fg, pairwise, backend, dyadic_order :
        Passed through to ``free_kernel``.

    increment_input :
        If ``False``, interpret ``X`` and ``Y`` as paths and first difference them.
        If ``True``, interpret them directly as interval increments.

    core :
        Tensor algebra backend. Defaults to ``Jax()``.

    Returns
    -------
    Whatever ``free_kernel`` returns for the piecewise log-linear approximations.
    """
    log_steps_x, log_steps_y = log_steps
    log_degree_x, log_degree_y = log_degree
    # log_degree_x, log_degree_y = max(log_degree_x + 1, 0), max(log_degree_y + 1, 0)

    if log_steps_x <= 0:
        raise ValueError(f"log_steps[0] must be positive, got {log_steps_x}.")
    if log_steps_y <= 0:
        raise ValueError(f"log_steps[1] must be positive, got {log_steps_y}.")
    if log_degree_x < 1:
        raise ValueError(f"log_degree[0] must be at least 1, got {log_degree_x}.")
    if log_degree_y < 1:
        raise ValueError(f"log_degree[1] must be at least 1, got {log_degree_y}.")

    X = (jnp.asarray(X),) if not isinstance(X, (tuple, list)) else tuple(jnp.asarray(level) for level in X)
    Y = (jnp.asarray(Y),) if not isinstance(Y, (tuple, list)) else tuple(jnp.asarray(level) for level in Y)

    if not X:
        raise ValueError("Expected at least one positive tensor level in X.")
    if not Y:
        raise ValueError("Expected at least one positive tensor level in Y.")

    dx = X if increment_input else tuple(jnp.diff(level, axis=-2) for level in X)
    dy = Y if increment_input else tuple(jnp.diff(level, axis=-2) for level in Y)

    sx = dx[0].shape[-2]
    sy = dy[0].shape[-2]

    if sx % log_steps_x != 0:
        raise ValueError(
            f"log_steps[0]={log_steps_x} must divide the number of X-intervals {sx}."
        )
    if sy % log_steps_y != 0:
        raise ValueError(
            f"log_steps[1]={log_steps_y} must divide the number of Y-intervals {sy}."
        )

    sig_x = free_development(dx, increment_input=True, seq_core=_SEQ_CORE, trunc=log_degree_x, axis=-2,
                             block_size=log_steps_x, accumulate=False, output_starting_point=False, core=_CORE)
    sig_y = free_development(dy, increment_input=True, seq_core=_SEQ_CORE, trunc=log_degree_y, axis=-2,
                             block_size=log_steps_y, accumulate=False, output_starting_point=False, core=_CORE)

    log_x = _CORE.tensor_logarithm(
        sig_x[1:],
        trunc=log_degree_x,
        output_zero_level=False,
    )
    log_y = _CORE.tensor_logarithm(
        sig_y[1:],
        trunc=log_degree_y,
        output_zero_level=False,
    )

    if sx // log_steps_x == 1:
        log_x = tuple(level[..., None, :] for level in log_x)
    if sy // log_steps_y == 1:
        log_y = tuple(level[..., None, :] for level in log_y)

    return free_kernel(log_x, log_y, evaluate=evaluate, return_fg=return_fg, pairwise=pairwise,
                       backend=backend, dyadic_order=dyadic_order,
                       increment_in=True, num_devices=num_devices)


@dataclass(frozen=True)
class HigherOrderKernel(BaseKernel):
    """
    Higher-order kernel based on log-linear approximations of tensor-valued paths.

    This class evaluates the higher-order kernel determined by the chosen
    log-step and log-degree parameters and provides empirical kernel statistics
    such as batchwise kernel values, Gram matrices, MMD, and scoring rules.

    Input convention
    ----------------
    Inputs are tensor-valued paths in packed positive-level form.

    - A single array is interpreted as a level-1 tensor path.
    - A tuple/list is interpreted as packed positive tensor levels.
    - A missing leading sample axis is promoted to batch size ``1``.

    The parameters ``log_steps`` and ``log_degree`` determine the log-linear
    approximation used in the kernel evaluation.
    """

    log_steps: tuple[int, int]
    log_degree: tuple[int, int]
    backend: str = "scan"
    dyadic_order: DyadicOrder = 0
    increment_input: bool = False
    num_devices: int = 1

    def __call__(
            self,
            X,
            Y,
            *,
            evaluate: str = "terminal",
            return_fg: bool = False,
            pairwise: bool = False,
    ):
        """
        Evaluate the configured higher-order kernel.

        Parameters
        ----------
        X, Y :
            Normalized tensor-path inputs in packed positive-level form.
        evaluate : {"terminal", "grid"}, default="terminal"
            Whether to return only the terminal kernel values or the full discrete
            solution.
        return_fg : bool, default=False
            Whether to additionally return auxiliary outputs, if supported by the
            underlying solver.
        pairwise : bool, default=False
            Whether to evaluate batchwise or pairwise over the empirical samples.

        Returns
        -------
        Array or tuple
            Output of ``higher_order_kernel`` with the stored hyperparameters.
        """
        return higher_order_kernel(
            X,
            Y,
            evaluate=evaluate,
            return_fg=return_fg,
            pairwise=pairwise,
            log_steps=self.log_steps,
            log_degree=self.log_degree,
            backend=self.backend,
            dyadic_order=self.dyadic_order,
            increment_input=self.increment_input,
            num_devices=self.num_devices,
        )

    def _as_sample_batch(self, X):
        """
        Normalize tensor-path input to packed positive-level form with a leading
        empirical sample axis.

        Parameters
        ----------
        X :
            Either a single tensor level or a tuple/list of tensor levels. Each
            level is expected to have shape ``(batch, steps, width)`` or
            ``(steps, width)`` for a single sample.

        Returns
        -------
        tuple
            Tuple of arrays, each carrying the empirical sample axis on axis ``0``.

        Raises
        ------
        ValueError
            If no tensor levels are provided or if tensor levels inconsistently
            include a sample axis.
        """
        levels = (jnp.asarray(X),) if not isinstance(X, (tuple, list)) else tuple(jnp.asarray(z) for z in X)
        if not levels:
            raise ValueError("Expected at least one positive tensor level.")

        if all(level.ndim == 2 for level in levels):
            levels = tuple(level[None, ...] for level in levels)
        elif any(level.ndim == 2 for level in levels):
            raise ValueError("All levels must either all have a sample axis or all omit it.")

        return levels
