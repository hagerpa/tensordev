# free_lie_algebra.py
from __future__ import annotations
from typing import List, Optional, Literal, Union

Array = object  # duck-typed (NumPy / Torch / JAX / CuPy)
LieElem = List[Optional[Array]]  # [L1, L2, ..., LN], None means zero at that degree

from _old_code.tensor_algebra import _zeros_and_concat
from _old_code.tensor_algebra import _reduce_along_axis, _trim_trailing_nones


# ---- Lyndon basis sizes via Witt numbers -------------------------------------

def _mobius_mu(n: int) -> int:
    """Möbius μ(n)."""
    if n == 1:
        return 1
    p, m, nn = 0, 1, n
    f = 2
    while f * f <= nn:
        if nn % f == 0:
            p += 1
            nn //= f
            if nn % f == 0:
                return 0  # squared prime factor
            m *= -1
        f += 1 if f == 2 else 2  # small speedup
    if nn > 1:
        m *= -1;
        p += 1
    return m


def _witt_numbers(d: int, depth: int) -> List[int]:
    """
    Witt formula: w_k = (1/k) * sum_{m|k} μ(m) * d^{k/m}
    Returns [w_1, ..., w_depth].
    """
    if d <= 0 or depth < 0:
        return []
    w = []
    for k in range(1, depth + 1):
        s = 0
        m = 1
        while m * m <= k:
            if k % m == 0:
                s += _mobius_mu(m) * (d ** (k // m))
                if m * m != k:
                    m2 = k // m
                    s += _mobius_mu(m2) * (d ** (k // m2))
            m += 1
        # we double-counted both divisors in the loop; correct by /2 then /k
        # but the above “pair add” already accounted properly; just divide by k:
        w.append(s // k)
    return w


# ---- I/O: flatten / unflatten for Lyndon basis --------------------------------

def lie_from_flat(flat: "Array", *, dim: int, depth: Optional[int] = None) -> "LieElem":
    """
    Convert a **flat logsig vector** (shape (..., total)) into graded levels
    `[L1, ..., LN]` where `Lk.shape[-1] == w_k` (Witt number for (dim, k)).

    Backend-free
    ------------
    No backend resolution is used. We rely on simple last-axis slicing:
    each level is taken as `flat[..., off:off+wk]` and thus stays on the same
    backend/device/dtype. Backends that support views will avoid copies.

    Parameters
    ----------
    flat : array
        Backend array with shape (..., total).
    dim : int
        Path dimension d (alphabet size), used to compute Witt numbers.
    depth : int, optional
        Maximum degree (inclusive). If omitted, inferred so that
        `total = w_1 + ... + w_depth`.

    Returns
    -------
    list[array]
        `[L1, ..., L_depth]` (no level-0). Slices/views of `flat` where possible.

    Raises
    ------
    TypeError
        If `flat` does not expose a `.shape`.
    ValueError
        If `dim < 1`, or `total` cannot be written as a sum of Witt numbers for `dim`,
        or the provided `depth` disagrees with `total`.
    """
    if not hasattr(flat, "shape"):
        raise TypeError("lie_from_flat expects an array with a .shape.")
    if dim < 1:
        raise ValueError("dim must be >= 1.")

    total = flat.shape[-1]

    # Infer depth if needed by summing Witt numbers until we match `total`.
    if depth is None:
        N, s = 0, 0
        # Guard to avoid runaway if `total` is incompatible.
        while s < total and N < 1024:
            N += 1
            s += _witt_numbers(dim, N)[-1]
        if s != total:
            raise ValueError(
                f"flat last-dim={total} cannot be written as a sum of Witt numbers for d={dim}."
            )
        depth = N

    w = _witt_numbers(dim, depth)
    if sum(w) != total:
        raise ValueError(
            f"flat last-dim={total} != sum_k w_k(d={dim}) up to depth={depth}."
        )

    # Slice per level along the last axis (keeps backend/device/dtype).
    out: "LieElem" = []
    off = 0
    for wk in w:
        out.append(flat[..., off:off + wk])
        off += wk
    return out



# ---- Basic Lie operations ------------------------------------------------------

def lie_add(
        a: LieElem,
        b: LieElem,
        *,
        trunc: Optional[int] = None,
) -> LieElem:
    """
    Level-wise addition in the free Lie algebra coordinates (Lyndon basis).

    (a ⊕ b)_k = a_k + b_k  for each degree k (treating `None` as zero).
    """
    K = (max(len(a), len(b)) if trunc is None else trunc)
    if K <= 0:
        return []
    out: LieElem = []
    for k in range(1, K + 1):
        ak = a[k - 1] if k - 1 < len(a) else None
        bk = b[k - 1] if k - 1 < len(b) else None
        if ak is None:
            out.append(bk)
        elif bk is None:
            out.append(ak)
        else:
            # last dims must match (same basis size w_k)
            if ak.shape[-1] != bk.shape[-1]:
                raise ValueError(f"degree {k} last-dim mismatch: {ak.shape[-1]} vs {bk.shape[-1]}")
            out.append(ak + bk)
    return _trim_trailing_nones(out)


def lie_scalar_multiply(levels: "LieElem", scalar) -> "LieElem":
    """
    Multiply each non-None level by a scalar, preserving `None`.

    Behavior
    --------
    For a graded element `levels = [L₀, L₁, …]`, this returns
        [ None if Lₖ is None else (Lₖ * scalar) for each k ].
    The function is backend-agnostic: it does not resolve or coerce types.
    Broadcasting, dtype, and device handling are entirely delegated to the
    underlying backend (NumPy, JAX, PyTorch, CuPy, …).

    `scalar` can be **any object** that your backend can broadcast-multiply
    with each non-None level (e.g., a Python number, a 0-D tensor/array,
    or a shape that’s broadcastable to the level’s shape).

    Parameters
    ----------
    levels : LieElem
        List of levels; each entry is an array/tensor or `None` (meaning exact zero).
    scalar : Any
        A value that supports broadcasted multiplication with the backend arrays.

    Returns
    -------
    LieElem
        New list with the same structure, where each non-None level is scaled.

    Notes
    -----
    • If `scalar` is from a different backend/device or not broadcastable to a level,
      the underlying backend will raise an error.
    """
    return [None if L is None else (L * scalar) for L in levels]


def lie_dilation(levels: LieElem, lam: Union[float, int, Array]) -> LieElem:
    """
    Dilate a free Lie-algebra element by a scalar λ:  L_k ↦ λ^k · L_k.

    Parameters
    ----------
    levels : sequence of Optional[array]
        Graded flat storage [L_0, ..., L_N]; each F_k has last-dim d**k or is None.
    lam : float
        Dilation factor λ. (E.g., scaling the underlying path by λ.)

    Returns
    -------
    list[Optional[array]]
        Dilated levels; trailing None levels are trimmed.

    Notes
    -----
    • Level 0 is not scaled (remains the scalar unit if present).
    • `None` levels remain `None`.
    """
    if not levels:
        return []
    if lam == 0.0:
        return []

    K = len(levels)
    if K < 1:
        return []

    out: List[Optional[Array]] = []
    for k in range(K):
        Lk = levels[k]
        if Lk is None:
            out.append(None)
        else:
            out.append(Lk * (lam ** (k + 1)))
    return _trim_trailing_nones(out)


# ---- Reductions along an axis (uses your _reduce_along_axis) -------------------

def lie_reduce(
        X: LieElem,  # packed per-degree arrays with a **time axis** at -2
        *,
        axis: int = -2,
        op: Literal["add"] = "add",
        block_size: Optional[int] = None,  # if provided, must evenly tile the axis length
        accumulate: bool = False,  # if True, return prefixes across blocks/items
        starting_point: Optional[LieElem] = None,  # element to add at the start (level-wise)
        output_starting_point: bool = False,
) -> Union[LieElem, List[LieElem]]:
    """
    Reduce a **sequence along `axis`** of Lie elements stored in **packed per-degree form**.

    Layout
    ------
    - Input is a graded list `[L1, ..., LN]`.
    - For each degree k, `Lk` is an array whose **second-to-last** axis (by default `-2`)
      indexes the sequence/time (e.g., blocks or steps). The last axis indexes Lyndon basis.
    - This function reduces along that axis by:
        • `op="add"`: level-wise addition (no BCH; for BCH see `lie_bch`, stub for now).

    Semantics
    ---------
    - If `block_size` is given, the axis is split into contiguous blocks of that size and
      each block is reduced; otherwise, a single full reduction is performed.
    - If `accumulate=True`, return **prefix reductions** of the (possibly block-reduced)
      sequence.
    - If `starting_point` is provided, it is **added** level-wise after reduction of each
      block/prefix. If `output_starting_point=True`, the `starting_point` itself is emitted
      as the first element.

    Returns
    -------
    LieElem | list[LieElem]
        A single Lie element if there is exactly one output and `output_starting_point=False`,
        otherwise a list of Lie elements. Trailing `None`s are trimmed.

    Notes
    -----
    - This delegates the actual scan/packing to your library’s `_reduce_along_axis` helper.
      A JIT/compiler can override `_reduce_along_axis` for parallel tree/scan without
      changing this API.
    """
    if op != "add":
        raise ValueError("lie_reduce currently supports only op='add' (BCH/⊞ to come).")

    def _pair(u: LieElem, v: LieElem) -> LieElem:
        return lie_add(u, v)

    # No apply_fn (pure additive)
    packed = _reduce_along_axis(
        X,
        axis=axis,
        pair_fn=_pair,
        apply_fn=None,
        block_size=block_size,
        accumulate=accumulate,
        starting_point=starting_point,
        output_starting_point=output_starting_point,
    )
    # _reduce_along_axis already packs per-degree; just trim trailing None per degree.
    if isinstance(packed, list) and packed and isinstance(packed[0], (list, tuple)):
        return [_trim_trailing_nones(e) for e in packed]
    return _trim_trailing_nones(packed)


# ---- Stubs to be filled later --------------------------------------------------

def lie_bracket(a: LieElem, b: LieElem, *, trunc: Optional[int] = None) -> LieElem:
    """
    [a, b] := a ⊗ b − b ⊗ a in the free Lie algebra (Lyndon basis coordinates).

    Not implemented yet. Will require the change-of-basis maps between tensor and Lie
    (via shuffle/half-shuffle) or a direct Lyndon combinatorics implementation.
    """
    raise NotImplementedError("lie_bracket is not implemented yet.")


def lie_bch(a: LieElem, b: LieElem, *, trunc: Optional[int] = None) -> LieElem:
    """
    Baker–Campbell–Hausdorff:
        log( exp(a) ⊗ exp(b) )  in the free Lie algebra.

    Not implemented yet. For now, compose in tensor space:
        S = exp(a) ⊗ exp(b)   (tensor algebra)
        return log(S)          (then map back to Lie if needed)
    """
    raise NotImplementedError("lie_bch is not implemented yet.")


# -----------------------------------------------------------------------------
# Section: Flattening -- requires backend resolution
# -----------------------------------------------------------------------------


def lie_to_flat(levels: "LieElem", *, fill_none_with_zeros: bool = True) -> "Array":
    """
    Concatenate `[L1, ..., LN]` along the last axis into a single flat vector (..., total).

    Parameters
    ----------
    levels : list[Optional[array]]
        Graded logsig levels (Lyndon coordinates), no level-0.
    fill_none_with_zeros : bool, default: True
        If True, replace `None` levels by **explicit zero-length slices** (width 0) on the
        same backend/dtype(/device) as the data; if False, `None` levels are skipped.
        (Widths for absent Lie levels are not inferred here.)

    Returns
    -------
    array
        Flat vector (..., total). Backend follows the first present array.

    Raises
    ------
    ValueError
        If the element is empty, or all levels are None, or batch shapes disagree.
    """
    lv = _trim_trailing_nones(levels)
    if not lv:
        raise ValueError("lie_to_flat: cannot flatten an empty Lie element")

    # Need at least one real array to infer backend/device/batch
    proto = next((L for L in lv if L is not None), None)
    if proto is None:
        raise ValueError("lie_to_flat: all levels are None; cannot infer backend/device for zeros")

    zeros, concat, batch = _zeros_and_concat(lv)

    parts = []
    for L in lv:
        if L is None:
            if not fill_none_with_zeros:
                continue
            parts.append(zeros(0))  # zero-length contribution, preserves dtype/device
        else:
            if L.shape[:-1] != batch:
                raise ValueError(
                    f"lie_to_flat: batch shape {L.shape[:-1]} differs from representative {batch}"
                )
            parts.append(L)

    return concat(parts) if len(parts) > 1 else parts[0]

