# ============================================================
# Lazy import utilities (no global optional imports at import time)
# ============================================================
from typing import Tuple, Dict, Any
import importlib
import importlib.util
from functools import lru_cache

import importlib.util as _imputil

from _old_code.tensor_algebra import _trim_trailing_nones

_PKG_CACHE: Dict[str, Any] = {}

# TODO: Add basepoint option as in signatory
# TODO: Implement keras runner

# ======================================
# Dispatcher (choose package + delegate)
# ======================================

def _pkg_available(name: str) -> bool:
    return _imputil.find_spec(name) is not None


def _normalize_package(package: Optional[str]) -> Optional[str]:
    """None/''/'auto'/'default' → no explicit request. 'no_package' stays literal."""
    if package is None:
        return None
    p = str(package).strip().lower()
    return None if p in ("", "auto", "default") else p


def _detect_backend_from_path(path) -> str:
    try:
        import torch
        if isinstance(path, torch.Tensor):
            return "torch"
    except Exception:
        pass
    try:
        import jax
        if isinstance(path, getattr(jax, "Array", ())):
            return "jax"
    except Exception:
        pass
    try:
        import jax.numpy as jnp
        if isinstance(path, getattr(jnp, "ndarray", ())):
            return "jax"
    except Exception:
        pass
    try:
        import tensorflow as tf
        if isinstance(path, tf.Tensor):
            return "tensorflow"
    except Exception:
        pass
    return "numpy"


def _normalize_backend(backend: Optional[str], path) -> str:
    if not backend or backend.lower() == "auto":
        return _detect_backend_from_path(path)
    return backend.lower()


def _normalize_device(device: Optional[str], backend: str, path) -> str:
    if not device or device.lower() == "auto":
        if backend == "torch":
            try:
                import torch
                if isinstance(path, torch.Tensor) and path.is_cuda:
                    return "cuda"
            except Exception:
                pass
        # default without probing
        return "cpu"
    d = device.lower()
    return d.split(":", 1)[0] if ":" in d else d


def _pip_hint(pkg: str) -> str:
    return {
        "signatory": "pip install signatory",
        "iisignature": "pip install iisignature",
        "signax": "pip install signax",
        "keras_sig": "pip install keras_sig",
    }.get(pkg, f"pip install {pkg}")


def _policy_candidates(backend: str, device: str) -> Tuple[str, list[str]]:
    """
    Return (optimal, candidates_in_order).
    Accelerator-only policies deliberately avoid CPU fallbacks when device demands one.
    """
    if backend == "torch":
        if device == "cuda":
            return "signatory", ["signatory"]  # accel required
        return "signatory", ["signatory", "iisignature", "no_package"]  # CPU ok

    if backend == "jax":
        if device in ("gpu", "tpu"):
            return "signax", ["signax"]  # accel required
        return "signax", ["signax", "iisignature", "no_package"]  # CPU ok

    if backend in ("tensorflow", "keras"):
        if device == "gpu":
            return "keras_sig", ["keras_sig"]  # accel required
        return "keras_sig", ["keras_sig", "iisignature", "no_package"]  # CPU ok

    # numpy & others → CPU
    return "iisignature", ["iisignature", "no_package"]


def _dispatch(pkg: str, *,  # single unified call-site (no duplication)
              logarithm: bool, path, trunc: int, backend: str, device: str,
              block_size: Optional[int], accumulate: bool, starting_point,
              output_starting_point: bool, identify_zero_levels: bool,
              memory_consumption: Literal["low", "high"] = "low", **kwargs,
              ):
    if pkg == "signatory":
        return _run_signatory(
            path, trunc,
            logarithm=logarithm, backend=backend, device=device,
            block_size=block_size, accumulate=accumulate,
            starting_point=starting_point, output_starting_point=output_starting_point,
            identify_zero_levels=identify_zero_levels, memory_consumption=memory_consumption, **kwargs
        )
    if pkg == "iisignature":
        return _run_iisignature(
            path, trunc,
            logarithm=logarithm, backend=backend, device=device,
            block_size=block_size, accumulate=accumulate,
            starting_point=starting_point, output_starting_point=output_starting_point,
            identify_zero_levels=identify_zero_levels, memory_consumption=memory_consumption, **kwargs
        )
    if pkg == "signax":
        if "_run_signax" not in globals():
            raise RuntimeError("signax chosen but `_run_signax` is not defined.")
        return _run_signax(
            path, trunc,
            logarithm=logarithm, backend=backend, device=device,
            block_size=block_size, accumulate=accumulate,
            starting_point=starting_point, output_starting_point=output_starting_point,
            identify_zero_levels=identify_zero_levels, memory_consumption=memory_consumption, **kwargs
        )
    if pkg == "keras_sig":
        if "_run_keras_sig" not in globals():
            raise RuntimeError("keras_sig chosen but `_run_keras_sig` is not defined.")
        return _run_keras_sig(
            path, trunc,
            logarithm=logarithm, backend=backend, device=device,
            block_size=block_size, accumulate=accumulate,
            starting_point=starting_point, output_starting_point=output_starting_point,
            identify_zero_levels=identify_zero_levels, memory_consumption=memory_consumption, **kwargs
        )
    if pkg == "no_package":
        return _run_no_package(
            path, trunc,
            logarithm=logarithm, backend=backend, device=device,
            block_size=block_size, accumulate=accumulate,
            starting_point=starting_point, output_starting_point=output_starting_point,
            identify_zero_levels=identify_zero_levels, memory_consumption=memory_consumption, **kwargs
        )
    raise RuntimeError(f"Internal error: unsupported package '{pkg}'.")


# ---------------- dispatcher ----------------

def _path_sig_dispatch(
        *,
        logarithm: bool,
        path,  # Array
        trunc: int,
        backend: Optional[str],  # "torch" | "jax" | "tensorflow" | "numpy" | "auto" | None
        device: Optional[str],  # "cpu" | "cuda" | "gpu" | "tpu" | "auto" | None
        package: Optional[str],  # explicit override; "auto"/None => choose best; allow "no_package"
        block_size: Optional[int],
        accumulate: bool,
        starting_point,
        output_starting_point: bool,
        identify_zero_levels: bool,
        print_package_info: bool = True,
        memory_consumption: Literal["low", "high"] = "low",
        **kwargs,
):
    """
    A: If optimal package is installed → print one info line and use it.
    B: If optimal isn't installed but a fallback is → print a recommendation + which fallback is used.
    C: If a specific package is requested but missing → raise.
    D: If requirements can't be satisfied (e.g., torch+cuda with no GPU-capable backend) → raise.

    Packages supported: signatory (Torch), signax (JAX), keras_sig (TF/Keras), iisignature (NumPy), no_package.
    """
    be = _normalize_backend(backend, path)
    dev = _normalize_device(device, be, path)
    req_pkg = _normalize_package(package)

    # (C) explicit package request
    if req_pkg is not None:
        if req_pkg == "no_package":
            if print_package_info:
                print(f"[signature] backend={be}, device={dev}, package=no_package (user-requested)")
            return _dispatch("no_package", logarithm=logarithm, path=path, trunc=trunc,
                             backend=be, device=dev, block_size=block_size, accumulate=accumulate,
                             starting_point=starting_point, output_starting_point=output_starting_point,
                             identify_zero_levels=identify_zero_levels, **kwargs)
        if not _pkg_available(req_pkg):
            raise RuntimeError(
                f"Requested package '{req_pkg}' is not installed. "
                f"Install it first, e.g. `{_pip_hint(req_pkg)}`."
            )
        if print_package_info:
            print(f"[signature] backend={be}, device={dev}, package={req_pkg} (user-requested)")
        return _dispatch(req_pkg, logarithm=logarithm, path=path, trunc=trunc,
                         backend=be, device=dev, block_size=block_size, accumulate=accumulate,
                         starting_point=starting_point, output_starting_point=output_starting_point,
                         identify_zero_levels=identify_zero_levels, **kwargs)

    # No explicit package → choose per policy
    optimal, candidates = _policy_candidates(be, dev)

    # Select first usable candidate: installed packages pass `_pkg_available`;
    # 'no_package' is always usable as last CPU fallback.
    chosen = None
    for c in candidates:
        if c == "no_package":
            chosen = "no_package"
            break
        if _pkg_available(c):
            chosen = c
            break

    # (D) unmet accelerator requirements (no CPU fallback allowed)
    requires_accel = (
            (be == "torch" and dev == "cuda") or
            (be == "jax" and dev in ("gpu", "tpu")) or
            (be in ("tensorflow", "keras") and dev == "gpu")
    )
    if chosen is None and requires_accel:
        raise RuntimeError(
            f"No accelerator-capable signature backend is installed for backend='{be}', device='{dev}'. "
            f"Install `{optimal}` (e.g., `{_pip_hint(optimal)}`)."
        )

    # If nothing installed but CPU is fine, fall back to no_package
    if chosen is None:
        chosen = "no_package"

    # (A/B) printing
    if print_package_info:
        if chosen == optimal:
            print(f"[signature] backend={be}, device={dev}, package={chosen} (optimal)")
        else:
            if optimal != "no_package":
                print(f"[signature] Recommended: {optimal} for backend={be}, device={dev} "
                      f"(install via `{_pip_hint(optimal)}`).")
            print(f"[signature] Using: package={chosen}")

    # Dispatch once
    return _dispatch(chosen, logarithm=logarithm, path=path, trunc=trunc,
                     backend=be, device=dev, block_size=block_size, accumulate=accumulate,
                     starting_point=starting_point, output_starting_point=output_starting_point,
                     identify_zero_levels=identify_zero_levels, memory_consumption=memory_consumption, **kwargs)


# =========================
# Public API (thin wrappers)
# =========================

def path_signature(
        path: "Array",
        *,
        trunc: int,
        backend: str = "auto",
        device: str = "auto",
        package: str = "auto",
        block_size: Optional[int] = None,
        accumulate: bool = True,
        starting_point: Optional["Elem"] = None,
        output_starting_point: bool = False,
        identify_zero_levels: bool = False,
        print_package_info: bool = False,
        memory_consumption: Literal["low", "high"] = "low",
        **kwargs,
) -> "Elem":
    """
    Compute the (Chen) signature of a path up to depth `trunc`.

    Parameters
    ----------
    path : Array
        Tensor/ndarray shaped (..., J, d) with J time-ordered points and d channels.
    trunc : int
        Truncation depth of the tensor algebra (≥ 0).
    backend : {"auto","torch","jax","tensorflow","numpy"}, default "auto"
        Used for type/device inference and package policy selection.
    device : {"auto","cpu","cuda",...}, default "auto"
        Target device hint (respected when the backend supports it).
    package : {"auto","signatory","iisignature","no_package"}, default "auto"
        Concrete implementation. "auto" selects based on (backend, device) policy and availability.
    block_size : int | None, default None
        If None or -1: compute a single signature of the whole path (no block axis).
        If 1: return per-step streamed signatures (block axis length S=J-1).
        If >1: split S steps into blocks of exactly `block_size` steps (S must be divisible).
    accumulate : bool, default True
        If True and `block_size > 1`: return cumulative *prefixes across blocks*.
        If False and `block_size > 1`: return *independent* per-block signatures.
        If `block_size in (None,-1)`, this flag is ignored (single result).
    starting_point : Elem | None, default None
        Left-multiplicative starting element in signature space (graded tensor). If provided,
        it is applied consistently to emitted results (see below).
    output_starting_point : bool, default False
        When a block/stream axis exists (`block_size >= 1`), prepend one extra entry representing
        the starting element (unit if `starting_point` is None). Ignored when `block_size in (None,-1)`.
    identify_zero_levels : bool, default False
        If True, mark/trim empty graded levels in the returned structure.
    print_package_info : bool, default False
        If True, prints which package/runner was used.
    memory_consumption : {"low","high"}, default "low"
        Only affects the **accumulating blocks** case (`accumulate=True` and `block_size>1`) for
        packages that support both strategies (e.g., "signatory"):
          - "high": do a single streamed pass and slice block-ends (fastest, higher peak memory).
          - "low": iterate blocks one-by-one, threading `initial` through (lower memory, more kernel calls).
        Ignored in other modes and by runners that do not implement both strategies.

    Returns
    -------
    Elem
        Graded tensor-algebra element:
          - If `block_size in (None,-1)`: shape (..., graded-dims) — no time/block axis.
          - If `block_size == 1`: shape (..., S[+1 if output_starting_point], graded-dims).
          - If `block_size > 1`: shape (..., B[+1 if output_starting_point], graded-dims),
            with B = S // block_size.

    Notes
    -----
    - For signatures, level-0 (scalar 1) is included in the returned graded element.
    - `starting_point` is represented in signature space; for per-block independent mode,
      it is left-multiplied to each block.
    """
    # Your existing dispatcher should pass through memory_consumption (and any **kwargs)
    return _path_sig_dispatch(
        path=path,
        trunc=trunc,
        backend=backend,
        device=device,
        package=package,
        logarithm=False,
        block_size=block_size,
        accumulate=accumulate,
        starting_point=starting_point,
        output_starting_point=output_starting_point,
        identify_zero_levels=identify_zero_levels,
        print_package_info=print_package_info,
        memory_consumption=memory_consumption,
        **kwargs,
    )


def path_logsignature(
        path: "Array",
        *,
        trunc: int,
        backend: str = "auto",
        device: str = "auto",
        package: str = "auto",
        block_size: Optional[int] = None,
        accumulate: bool = True,
        starting_point: Optional["Elem"] = None,
        output_starting_point: bool = False,
        identify_zero_levels: bool = False,
        print_package_info: bool = False,
        memory_consumption: Literal["low", "high"] = "low",
        mode: str = "brackets",
        **kwargs,
) -> "Elem":
    """
    Compute the logsignature of a path up to depth `trunc`.

    Parameters
    ----------
    path, trunc, backend, device, package, block_size, accumulate, starting_point,
    output_starting_point, identify_zero_levels, print_package_info
        See `path_signature` for general semantics.
    memory_consumption : {"low","high"}, default "low"
        Only affects the **accumulating blocks** case (`accumulate=True` and `block_size>1`)
        when supported by the runner. "high" uses a streamed signature pass and slices block ends;
        "low" threads `initial` block-by-block.
    mode : {"brackets","words","expand"}, default "brackets"
        Basis to use for the logsignature (when the runner supports multiple bases).
        For Signatory, "brackets" is the default and recommended.

    Returns
    -------
    Elem
        Graded free-Lie element (no level-0). Shapes follow the same block/stream rules
        as `path_signature`.
    """
    return _path_sig_dispatch(
        path=path,
        trunc=trunc,
        backend=backend,
        device=device,
        package=package,
        logarithm=True,
        block_size=block_size,
        accumulate=accumulate,
        starting_point=starting_point,
        output_starting_point=output_starting_point,
        identify_zero_levels=identify_zero_levels,
        print_package_info=print_package_info,
        memory_consumption=memory_consumption,
        mode=mode,
        **kwargs,
    )


# ==============================================
# Per-package runners (ONE method per package)
# ==============================================

def _run_no_package(
        path: Array,
        trunc: int,
        *,
        logarithm: bool,
        backend: str,
        device: str,
        block_size: Optional[int],
        accumulate: bool,
        starting_point: Optional[Elem],
        output_starting_point: bool,
        identify_zero_levels: bool,
        memory_consumption: Literal["low", "high"] = "low",
) -> Elem:
    """
    Pure tensor-algebra fallback (backend-agnostic).
    Uses your tensor_development + tensor_logarithm; supports blocks/accumulate/starting point.
    """
    X = [path]  # packed first-level path
    if not logarithm:
        return tensor_development(
            X,
            trunc=trunc,
            starting_point=starting_point,
            block_size=block_size,
            accumulate=accumulate,
            output_starting_point=output_starting_point,
            identify_zero_levels=identify_zero_levels,
            memory_consumption=memory_consumption,
        )
    else:
        S = tensor_development(
            X,
            trunc=trunc,
            starting_point=starting_point,
            block_size=block_size,
            accumulate=accumulate,
            output_starting_point=output_starting_point,
            identify_zero_levels=False,
            memory_consumption=memory_consumption,
        )
        # If multiple outputs (blocks/prefixes), apply logsig per element
        if isinstance(S, list) and S and isinstance(S[0], (list, tuple)):
            return [tensor_logarithm(e, trunc=trunc, identify_zero_levels=identify_zero_levels) for e in S]
        return tensor_logarithm(S, trunc=trunc, identify_zero_levels=identify_zero_levels)


# ----------------------------- iisignature runner -----------------------------

@lru_cache(maxsize=None)
def _iisig_module():
    return importlib.import_module("iisignature")


def _np_module():
    return importlib.import_module("numpy")


@lru_cache(maxsize=None)
def _iisig_prepare(d: int, depth: int, method: str = "O", extra: str = ""):
    """
    iisignature.prepare(d, depth, methods=...)
    method: one of {'D','C','O','A','S','X'}; we default to 'D'.
    extra: any extra flags (e.g. '2' to enable logsigtosig in some APIs).
    """
    iisig = _iisig_module()
    methods = (method or "")
    if extra:
        methods += extra
    return iisig.prepare(d, depth, methods or None)


def _to_numpy(x):
    np = _np_module()
    return np.asarray(x, dtype=np.float64)


def _split_blocks(J: int, steps_per_block: int):
    """Yield (t0, t1) indices for contiguous blocks; slice uses t0:t1+1."""
    S = J - 1
    B = S // steps_per_block
    for b in range(B):
        t0 = b * steps_per_block
        t1 = (b + 1) * steps_per_block
        yield t0, t1


def _pack_blocks_graded(blocks, identify_zero_levels: bool):
    """Pack list[Elem] into a single Elem with block axis -2 (level-0 scalar-safe)."""
    if not blocks:
        return []
    K = max((len(b) for b in blocks), default=0) - 1
    if K < 0:
        return []
    rep = next((arr for b in blocks for arr in b if arr is not None and hasattr(arr, "__array_namespace__")), None)
    if rep is None:
        out = [None] * (K + 1)
        return tensor_identify_zero_levels(out) if identify_zero_levels else _trim_trailing_nones(out)
    xp = rep.__array_namespace__()
    packed = []
    for k in range(K + 1):
        sample = next((b[k] for b in blocks if k < len(b) and b[k] is not None), None)
        if sample is None:
            packed.append(None);
            continue

        def _norm(a):
            if a is None: return None
            if not hasattr(a, "ndim"):
                a = xp.asarray(a)
            # level-0 scalar → expand to (...,1) so we can stack on axis=-2
            if k == 0 and a.ndim == sample.ndim - 1:
                return xp.expand_dims(a, -1)
            return a

        zeros = xp.zeros_like(_norm(sample))
        cols = [(_norm(b[k]) if k < len(b) and b[k] is not None else zeros) for b in blocks]
        packed.append(xp.stack(cols, axis=-2))
    return tensor_identify_zero_levels(packed) if identify_zero_levels else _trim_trailing_nones(packed)


def _run_iisignature(
        path: Array,
        trunc: int,
        *,
        logarithm: bool,  # False → signature (tensor), True → logsignature (free_lie)
        backend: str,  # info only (iisignature=CPU/NumPy)
        device: str,  # info only
        block_size: Optional[int],
        accumulate: bool,
        starting_point: Optional[Elem],
        output_starting_point: bool,
        identify_zero_levels: bool,
        logsig_method: str = "O",  # default 'D'; basis fixed (Lyndon)
        memory_consumption: str = "low",  # {"low","high"}; API ensures validity
        **kwargs
):
    """
    iisignature runner (NumPy core_old; Torch supported via small autograd bridges).

    Blocking & accumulate
    ---------------------
    • S = J-1. If block_size in {None, -1}: single whole-path result (no block axis).
    • If block_size == 1: per-step stream (S items).
    • If block_size > 1:
        – accumulate=False → independent per-block results (B = S // block_size).
        – accumulate=True  → block prefixes:
          · memory="low"  : signatures via `sigcombine` (Torch uses `sigcombinebackprop`);
                            logsignatures via `logsigjoin` (Torch uses `logsigjoinbackprop`).
          · memory="high" : signatures use one prefix stream `sig(..., format=2)` and slice ends;
                            logsignatures compute all `[0, t_j]` prefixes then slice.
                            (When `requires_grad=True` this falls back to "low" to keep autograd efficient.)

    Starting point
    --------------
    • Signatures: left-multiply the emitted sequence by `starting_point` (and optionally prepend it
      when `output_starting_point=True`). Logsignatures currently support only the unit start (BCH is not implemented).

    Gradients (Torch)
    -----------------
    • Signatures: each block’s signature uses `sig`/`sigbackprop`; block-prefix accumulation is done
      in flat space via a chain of `sigcombine` nodes whose backward calls `sigcombinebackprop`.
    • Logsignatures (memory="low"): prefixes are built by iterating `logsigjoin`; each join step is a
      differentiable node whose backward calls `logsigjoinbackprop`. Increments are formed with torch
      differencing so gradients flow back to the path naturally.
    • For memory="high", Torch with `requires_grad=True` degrades to the memory="low" path.
    """
    iisig = _iisig_module()
    import numpy as np
    depth = trunc

    # --- shape & blocking
    J, d = path.shape[-2], path.shape[-1]
    if J < 1 or d < 1:
        raise ValueError("path must have shape (..., J, d) with J≥1, d≥1")
    S = J - 1
    if S == 0:
        return ([] if logarithm else [1.0])

    if block_size in (None, -1):
        L = S
    else:
        if block_size <= 0 or (S % block_size != 0):
            raise ValueError(f"block_size must divide J-1; got J={J}, S={S}, block_size={block_size}")
        L = block_size

    # ---------- helpers ----------
    def _split_blocks(J_pts: int, L_steps: int):
        """Yield (t0, t1) over point indices with exactly L_steps increments."""
        if L_steps in (None, -1):
            yield (0, J_pts - 1)
            return
        B = (J_pts - 1) // L_steps
        for k in range(B):
            t0 = k * L_steps
            t1 = (k + 1) * L_steps
            yield (t0, t1)

    # --- collect flat outputs per block/prefix
    flats = []  # each flat has shape (..., total)

    if logarithm:
        # logsig: BCH starting_point not supported
        if starting_point is not None:
            raise NotImplementedError("logsignature with nontrivial starting_point (BCH) not implemented.")

        prep = _iisig_prepare(d, depth, method=logsig_method)

        if backend == "torch":
            import torch
            requires_grad = getattr(path, "requires_grad", False)

            # HIGH-memory (Torch): only when NOT requires_grad → compute all prefix logsigs and slice
            if accumulate and L >= 1 and memory_consumption == "high" and not requires_grad:
                p = path.detach().cpu().numpy()
                stream = [iisig.logsig(p[..., :2, :], prep)]
                for j in range(2, S + 1):
                    stream.append(iisig.logsig(p[..., :j + 1, :], prep))
                stream = np.stack(stream, axis=-2)  # (..., S, total)
                if L == 1:
                    stacked = stream
                else:
                    idx = np.arange(L - 1, S, L)
                    stacked = stream[..., idx, :]
                if output_starting_point:
                    total = stacked.shape[-1]
                    header = np.zeros((*stacked.shape[:-2], 1, total))
                    stacked = np.concatenate([header, stacked], axis=-2)
                if stacked.shape[-2] == 1 and not output_starting_point:
                    flat1 = stacked[..., 0, :]
                    tflat = torch.from_numpy(np.asarray(flat1)).to(path)
                    return lie_from_flat(tflat, dim=d, depth=depth)
                tstack = torch.from_numpy(np.asarray(stacked)).to(path)
                return lie_from_flat(tstack, dim=d, depth=depth)

            # LOW-memory Torch: stream via logsigjoin with autograd
            class _IILogSigPrefix(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    # one prefix logsig on a (prefix) path
                    out = iisig.logsig(x.detach().cpu().numpy(), prep)
                    return torch.from_numpy(np.asarray(out)).to(x)

                @staticmethod
                def backward(ctx, gout):
                    # Fallback single-prefix bp; not used in the join stream below
                    raise RuntimeError("IILogSigPrefix.backward should not be hit in logsigjoin stream path.")

            class _IILogSigJoin(torch.autograd.Function):
                @staticmethod
                def forward(ctx, prev_flat, inc_vec):
                    # prev_flat: (..., total_logsig), inc_vec: (..., d)
                    prev_np = prev_flat.detach().cpu().numpy()
                    inc_np = inc_vec.detach().cpu().numpy()
                    out = iisig.logsigjoin(prev_np, inc_np, prep)  # (..., total_logsig)
                    ctx.save_for_backward(prev_flat, inc_vec)  # needed for backprop
                    return torch.from_numpy(np.asarray(out)).to(prev_flat)

                @staticmethod
                def backward(ctx, gout):
                    prev_flat, inc_vec = ctx.saved_tensors
                    g_np = gout.detach().cpu().numpy()
                    prev_np = prev_flat.detach().cpu().numpy()
                    inc_np = inc_vec.detach().cpu().numpy()
                    g_prev, g_inc = iisig.logsigjoinbackprop(prev_np, inc_np, g_np, prep)
                    return (torch.from_numpy(np.asarray(g_prev)).to(prev_flat),
                            torch.from_numpy(np.asarray(g_inc)).to(inc_vec))

            # Build full per-step logsig stream via joins, with gradients
            inc = path[..., 1:, :] - path[..., :-1, :]  # (…, S, d) → autograd-friendly
            # first prefix logsig (two points)
            first = _IILogSigPrefix.apply(path[..., :2, :])
            stream = [first]
            for s in range(1, S):
                nxt = _IILogSigJoin.apply(stream[-1], inc[..., s, :])
                stream.append(nxt)
            # stack and slice per block
            kstream = torch.stack(stream, dim=-2)  # (..., S, total)
            if L == 1:
                stacked = kstream
            else:
                idx = torch.arange(L - 1, S, L, device=kstream.device)
                stacked = kstream.index_select(dim=-2, index=idx)
            if output_starting_point:
                total = stacked.shape[-1]
                hdr = torch.zeros((*stacked.shape[:-2], 1, total), dtype=stacked.dtype, device=stacked.device)
                stacked = torch.cat([hdr, stacked], dim=-2)
            # squeeze if single
            if stacked.shape[-2] == 1 and not output_starting_point:
                flat1 = stacked[..., 0, :]
                return lie_from_flat(flat1, dim=d, depth=depth)
            return lie_from_flat(stacked, dim=d, depth=depth)

        else:  # numpy backend
            p = np.asarray(path)

            if accumulate and L >= 1 and memory_consumption == "high":
                # build all prefix logsigs and slice
                stream = [iisig.logsig(p[..., :2, :], prep)]
                for j in range(2, S + 1):
                    stream.append(iisig.logsig(p[..., :j + 1, :], prep))
                stream = np.stack(stream, axis=-2)  # (..., S, total)
                if L == 1:
                    stacked = stream
                else:
                    idx = np.arange(L - 1, S, L)
                    stacked = stream[..., idx, :]
                if output_starting_point:
                    total = stacked.shape[-1]
                    header = np.zeros((*stacked.shape[:-2], 1, total))
                    stacked = np.concatenate([header, stacked], axis=-2)
                if stacked.shape[-2] == 1 and not output_starting_point:
                    flat1 = stacked[..., 0, :]
                    return lie_from_flat(flat1, dim=d, depth=depth)
                return lie_from_flat(stacked, dim=d, depth=depth)

            # LOW-memory NumPy: logsigjoin stream or per-block
            if accumulate and hasattr(iisig, "logsigjoin"):
                inc = p[..., 1:, :] - p[..., :-1, :]
                # first prefix (two points)
                flat = iisig.logsig(p[..., :2, :], prep)
                stream = [flat]
                for s in range(1, S):
                    flat = iisig.logsigjoin(flat, inc[..., s, :], prep)
                    stream.append(flat)
                stream = np.stack(stream, axis=-2)  # (..., S, total)
                if L == 1:
                    stacked = stream
                else:
                    idx = np.arange(L - 1, S, L)
                    stacked = stream[..., idx, :]
                if output_starting_point:
                    total = stacked.shape[-1]
                    stacked = np.concatenate([np.zeros((*stacked.shape[:-2], 1, total)), stacked], axis=-2)
                if stacked.shape[-2] == 1 and not output_starting_point:
                    flat1 = stacked[..., 0, :]
                    return lie_from_flat(flat1, dim=d, depth=depth)
                return lie_from_flat(stacked, dim=d, depth=depth)
            else:
                for (t0, t1) in _split_blocks(J, L):
                    pb = p[..., :t1 + 1, :] if accumulate else p[..., t0:t1 + 1, :]
                    flats.append(iisig.logsig(pb, prep))
                if output_starting_point:
                    total = flats[0].shape[-1]
                    flats = [np.zeros((*flats[0].shape[:-1], total))] + flats
                stacked = np.stack(flats, axis=-2)
                if stacked.shape[-2] == 1 and not output_starting_point:
                    flat1 = stacked[..., 0, :]
                    return lie_from_flat(flat1, dim=d, depth=depth)
                return lie_from_flat(stacked, dim=d, depth=depth)

    else:
        # -------- SIGNATURE --------
        if backend == "torch":
            import torch
            requires_grad = getattr(path, "requires_grad", False)

            # HIGH-memory (Torch): prefix stream via format=2 (only when NOT requires_grad)
            if accumulate and L >= 1 and memory_consumption == "high" and not requires_grad:
                p = path.detach().cpu().numpy()
                flat_stream = iisig.sig(p, depth, 2)  # (..., S, total) includes F0
                if L == 1:
                    stacked = flat_stream
                else:
                    idx = np.arange(L - 1, S, L)
                    stacked = flat_stream[..., idx, :]
                # header
                if output_starting_point:
                    total = stacked.shape[-1]
                    if starting_point is not None:
                        header = np.asarray(tensor_to_flat(starting_point, d, include_zero_level=True))[..., None, :]
                    else:
                        unit = np.zeros((*stacked.shape[:-2], 1, total))
                        unit[..., 0] = 1.0
                        header = unit
                    stacked = np.concatenate([header, stacked], axis=-2)
                # convert
                if stacked.shape[-2] == 1 and not output_starting_point:
                    flat1 = stacked[..., 0, :]
                    tflat = torch.from_numpy(np.asarray(flat1)).to(path)
                    out = tensor_from_flat(tflat, dim=d, insert_zero_level=1, identify_zero_levels=identify_zero_levels)
                else:
                    tstack = torch.from_numpy(np.asarray(stacked)).to(path)
                    out = tensor_from_flat(tstack, dim=d, insert_zero_level=1,
                                           identify_zero_levels=identify_zero_levels)
                # left-multiply by starting_point (preserve semantics)
                if accumulate and starting_point is not None:
                    out = tensor_product(starting_point, out, trunc=trunc, identify_zero_levels=False)
                return out

            # LOW-memory Torch with autograd:
            class _IISigBlock(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x_block):
                    flat = iisig.sig(x_block.detach().cpu().numpy(), depth, 0)  # includes F0
                    ctx.save_for_backward(x_block)
                    return torch.from_numpy(np.asarray(flat)).to(x_block)

                @staticmethod
                def backward(ctx, gout):
                    (x_block,) = ctx.saved_tensors
                    g = iisig.sigbackprop(x_block.detach().cpu().numpy(),
                                          gout.detach().cpu().numpy(),
                                          depth)
                    return torch.from_numpy(np.asarray(g)).to(x_block)

            class _IISigCombine(torch.autograd.Function):
                @staticmethod
                def forward(ctx, a_flat, b_flat):
                    # both (..., total)
                    a_np = a_flat.detach().cpu().numpy()
                    b_np = b_flat.detach().cpu().numpy()
                    c_np = iisig.sigcombine(a_np, b_np, d, depth)
                    ctx.save_for_backward(a_flat, b_flat)
                    return torch.from_numpy(np.asarray(c_np)).to(a_flat)

                @staticmethod
                def backward(ctx, gc):
                    a_flat, b_flat = ctx.saved_tensors
                    ga_np, gb_np = iisig.sigcombinebackprop(
                        a_flat.detach().cpu().numpy(),
                        b_flat.detach().cpu().numpy(),
                        gc.detach().cpu().numpy(),
                        d, depth
                    )
                    return (torch.from_numpy(np.asarray(ga_np)).to(a_flat),
                            torch.from_numpy(np.asarray(gb_np)).to(b_flat))

            if accumulate:
                # per-block signatures, then prefix via chained sigcombine nodes
                block_flats = []
                for (t0, t1) in _split_blocks(J, L):
                    block_flats.append(_IISigBlock.apply(path[..., t0:t1 + 1, :]))
                prefixes = []
                pref = None
                for bf in block_flats:
                    pref = bf if pref is None else _IISigCombine.apply(pref, bf)
                    prefixes.append(pref)
                flats = prefixes
            else:
                for (t0, t1) in _split_blocks(J, L):
                    flats.append(_IISigBlock.apply(path[..., t0:t1 + 1, :]))

            # header
            if output_starting_point:
                if starting_point is not None:
                    flats = [tensor_to_flat(starting_point, d, include_zero_level=True)] + flats
                else:
                    total = flats[0].shape[-1]
                    unit = torch.zeros((*flats[0].shape[:-1], total),
                                       dtype=flats[0].dtype, device=flats[0].device)
                    unit[..., 0] = 1
                    flats = [unit] + flats

            stacked = torch.stack(flats, dim=-2)  # (..., B, total)
            if stacked.shape[-2] == 1 and not output_starting_point:
                flat1 = stacked[..., 0, :]
                out = tensor_from_flat(flat1, dim=d, insert_zero_level=1, identify_zero_levels=identify_zero_levels)
            else:
                out = tensor_from_flat(stacked, dim=d, insert_zero_level=1, identify_zero_levels=identify_zero_levels)

            # left-multiply starting_point after graded conversion (same semantics as before)
            if accumulate and starting_point is not None:
                out = tensor_product(starting_point, out, trunc=trunc, identify_zero_levels=False)
            return out

        else:  # numpy backend signatures
            p = np.asarray(path)

            # HIGH-memory: prefix stream via format=2
            if accumulate and L >= 1 and memory_consumption == "high":
                flat_stream = iisig.sig(p, depth, 2)  # (..., S, total) includes F0
                if L == 1:
                    stacked = flat_stream
                else:
                    idx = np.arange(L - 1, S, L)
                    stacked = flat_stream[..., idx, :]
                if output_starting_point:
                    if starting_point is not None:
                        header = np.asarray(tensor_to_flat(starting_point, d, include_zero_level=True))[..., None, :]
                    else:
                        total = stacked.shape[-1]
                        unit = np.zeros((*stacked.shape[:-2], 1, total))
                        unit[..., 0] = 1.0
                        header = unit
                    stacked = np.concatenate([header, stacked], axis=-2)
                if stacked.shape[-2] == 1 and not output_starting_point:
                    flat1 = stacked[..., 0, :]
                    out = tensor_from_flat(flat1, dim=d, insert_zero_level=1, identify_zero_levels=identify_zero_levels)
                else:
                    out = tensor_from_flat(stacked, dim=d, insert_zero_level=1,
                                           identify_zero_levels=identify_zero_levels)
                if accumulate and starting_point is not None:
                    out = tensor_product(starting_point, out, trunc=trunc, identify_zero_levels=False)
                return out

            # LOW-memory NumPy: per-block + sigcombine
            if accumulate:
                pref = iisig.sig(p[..., :L + 1, :], depth, 0)
                flats.append(pref)
                for (t0, t1) in _split_blocks(J, L):
                    if t0 == 0:
                        continue
                    blk = iisig.sig(p[..., t0:t1 + 1, :], depth, 0)
                    pref = iisig.sigcombine(pref, blk, d, depth)
                    flats.append(pref)
            else:
                for (t0, t1) in _split_blocks(J, L):
                    flats.append(iisig.sig(p[..., t0:t1 + 1, :], depth, 0))

            if output_starting_point:
                if starting_point is not None:
                    flats = [tensor_to_flat(starting_point, d, include_zero_level=True)] + flats
                else:
                    total = flats[0].shape[-1]
                    unit = np.zeros((*flats[0].shape[:-1], total))
                    unit[..., 0] = 1.0
                    flats = [unit] + flats

            stacked = np.stack(flats, axis=-2)
            if stacked.shape[-2] == 1 and not output_starting_point:
                flat1 = stacked[..., 0, :]
                out = tensor_from_flat(flat1, dim=d, insert_zero_level=1, identify_zero_levels=identify_zero_levels)
            else:
                out = tensor_from_flat(stacked, dim=d, insert_zero_level=1, identify_zero_levels=identify_zero_levels)

            if accumulate and starting_point is not None:
                out = tensor_product(starting_point, out, trunc=trunc, identify_zero_levels=False)
            return out


# ----------------------------- signatory runner -----------------------------

# Catch initial without basepoint warning as we have overlapping paths
import warnings

warnings.filterwarnings(
    "ignore",  # or "once" if you prefer to see it the first time
    message=r"(?i).*initial.*basepoint.*",
    category=UserWarning,
    module=r"^signatory(\.|$)",
)


def _run_signatory(
        path: "Array",
        trunc: int,
        *,
        logarithm: bool,  # False → signature; True → logsignature
        backend: str,
        device: str,
        block_size: Optional[int],
        accumulate: bool,
        starting_point: Optional["Elem"],
        output_starting_point: bool,
        identify_zero_levels: bool,
        mode: str = "brackets",  # logsig basis (Signatory supports "brackets","words","expand")
        memory_consumption: str = "low",  # API guarantees one of {"low","high"}; no validation here
        **kwargs,
) -> "Elem":
    """
    Signatory runner (flat-internal, single conversion at return).

    Blocking & accumulate
    ---------------------
    • Steps S = J-1. If block_size in {None, -1}: compute a single whole-path result (no block axis).
    • If block_size == 1: return a per-step stream (S items).
    • If block_size > 1:
        – accumulate=False → independent per-block results (B = S // block_size).
        – accumulate=True  → block prefixes. With memory="high" we stream once and slice block ends;
                             with memory="low" we iterate blocks and thread the current state.

    Starting point
    --------------
    • Signatures: applied in signature space. We use `initial=` where appropriate (or left-multiply
      emitted items) and optionally prepend it when `output_starting_point=True`.
    • Logsignatures: non-unit start is supported by Sig→LogSig (we first compute signatures with the
      start applied, then convert; default logsig basis is `mode="brackets"`).

    Memory
    ------
    • memory="high": prefer a single streamed pass (prefixes) and slice endpoints for accumulate=True.
    • memory="low": compute per-block signatures and build prefixes by threading the state.

    Notes
    -----
    • Everything stays flat (no scalar term for signatures) until the very end; we then convert once
      to graded tensor / free-Lie outputs, and we can prepend a header row if requested.
    """
    import torch
    import signatory as siggy

    # --- Helpers (local; no external deps besides tensor_* converters already imported elsewhere) ---
    def _to_torch(x):
        if isinstance(x, torch.Tensor):
            return x
        import numpy as _np
        x = _np.asarray(x)
        return torch.from_numpy(x).to(dtype=torch.float64, device="cpu")

    def _flatten_leading(x_t):
        lead = x_t.shape[:-2]
        N = 1
        for s in lead: N *= int(s)
        return x_t.reshape(N, x_t.shape[-2], x_t.shape[-1]), lead

    def _unflatten_leading(x_t, lead):
        if x_t.ndim == 2:
            return x_t.reshape(*lead, x_t.shape[-1]) if lead else x_t
        else:
            return x_t.reshape(*lead, *x_t.shape[-2:]) if lead else x_t

    # --- Input normalisation ---
    x_t = _to_torch(path) if not isinstance(path, torch.Tensor) else path
    if device != "auto":
        x_t = x_t.to(device)  # trust caller/dispatcher here
    x_t, lead = _flatten_leading(x_t)
    N, L, C = x_t.shape
    if L < 1:
        raise ValueError("Path must have at least one point.")
    S = L - 1
    depth = int(trunc)

    # Single-point path: short-circuit
    if S == 0:
        if logarithm:
            # free-Lie unit → zero vector
            zeros = torch.zeros((N, 0), dtype=x_t.dtype,
                                device=x_t.device)  # flat size 0; converter will produce empties per level
            zeros = _unflatten_leading(zeros, lead)
            return lie_from_flat(zeros, dim=C, depth=depth)  # NOTE: replace if your name differs
        else:
            # signature unit → all zeros flat (since scalar term omitted)
            D = siggy.signature_channels(C, depth, scalar_term=False)
            zeros = torch.zeros((N, D), dtype=x_t.dtype, device=x_t.device)
            zeros = _unflatten_leading(zeros, lead)
            return tensor_from_flat(zeros, dim=C, insert_zero_level=1, identify_zero_levels=identify_zero_levels)

    # Starting point as FLAT signature (NO scalar term)
    sp_flat = None
    if starting_point is not None:
        # tensor_to_flat is part of your tensor algebra (include_zero_level=False → no scalar term)
        sp_flat = tensor_to_flat(starting_point, C, include_zero_level=False)
        if not isinstance(sp_flat, torch.Tensor):
            sp_flat = _to_torch(sp_flat)
        sp_flat = sp_flat.reshape(-1, sp_flat.shape[-1])  # (batch?, D)
        if sp_flat.shape[0] == 1 and N > 1:
            sp_flat = sp_flat.expand(N, -1)

    # --- (A) No blocks: single tensor (stream=False) ---
    if block_size in (None, -1):
        if not logarithm:
            flats = siggy.signature(x_t, depth, stream=False, initial=sp_flat, scalar_term=False)  # (N, D)
            flats = _unflatten_leading(flats, lead)
            return tensor_from_flat(flats, dim=C, insert_zero_level=1, identify_zero_levels=identify_zero_levels)
        else:
            if sp_flat is None:
                k = siggy.logsignature(x_t, depth, stream=False, mode=mode)  # (N, K)
            else:
                s = siggy.signature(x_t, depth, stream=False, initial=sp_flat, scalar_term=False)  # (N, D)
                k = siggy.signature_to_logsignature(s, channels=C, depth=depth,
                                                    stream=False, mode=mode, scalar_term=False)  # (N, K)
            k = _unflatten_leading(k, lead)
            return lie_from_flat(k, dim=C, depth=depth)

    # Validate blocks
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer, or None/-1 for no blocks.")
    if S % block_size != 0:
        raise ValueError(f"block_size={block_size} must divide steps S={S} exactly.")
    b = int(block_size)
    B = S // b

    # --- (B) Per-step: streamed (block_size == 1) ---
    if b == 1:
        if not logarithm:
            s_stream = siggy.signature(x_t, depth, stream=True, initial=sp_flat, scalar_term=False)  # (N, S, D)
            if output_starting_point:
                D = siggy.signature_channels(C, depth, scalar_term=False)
                head = sp_flat if sp_flat is not None else torch.zeros((N, D), dtype=x_t.dtype, device=x_t.device)
                s_stream = torch.cat([head.unsqueeze(1), s_stream], dim=1)  # (N, S+1, D)
            s_stream = s_stream.reshape(*lead, s_stream.shape[1], s_stream.shape[2])
            return tensor_from_flat(s_stream, dim=C, insert_zero_level=1, identify_zero_levels=identify_zero_levels)
        else:
            if sp_flat is None:
                k_stream = siggy.logsignature(x_t, depth, stream=True, mode=mode)  # (N, S, K)
            else:
                s_stream = siggy.signature(x_t, depth, stream=True, initial=sp_flat, scalar_term=False)
                k_stream = siggy.signature_to_logsignature(s_stream, channels=C, depth=depth,
                                                           stream=True, mode=mode, scalar_term=False)  # (N, S, K)
            if output_starting_point:
                K = siggy.logsignature_channels(C, depth)
                k_stream = torch.cat([torch.zeros((N, 1, K), dtype=x_t.dtype, device=x_t.device), k_stream], dim=1)
            k_stream = k_stream.reshape(*lead, k_stream.shape[1], k_stream.shape[2])
            return lie_from_flat(k_stream, dim=C, depth=depth)

    # --- (C) Blocks > 1 ---
    idx = torch.arange(b - 1, S, b, device=x_t.device)  # positions of block ends in the stream

    if accumulate:
        # (C1) Accumulating across blocks
        if memory_consumption == "high":
            # Stream once, slice block ends
            sig_stream = siggy.signature(x_t, depth, stream=True, initial=sp_flat, scalar_term=False)  # (N, S, D)
            if not logarithm:
                flats = sig_stream.index_select(1, idx)  # (N, B, D)
                if output_starting_point:
                    D = siggy.signature_channels(C, depth, scalar_term=False)
                    head = sp_flat if sp_flat is not None else torch.zeros((N, D), dtype=x_t.dtype, device=x_t.device)
                    flats = torch.cat([head.unsqueeze(1), flats], dim=1)  # (N, B+1, D)
                flats = flats.reshape(*lead, flats.shape[1], flats.shape[2])
                return tensor_from_flat(flats, dim=C, insert_zero_level=1, identify_zero_levels=identify_zero_levels)
            else:
                log_stream = siggy.signature_to_logsignature(sig_stream, channels=C, depth=depth,
                                                             stream=True, mode=mode, scalar_term=False)  # (N, S, K)
                logs = log_stream.index_select(1, idx)  # (N, B, K)
                if output_starting_point:
                    K = siggy.logsignature_channels(C, depth)
                    logs = torch.cat([torch.zeros((N, 1, K), dtype=x_t.dtype, device=x_t.device), logs], dim=1)
                logs = logs.reshape(*lead, logs.shape[1], logs.shape[2])
                return lie_from_flat(logs, dim=C, depth=depth)

        else:
            # "low" memory: iterate blocks, threading `initial`
            outs = []
            cur_init = sp_flat  # (N, D) or None (unit)
            for k in range(B):
                t0, t1 = k * b, (k + 1) * b
                block = x_t[:, t0:t1 + 1, :]  # (N, b+1, C)
                cur_sig = siggy.signature(block, depth, stream=False, initial=cur_init, scalar_term=False)  # (N, D)
                if logarithm:
                    out_k = siggy.signature_to_logsignature(cur_sig, channels=C, depth=depth,
                                                            stream=False, mode=mode, scalar_term=False)  # (N, K)
                else:
                    out_k = cur_sig
                outs.append(out_k)
                cur_init = cur_sig  # thread initial to the next block

            flats = torch.stack(outs, dim=1)  # (N, B, D/K)
            if output_starting_point:
                if not logarithm:
                    D = siggy.signature_channels(C, depth, scalar_term=False)
                    head = sp_flat if sp_flat is not None else torch.zeros((N, D), dtype=x_t.dtype, device=x_t.device)
                else:
                    K = siggy.logsignature_channels(C, depth)
                    head = torch.zeros((N, K), dtype=x_t.dtype, device=x_t.device)
                flats = torch.cat([head.unsqueeze(1), flats], dim=1)  # (N, B+1, ·)
            flats = flats.reshape(*lead, flats.shape[1], flats.shape[2])
            if not logarithm:
                return tensor_from_flat(flats, dim=C, insert_zero_level=1, identify_zero_levels=identify_zero_levels)
            else:
                return lie_from_flat(flats, dim=C, depth=depth)

    # (C2) Independent blocks (accumulate=False): Path.signature per block, then optional starting_point via combine
    P = siggy.Path(x_t, depth, scalar_term=False)
    items = [P.signature(k * b, (k + 1) * b + 1) for k in range(B)]  # each (N, D)
    flats = torch.stack(items, dim=1)  # (N, B, D)

    if not logarithm:
        if sp_flat is not None:
            # left-multiply each block by starting point
            out = torch.empty_like(flats)
            for k in range(B):
                out[:, k, :] = siggy.signature_combine(sp_flat, flats[:, k, :], C, depth, scalar_term=False)
            flats = out
        if output_starting_point:
            D = siggy.signature_channels(C, depth, scalar_term=False)
            head = sp_flat if sp_flat is not None else torch.zeros((N, D), dtype=x_t.dtype, device=x_t.device)
            flats = torch.cat([head.unsqueeze(1), flats], dim=1)  # (N, B+1, D)
        flats = flats.reshape(*lead, flats.shape[1], flats.shape[2])
        return tensor_from_flat(flats, dim=C, insert_zero_level=1, identify_zero_levels=identify_zero_levels)

    # logsig (independent blocks)
    s_in = flats
    if sp_flat is not None:
        # No general BCH available, do it via Sig→LogSig
        out = torch.empty_like(flats)
        for k in range(B):
            out[:, k, :] = siggy.signature_combine(sp_flat, flats[:, k, :], C, depth, scalar_term=False)
        s_in = out
    logs = siggy.signature_to_logsignature(s_in.reshape(-1, s_in.shape[-1]),
                                           channels=C, depth=depth, stream=False, mode=mode, scalar_term=False
                                           ).reshape(N, B, -1)
    if output_starting_point:
        K = siggy.logsignature_channels(C, depth)
        logs = torch.cat([torch.zeros((N, 1, K), dtype=x_t.dtype, device=x_t.device), logs], dim=1)
    logs = logs.reshape(*lead, logs.shape[1], logs.shape[2])
    return lie_from_flat(logs, dim=C, depth=depth)


# ----------------------------- signax runner -----------------------------

def _run_signax(
        path: Array,
        trunc: int,
        *,
        logarithm: bool,  # False → signature (tensor), True → logsignature-as-tensor (Lyndon, no level-0)
        backend: str,  # info only ("signax"/JAX)
        device: str,  # info only
        block_size: Optional[int],
        accumulate: bool,
        starting_point: Optional[Elem],
        output_starting_point: bool,
        identify_zero_levels: bool,
        memory_consumption: str = "low",  # {"low","high"}; validated at API level
        **kwargs,
):
    """
    Signax runner (JAX). Keeps Signax “levels” (list of per-level tensors) until the end; flattens once on return.

    Blocking & accumulate
    ---------------------
    • S = J−1. If block_size in {None, −1}: single whole-path result (no block axis).
    • If block_size == 1: per-step stream (S items).
    • If block_size > 1:
        – accumulate=False → independent per-block results (B = S // block_size).
        – accumulate=True  → block-prefixes:
            · memory="high": per-sample streamed signatures; slice block ends.
            · memory="low" : per-sample per-block signatures; prefix via per-sample `lax.scan` (Chen).

    Starting point
    --------------
    • Signatures: left-multiply emitted items in signature space; optionally prepend it when
      `output_starting_point=True`.
    • Logsignatures: computed *from signatures*; we apply the starting point in signature space first,
      then convert emitted signatures to logsig (no BCH inside the Lie algebra).

    Memory
    ------
    • "high": single streamed pass per-sample, then slice (fast when memory allows).
    • "low" : per-block per-sample compute + scan prefixing (lower peak memory).

    Logsignature note
    -----------------
    • Signax logsignatures are **Lyndon** coefficients. This runner returns them as a tensor-like
      graded object with **no level-0** (we do not return a free–Lie element here).
    """
    import jax
    import jax.numpy as jnp
    import signax as sgn

    depth = trunc
    J, d = path.shape[-2], path.shape[-1]
    if J < 1 or d < 1:
        raise ValueError("path must have shape (..., J, d) with J≥1, d≥1")
    S = J - 1
    if S == 0:
        return ([] if logarithm else [1.0])

    # ---- blocking params ----
    if block_size in (None, -1):
        L = S
    else:
        if block_size <= 0 or (S % block_size != 0):
            raise ValueError(f"block_size must divide J-1; got J={J}, S={S}, block_size={block_size}")
        L = block_size
    B = (S // L) if L > 0 else 1

    # =============== helpers (levels-native) ===============

    def _level_zeros(dtype):
        # list [Z1..ZK], Zi shape (d,)*i ; unit in signature space (F0=1, others 0)
        return [jnp.zeros((d,) * i, dtype=dtype) for i in range(1, depth + 1)]

    def _levels_flat_signature(levels):
        # concat per-level trailing (d,…,d); prepend F0=1
        flats = [Lk.reshape(Lk.shape[:-k] + (-1,)) for k, Lk in enumerate(levels, start=1)]
        flat = jnp.concatenate(flats, axis=-1) if len(flats) > 1 else flats[0]
        one = jnp.ones((*flat.shape[:-1], 1), dtype=flat.dtype)
        return jnp.concatenate([one, flat], axis=-1)

    def _levels_flat_logsig(levels):
        flats = [Lk.reshape(Lk.shape[:-k] + (-1,)) for k, Lk in enumerate(levels, start=1)]
        return jnp.concatenate(flats, axis=-1) if len(flats) > 1 else flats[0]

    def _sp_levels(sp_elem, *, dtype):
        # convert graded starting_point → Signax levels (drop level-0)
        sp_flat = tensor_to_flat(sp_elem, d, include_zero_level=True)
        sp_flat = jnp.asarray(sp_flat, dtype=dtype)
        vec = sp_flat[..., 1:]  # drop F0
        offs = 0
        out = []
        for k in range(1, depth + 1):
            sz = d ** k
            out.append(vec[..., offs:offs + sz].reshape((d,) * k))
            offs += sz
        return out

    def _prepend_header(levels_seq, header_levels):
        # Prepend header along the sequence axis (leading axis=0)
        out = []
        for k, seq_k in enumerate(levels_seq, start=1):
            h = header_levels[k - 1]
            shape = (1,) + seq_k.shape[1:]  # broadcast over any batch axes
            h_b = jnp.broadcast_to(h, shape)
            out.append(jnp.concatenate([h_b, seq_k], axis=0))
        return out

    def _slice_block_ends(stream_levels):
        # stream_levels leaves: (S, …) → pick L-1, 2L-1, ...
        if L == 1:
            return stream_levels
        idx = jnp.arange(L - 1, S, L)
        return [jnp.take(Lk, idx, axis=0) for Lk in stream_levels]  # (B, …)

    # ---- convert starting_point ONCE (levels), reuse everywhere ----
    sp_levels = _sp_levels(starting_point, dtype=path.dtype) if starting_point is not None else None

    # ---- flatten batch dims to a single axis N, process per-sample, then reshape back ----
    batch_shape = path.shape[:-2]
    N = int(jnp.prod(jnp.array(batch_shape))) if len(batch_shape) else 1
    path_N = path.reshape((N, J, d))

    # =============== per-sample kernels ===============

    def _blocks_for_one(p_1: jax.Array) -> jax.Array:
        # p_1: (J, d) → (B, L+1, d)
        return jnp.stack([p_1[k * L:(k + 1) * L + 1, :] for k in range(B)], axis=0)

    def _sig_blocks_for_one(p_1: jax.Array):
        # (J,d) → list levels with leaves (B, d,…,d)
        blocks = _blocks_for_one(p_1)  # (B, L+1, d)
        return jax.vmap(lambda q: sgn.signature(q, depth=depth, stream=False, flatten=False))(blocks)

    def _prefix_scan_for_one(sig_blocks_levels):
        # sig_blocks_levels: list with leaves (B, d,…,d) → prefixes: list with leaves (B, d,…,d)
        init = [jnp.zeros_like(Lk[0]) for Lk in sig_blocks_levels]  # identity (levels 1..K zeros)

        def step(carry, x):
            y = sgn.signature_combine(carry, x)
            return y, y

        _, seq = jax.lax.scan(step, init, sig_blocks_levels)  # (B, …) per-level
        return seq

    def _apply_start_to_seq(seq_levels):
        # seq_levels: list with leaves (T, d,…,d); left-multiply each by sp_levels (no batch)
        if sp_levels is None:
            return seq_levels
        return jax.vmap(lambda lev: sgn.signature_combine(sp_levels, lev))(seq_levels)

    def _sig_stream_for_one(p_1: jax.Array):
        # (J,d) → list levels with leaves (S, d,…,d)
        return sgn.signature(p_1, depth=depth, stream=True, flatten=False)

    def _finalize_seq(seq_levels, *, is_log: bool):
        # seq_levels leaves: (T, d,…,d)  → flat (T, total) then move T to -2 after vmapping
        if is_log:
            # convert each step/block to logsig levels, then flatten (no level-0)
            ls_levels = jax.vmap(sgn.signature_to_logsignature)(seq_levels)  # leaves: (T, d,…,d)
            flats = jax.vmap(_levels_flat_logsig)(ls_levels)  # (T, total)
            return flats
        else:
            flats = jax.vmap(_levels_flat_signature)(seq_levels)  # (T, total_with_F0)
            return flats

    if L in (None, -1) or L == S:  # =============== WHOLE PATH (no blocks) ===============
        def _one(p_1):
            sig = sgn.signature(p_1, depth=depth, stream=False, flatten=False)  # list, leaves (d,…,d)
            if sp_levels is not None:
                sig = sgn.signature_combine(sp_levels, sig)
            if logarithm:
                ls = sgn.signature_to_logsignature(sig)
                flat = _levels_flat_logsig(ls)
                if output_starting_point:
                    zero_h = _levels_flat_logsig([jnp.zeros_like(x) for x in ls])[None, ...]
                    seq = jnp.concatenate([zero_h, flat[None, ...]], axis=0)  # (2, total)
                else:
                    seq = flat
                return seq
            else:
                flat = _levels_flat_signature(sig)
                if output_starting_point:
                    unit_h = _levels_flat_signature(_level_zeros(sig[0].dtype))[None, ...] if sp_levels is None \
                        else _levels_flat_signature(sp_levels)[None, ...]
                    seq = jnp.concatenate([unit_h, flat[None, ...]], axis=0)  # (2, total)
                else:
                    seq = flat
                return seq

        outs = jax.vmap(_one)(path_N)  # shape: (N, total) or (N, 2, total)
        if outs.ndim == 2:  # (N, total)
            outs = outs.reshape(batch_shape + (outs.shape[-1],))
        else:  # (N, 2, total)
            outs = outs.reshape(batch_shape + (outs.shape[-2], outs.shape[-1]))
    elif not accumulate:  # =============== BLOCKS EXIST ===============
        # independent per-block signatures, per-sample
        def _one_blocks(p_1):
            sig_blocks = _sig_blocks_for_one(p_1)  # leaves: (B, d,…,d)
            if sp_levels is not None:
                sig_blocks = _apply_start_to_seq(sig_blocks)  # left-multiply each block
            flats = _finalize_seq(sig_blocks, is_log=logarithm)  # (B, total)
            if output_starting_point:
                hdr = (_levels_flat_logsig([jnp.zeros_like(x) for x in sig_blocks]) if logarithm
                       else _levels_flat_signature(_level_zeros(sig_blocks[0].dtype)))
                flats = jnp.concatenate([hdr[None, ...], flats], axis=0)  # (B+1, total)
            return flats

        outs = jax.vmap(_one_blocks)(path_N)  # (N, B or B+1, total)
        outs = outs.reshape(batch_shape + (outs.shape[-2], outs.shape[-1]))
    elif memory_consumption == "high":  # accumulate=True
        # per-sample stream → slice block ends → apply start → finalize
        def _one_stream(p_1):
            stream = _sig_stream_for_one(p_1)  # leaves: (S, d,…,d)
            picked = _slice_block_ends(stream)  # leaves: (B, d,…,d)
            if sp_levels is not None:
                picked = _apply_start_to_seq(picked)
            flats = _finalize_seq(picked, is_log=logarithm)  # (B, total)
            if output_starting_point:
                hdr = (_levels_flat_logsig([jnp.zeros_like(x) for x in picked]) if logarithm
                       else _levels_flat_signature(_level_zeros(picked[0].dtype)))
                flats = jnp.concatenate([hdr[None, ...], flats], axis=0)  # (B+1, total)
            return flats

        outs = jax.vmap(_one_stream)(path_N)  # (N, B or B+1, total)
        outs = outs.reshape(batch_shape + (outs.shape[-2], outs.shape[-1]))
    else:  # memory="low": per-sample per-block sigs → prefix via scan → apply start → finalize
        def _one_scan(p_1):
            sig_blocks = _sig_blocks_for_one(p_1)  # leaves: (B, d,…,d)
            prefixes = _prefix_scan_for_one(sig_blocks)  # leaves: (B, d,…,d)
            if sp_levels is not None:
                prefixes = _apply_start_to_seq(prefixes)
            flats = _finalize_seq(prefixes, is_log=logarithm)  # (B, total)
            if output_starting_point:
                hdr = (_levels_flat_logsig([jnp.zeros_like(x) for x in prefixes]) if logarithm
                       else _levels_flat_signature(_level_zeros(prefixes[0].dtype)))
                flats = jnp.concatenate([hdr[None, ...], flats], axis=0)  # (B+1, total)
            return flats

        outs = jax.vmap(_one_scan)(path_N)  # (N, B or B+1, total)
        outs = outs.reshape(batch_shape + (outs.shape[-2], outs.shape[-1]))

    if logarithm:
        return lie_from_flat(outs, dim=d, depth=depth)
    else:
        return tensor_from_flat(outs, dim=d, insert_zero_level=1, identify_zero_levels=identify_zero_levels)


def _run_keras_sig(
        path: Array,
        depth: int,
        *,
        logarithm: bool,
        backend: str,
        device: str,
        block_size: Optional[int],
        accumulate: bool,
        starting_point: Optional[Elem],
        output_starting_point: bool,
        identify_zero_levels: bool,
) -> Elem:
    """
    keras_sig runner (kept opt-in, not chosen by 'auto').
    Responsibilities:
      • Use its fast function for signatures; for logsig, compute signature then tensor_logarithm.
      • If blocks present and the package exposes chunk/join, use those; else per-block + Chen product.
    """
    # TODO: implement
    raise NotImplementedError("keras_sig path runner not implemented yet")
