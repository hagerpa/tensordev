# Tensor algebra

This page covers truncated tensors in `tensordev`: how they are stored, the module-level
operations on them, and the cores that select total degree, bidegree, shear coordinates or
partial symmetrization. The examples on this page share one session that starts here:

```python
import jax
jax.config.update("jax_enable_x64", True)

import math
import numpy as np
import jax.numpy as jnp
import tensordev as td
from tensordev.util import random_trigonometric_polynomial_paths
```

## Truncated tensors

A truncated tensor of total degree $N$ over $\mathbb{R}^d$ is a tuple of levels
$(X_0, X_1, \dots, X_N)$. Level $k$ has shape `batch + (d**k,)`: leading batch axes, then the
$d^k$ coordinates of $(\mathbb{R}^d)^{\otimes k}$ on the last axis. With letters numbered from 0,
the word $i_1 \cdots i_k$ sits at index $\sum_{l=1}^{k} i_l\, d^{k-l}$, so the first letter
varies slowest. The scalar level $X_0$ has width 1.

The signature of a path that moves one unit along the first axis and then one unit along the
second shows the layout. Its level 2 holds the words 00, 01, 10 and 11. The word 01 (a step along
the first axis, then one along the second) has coefficient 1, and the word 10 has coefficient 0:

```python
L = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
sig_L = td.path_signature(L, trunc=2)

print([level.shape for level in sig_L])  # [(1,), (2,), (4,)]
print(sig_L[2])                          # [0.5 1.  0.  0.5]
```

Batch axes come first on every level:

```python
X = random_trigonometric_polynomial_paths(batch=4, steps=32, dim=2, key=0)
sig = td.path_signature(X, trunc=3)

print([level.shape for level in sig])  # [(4, 1), (4, 2), (4, 4), (4, 8)]
```

Four functions convert between layouts:

- [`tensor_to_flat`][tensordev.core.universal.Universal.tensor_to_flat] concatenates the levels
  on the last axis. `start_at_level_one=True` drops $X_0$.
- [`tensor_from_flat`][tensordev.core.universal.Universal.tensor_from_flat] splits a flat array
  back into levels. It infers $N$ from the width and `dim`, and always returns $X_0$:
  `insert_zero_level=None` keeps a stored $X_0$ or inserts zeros, `True` sets ones, `False` sets
  zeros, and a float sets that value.
- [`tensor_densify`][tensordev.core.universal.Universal.tensor_densify] fills levels given as
  `None` with zeros, including $X_0$. Use it to write a linear combination of words.
- [`tensor_slice`][tensordev.core.universal.Universal.tensor_slice] returns a proxy that applies
  an index to every level, for example to select paths or time steps.

```python
flat = td.tensor_to_flat(sig)
positive = td.tensor_to_flat(sig, start_at_level_one=True)
print(flat.shape, positive.shape)  # (4, 15) (4, 14)

back = td.tensor_from_flat(positive, dim=2, insert_zero_level=True)
np.testing.assert_allclose(td.tensor_to_flat(back), flat)

word_1 = td.tensor_densify((None, jnp.array([0.0, 1.0])))
print(word_1)  # (Array([0.], dtype=float64), Array([0., 1.], dtype=float64))

first_two = td.tensor_slice(sig)[:2]
print([level.shape for level in first_two])  # [(2, 1), (2, 2), (2, 4), (2, 8)]
```

A bidegree core stores a [`BigradedTensor`][tensordev.BigradedTensor] instead of a tuple; see
[Bidegree tensors](#bidegree-tensors). The same four functions apply to it through its core.

## Module-level operations

`td.tensor_product`, `td.tensor_exponential` and the other `td.tensor_*` names are bound methods
of the installed default core. Unless you change it, the default core is `Jax()` with no fixed
dimension and no fixed truncation. It accepts any $d$ and any total degree, so operations that
build a series need an explicit `trunc`.

```python
print(td.get_default_core())                                # Jax()
print(td.tensor_product.__self__ is td.get_default_core())  # True
```

| Operation | Computes |
|---|---|
| [`tensor_product`][tensordev.core.universal.Universal.tensor_product] | the truncated tensor (concatenation) product $A \otimes B$ |
| [`tensor_exponential`][tensordev.core.universal.Universal.tensor_exponential] | $\exp_\otimes(X) = \sum_{k \ge 0} X^{\otimes k} / k!$ for $X$ without scalar part; pass the levels from degree 1 |
| [`tensor_logarithm`][tensordev.core.universal.Universal.tensor_logarithm] | $\log_\otimes(1 + H) = \sum_{n \ge 1} (-1)^{n+1} H^{\otimes n} / n$; pass the levels from degree 1, the scalar part is taken to be 1 |
| [`tensor_shuffle_product`][tensordev.core.universal.Universal.tensor_shuffle_product] | the shuffle product of two linear combinations of words; needs shuffle plans |
| [`tensor_inner_product`][tensordev.core.universal.Universal.tensor_inner_product] | the Euclidean pairing, summed over all levels; returns a batch-shaped array |
| [`tensor_adjoint_product`][tensordev.core.universal.Universal.tensor_adjoint_product] | the adjoint of left (`side="left"`) or right multiplication by $W$ |

Several operations accept flags such as `a_first_on=True` for tuples that start at degree 1.
The [API reference](api.md#tensor-operations) lists all module-level operations.

The signature of a piecewise linear path is the product of the exponentials of its increments
(Chen's identity). For the corner path above:

```python
a = jnp.array([1.0, 0.0])
b = jnp.array([0.0, 1.0])

chen = td.tensor_product(
    td.tensor_exponential((a,), trunc=2),
    td.tensor_exponential((b,), trunc=2),
    trunc=2,
)
np.testing.assert_allclose(td.tensor_to_flat(chen), td.tensor_to_flat(sig_L))
```

The logarithm of a signature is its log-signature. For the corner path, level 2 is the
antisymmetric Lévy-area term $\tfrac12(a \otimes b - b \otimes a)$. Exponentiating the
log-signature recovers the signature:

```python
log_L = td.tensor_logarithm(sig_L[1:], trunc=2)
print(log_L[2])  # [ 0.   0.5 -0.5  0. ]

log_sig = td.tensor_logarithm(sig[1:], trunc=3, output_zero_level=False)
again = td.tensor_exponential(log_sig, trunc=3)
np.testing.assert_allclose(td.tensor_to_flat(again), td.tensor_to_flat(sig), atol=1e-12)
```

The shuffle product needs a bounded core with shuffle plans (see
[Shuffle plans and memory](#shuffle-plans-and-memory)). Words are tensors built with
`tensor_densify`. The shuffle of the words 0 and 1 is $01 + 10$, and pairing it with a signature
gives the product of the two increments:

```python
shuffle_core = td.make_core(dims=2, max_trunc=2, precompute_shuffle=True)

word_0 = td.tensor_densify((None, a))
word_1 = td.tensor_densify((None, b))
words = shuffle_core.tensor_shuffle_product(word_0, word_1, trunc=2)

print(words[2])                               # [0. 1. 1. 0.]
print(td.tensor_inner_product(words, sig_L))  # 1.0
print(td.tensor_inner_product(sig, sig).shape)  # (4,)
```

`tensor_adjoint_product(W, V, side="left")` is the adjoint of $Z \mapsto W \otimes Z$, and
`side="right"` is the adjoint of $Z \mapsto Z \otimes W$:

$$
\langle W \otimes Z, V \rangle = \langle Z, \mathrm{adj}_{\mathrm{left}}(W, V) \rangle,
\qquad
\langle Z \otimes W, V \rangle = \langle Z, \mathrm{adj}_{\mathrm{right}}(W, V) \rangle.
$$

In the notation of P. K. Friz and P. P. Hager,
[*Expected Signature Kernels for Rough Paths*](https://arxiv.org/abs/2509.07893), Section 2.2, these are
$\mathrm{adj}_{\mathrm{left}}(W, V) = \mathrm{L}_W V = \sum_w e_w \langle V, W \otimes e_w\rangle$ and
$\mathrm{adj}_{\mathrm{right}}(W, V) = \mathrm{R}_W V = \sum_w e_w \langle V, e_w \otimes W\rangle$, truncated at `trunc`.

```python
kw, kz, kv = jax.random.split(jax.random.PRNGKey(0), 3)
W = td.tensor_from_flat(jax.random.normal(kw, (7,)), dim=2)   # degrees 0 to 2
Z = td.tensor_from_flat(jax.random.normal(kz, (7,)), dim=2)   # degrees 0 to 2
V = td.tensor_from_flat(jax.random.normal(kv, (31,)), dim=2)  # degrees 0 to 4

left = td.tensor_adjoint_product(W, V, trunc=2, side="left")
right = td.tensor_adjoint_product(W, V, trunc=2, side="right")

np.testing.assert_allclose(
    td.tensor_inner_product(td.tensor_product(W, Z, trunc=4), V),
    td.tensor_inner_product(Z, left),
)
np.testing.assert_allclose(
    td.tensor_inner_product(td.tensor_product(Z, W, trunc=4), V),
    td.tensor_inner_product(Z, right),
)
```

## Cores

--8<-- "README.md:core_overview"

The rows of this table select these core classes:

| Core | Class |
|---|---|
| dense total degree | [`Jax`][tensordev.Jax] |
| bidegree | [`JaxBigraded`][tensordev.JaxBigraded] |
| bidegree, `partially_symmetrized=True` | [`JaxPartiallySymmetrizedBigraded`][tensordev.JaxPartiallySymmetrizedBigraded] |
| dense total-degree shear | [`JaxShearTotal`][tensordev.core.JaxShearTotal] |
| bidegree shear | [`JaxShearBigraded`][tensordev.core.JaxShearBigraded] |
| bidegree shear, `partially_symmetrized=True` | [`JaxPartiallySymmetrizedShearBigraded`][tensordev.JaxPartiallySymmetrizedShearBigraded] |

Other combinations of integers and pairs raise a `ValueError`; partial symmetrization requires
pairs for both `dims` and `max_trunc`. A bounded core uses its default truncation whenever
`trunc` is omitted, and rejects a truncation above its capacity with a `ValueError`.
`at_truncation` returns a view of the same core, with the same plans, and another default:

```python
core = td.make_core(dims=2, max_trunc=4, default_trunc=3)
print(core)                                   # Jax(d=2, max_trunc=4, default_trunc=3)
print(len(td.path_signature(X, core=core)))   # 4
print(core.at_truncation(2))                  # Jax(d=2, max_trunc=4, default_trunc=2)
```

## Constructing a core

--8<-- "README.md:core_example"

Here the path has three channels split as `dims=(2, 1)`. The installed core is a
`JaxPartiallySymmetrizedShearBigraded` with capacity `(4, 3)`, so `path_signature` returns a
`BigradedTensor` at the default truncation `(2, 2)`.

| Function | Returns | Default core | Use it to |
|---|---|---|---|
| [`make_core`][tensordev.make_core]`(**config)` | a new core | unchanged | pass `core=...` to a development, or call `core.tensor_*` |
| [`set_default_core`][tensordev.set_default_core]`(**config)` | the installed core | replaced; `td.tensor_*` are rebound | run a whole script in one configuration |
| `set_default_core(core)` | `core` | replaced | install a core built earlier |
| [`reset_default_core`][tensordev.reset_default_core]`()` | `None` | the `TENSORDEV_BACKEND` selection | undo `set_default_core` |
| [`shear_core`][tensordev.shear_core]`(core, ...)` | the matching shear core | unchanged | change the coordinates of a standard core |
| [`symmetrize_core`][tensordev.symmetrize_core]`(core, ...)` | the matching partially symmetrized core | unchanged | partially symmetrize an ordered bidegree core |

`set_default_core` raises a `TypeError` when a core is combined with construction arguments.
`shear_core` and `symmetrize_core` inherit the capacity, the default truncation and the shuffle
scope of their source unless you pass `precompute_shuffle`:

- `shear_core` accepts a standard `Jax`, `JaxBigraded` or `JaxPartiallySymmetrizedBigraded` core.
  A total-degree source does not record the alphabet split, so pass `dims=(d', d'')`; an
  unbounded `Jax()` also needs `max_trunc`. A bidegree source shares its compatible plan stores
  with the new core.
- `symmetrize_core` accepts an ordered bidegree core, `JaxBigraded` or `JaxShearBigraded`.
  Total-degree and already partially symmetrized sources raise a `TypeError`.

Both orders, `shear_core(symmetrize_core(core))` and `symmetrize_core(shear_core(core))`, give a
`JaxPartiallySymmetrizedShearBigraded` core.

### The default core

--8<-- "README.md:core_defaults"

```python
bounded = td.set_default_core(dims=2, max_trunc=3)
print(td.tensor_product.__self__ is bounded)  # True
print(len(td.path_signature(X)))              # 4
td.reset_default_core()
```

Some specialized entry points, such as `free_kernel` and `higher_order_kernel`, require a
standard total-degree default core and raise a `RuntimeError` under another default.

!!! note "Use `td.tensor_*` through the module"

    `set_default_core` rebinds the names on the `tensordev` module. A name imported earlier with
    `from tensordev import tensor_product` stays bound to the previous core. Call `td.tensor_*`,
    or call the methods of a core object, `core.tensor_product(...)`.

## Bidegree tensors

Split the $d$ path channels into the first $d'$ channels, with letters $\mathcal{A}'$, and the
last $d''$ channels, with letters $\mathcal{A}''$; this is `dims=(d', d'')`. A word has bidegree
$(n, m)$ when it has $n$ letters from $\mathcal{A}'$ and $m$ letters from $\mathcal{A}''$.
Bidegree truncation at $(N, M)$ keeps the words with $n \le N$ and $m \le M$:

$$
T^{N,M}(\mathbb{R}^d) = \bigoplus_{n=0}^{N} \bigoplus_{m=0}^{M} T^{(n,m)}(\mathbb{R}^d),
\qquad
T^{\min(N,M)}(\mathbb{R}^d) \subseteq T^{N,M}(\mathbb{R}^d) \subseteq T^{N+M}(\mathbb{R}^d).
$$

So a bidegree core keeps every word of length at most $\min(N, M)$ and some, but not all, words
up to length $N + M$. Concatenation and shuffle add bidegrees, so both descend to $T^{N,M}$, and
Chen's identity holds there. A total-degree tensor of degree $N + M$ has
$\sum_{k=0}^{N+M} d^k$ coordinates; the ordered bidegree tensor has

$$
\sum_{n=0}^{N} \sum_{m=0}^{M} \binom{n+m}{n} d'^{\,n} d''^{\,m}
$$

coordinates, where $\binom{n+m}{n}$ counts the placements of the $\mathcal{A}'$-letters in a
word.

--8<-- "README.md:bigraded_tensor"

Blocks are ordered by total degree, then by decreasing first degree. For a four-channel path
split as `(2, 2)`, the bidegree signature at `(3, 3)` is the projection of the total-degree
signature at degree 6:

```python
X4 = random_trigonometric_polynomial_paths(batch=3, steps=16, dim=4, key=1)
bidegree = td.make_core(dims=(2, 2), max_trunc=(3, 3))

sig_total = td.path_signature(X4, trunc=6)
sig_bi = td.path_signature(X4, core=bidegree)

print(td.tensor_to_flat(sig_total).shape)    # (3, 5461)
print(bidegree.tensor_to_flat(sig_bi).shape)  # (3, 2229)
print(sum(math.comb(n + m, n) * 2**n * 2**m for n in range(4) for m in range(4)))  # 2229

print(sig_bi.grades[:6])  # ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
print(sig_bi[2, 1].shape)  # (3, 24)

projected = bidegree.tensor_from_total(sig_total)
for grade in sig_bi.grades:
    np.testing.assert_allclose(sig_bi[grade], projected[grade], rtol=1e-10, atol=1e-12)
```

`tensor_to_total` goes the other way. It returns dense total levels up to degree $N + M$, with
zeros at the words that bidegree truncation discards. `BigradedTensor` is a JAX PyTree with one
leaf per block, and its [`BigradedSpec`][tensordev.BigradedSpec] records the split, the
truncation, the coordinates and whether the tensor is partially symmetrized.

## Shuffle plans and memory

--8<-- "README.md:core_plans"

| `precompute_shuffle` | Retained shuffle plans | Cores |
|---|---|---|
| `False` (default) | none | all |
| `"generator"` | shuffles with a first-level factor | bidegree and shear cores |
| `True` | all shuffles within capacity; required by `tensor_shuffle_product` | all bounded cores |

!!! warning "Shuffle products without plans raise"

    `tensor_shuffle_product` on a core built without `precompute_shuffle=True`, including the
    default `Jax()`, raises a `RuntimeError`. A standard total-degree core accepts only `True` or
    `False`; `"generator"` raises a `TypeError` there.

[`core_expected_memory`][tensordev.core_expected_memory] takes the configuration arguments of
`make_core`, except `default_trunc`, and counts the bytes of the plan buffers that the core
retains, without constructing it. It excludes Python-container overhead, runtime temporaries and
compiled executables. The count is exact, not an upper bound: below it equals `memory_bytes()` of
the constructed core. A shear core that `shear_core` builds from a bidegree core shares that core's
compatible plan stores, and `memory_bytes()` counts them in both cores; `symmetrize_core`, and
`shear_core` with a total-degree source, build their own plans.

```python
print(td.core_expected_memory(dims=2, max_trunc=4))  # 0.0

mib = td.core_expected_memory(dims=2, max_trunc=10, precompute_shuffle=True)
print(round(mib, 3))  # 0.718

parts = td.core_expected_memory(
    dims=2, max_trunc=10, precompute_shuffle=True, unit="bytes", breakdown=True,
)
print(parts)
# {'axis_permutations': 10776.0, 'derived_execution_maps': 712704.0,
#  'derived_execution_coefficients': 28444.0, 'derived_execution_masks': 1152.0,
#  'total': 753076.0}

core = td.make_core(dims=2, max_trunc=10, precompute_shuffle=True)
print(core.memory_bytes() == parts["total"])  # True
```

A standard total-degree core retains no plans without shuffle precomputation. Bidegree,
partially symmetrized and shear cores always retain layout and product plans, and the shuffle
scope adds to them:

```python
for scope in (False, "generator", True):
    kib = td.core_expected_memory(
        dims=(2, 2), max_trunc=(4, 4), partially_symmetrized=True,
        precompute_shuffle=scope, unit="KiB",
    )
    print(scope, round(kib, 1))
# False 44.7
# generator 307.3
# True 347.1
```

## Shear coordinates and partial symmetrization

Write the path as $Y = (A, X)$, with $A$ the first $d'$ channels and $X$ the last $d''$. A word
of bidegree $(n, m)$ factors uniquely as $w = w_0\, i_1\, w_1 \cdots i_n\, w_n$, with letters
$i_j \in \mathcal{A}'$ and words $w_j$ in the letters $\mathcal{A}''$. Shear coordinates pair
words with $\Psi(\mathbf{a})$ instead of $\mathbf{a}$, for an invertible linear map $\Psi$ that
preserves bidegrees; a shear core stores $\Psi(\mathbf{a})$. On signatures they are

$$
\langle w, \mathrm{Sig}(Y)_{s,t} \rangle_{\Psi}
=
\langle w_n, \mathrm{Sig}(X)_{s,t} \rangle
\int_{s < t_1 < \cdots < t_n < t}
\prod_{j=1}^{n} \langle w_{j-1}, \mathrm{Sig}(X)_{s,t_j} \rangle \, \mathrm{d}A^{i_j}_{t_j}.
$$

Partial symmetrization identifies words that differ only by the order of letters within a run of
consecutive $\mathcal{A}''$-letters, $[w] = [w_0]\, i_1\, [w_1] \cdots i_n\, [w_n]$, and sums their
coefficients:

$$
\langle [w], \widehat{\mathbf{a}} \rangle = \sum_{[v] = [w]} \langle v, \mathbf{a} \rangle .
$$

A partially symmetrized block of bidegree $(n, m)$ therefore has
$\binom{m + (n+1) d'' - 1}{m} d'^{\,n}$ coordinates instead of $\binom{n+m}{n} d'^{\,n} d''^{\,m}$.
Its transpose, $[w] \mapsto \sum_{[v] = [w]} v$, is
[`tensor_to_ordered`][tensordev.core.universal.Universal.tensor_to_ordered].

The shear coordinates are the $\Psi$-coordinates of Proposition 3.2 in
P. P. Hager and L. Pelizzari,
[*Expected signatures via partial integration, coordinate change and symmetrization*](https://arxiv.org/abs/2607.29534),
arXiv preprint, 2026; their Remark 3.3 calls $\Psi$ shear-like, hence the name. The partial
symmetrization is their quotient by the commutators of the letters in $\mathcal{A}''$ (Sections 2.4
and 3.2). The data layouts and algorithms of these cores are described in
P. P. Hager and L. Pelizzari,
[*A Technical Note on Signature Computations with Bidegree Truncation, Coordinate Changes, and Symmetrization*](https://github.com/hagerpa/tensordev/tree/main/academia/bidegree),
technical note, 2026.

With one channel in each group, the $(1, 1)$ block has two entries, one per position of the
$\mathcal{A}'$-letter: first the word $AX$, then $XA$. By the formula, the shear coordinate of
$AX$ is the product of the two increments, and the coordinate of $XA$ does not change:

```python
t = jnp.linspace(0.0, 1.0, 33)[:, None]
Y = jnp.concatenate([t, jnp.sin(3.0 * t)], axis=-1)  # A = t, X = sin(3t)

standard = td.make_core(dims=(1, 1), max_trunc=(1, 1))
shear = td.shear_core(standard)

sig_standard = td.path_signature(Y, core=standard)
sig_shear = td.path_signature(Y, core=shear)

A_increment, X_increment = Y[-1] - Y[0]
np.testing.assert_allclose(sig_shear[1, 1][0], A_increment * X_increment)
np.testing.assert_allclose(sig_shear[1, 1][1], sig_standard[1, 1][1])

converted = shear.tensor_from_standard_coordinates(sig_standard)
for grade in sig_shear.grades:
    np.testing.assert_allclose(converted[grade], sig_shear[grade], atol=1e-12)
```

[`tensor_from_standard_coordinates`][tensordev.core.universal.Universal.tensor_from_standard_coordinates]
and [`tensor_to_standard_coordinates`][tensordev.core.universal.Universal.tensor_to_standard_coordinates]
apply $\Psi$ and its inverse; they change coordinates, not the grading.

A shear core multiplies in these coordinates, with the products of the technical note. Its
`tensor_product` is

$$
\mathbf{a}\odot\mathbf{b} = \Psi\bigl(\Psi^{-1}(\mathbf{a})\otimes\Psi^{-1}(\mathbf{b})\bigr),
$$

so that $\Psi(\mathrm{Sig}(Y)_{s,t}) = \Psi(\mathrm{Sig}(Y)_{s,u})\odot\Psi(\mathrm{Sig}(Y)_{u,t})$ for
$s \le u \le t$. Its `tensor_shuffle_product` is the dual product $\bullet$, with
$\langle \alpha, \mathrm{Sig}(Y)_{s,t}\rangle_\Psi \langle \beta, \mathrm{Sig}(Y)_{s,t}\rangle_\Psi = \langle \alpha\bullet\beta, \mathrm{Sig}(Y)_{s,t}\rangle_\Psi$.
Neither is the concatenation or shuffle of the stored arrays: for letters $i \in \mathcal{A}'$ and
$j \in \mathcal{A}''$, $i \bullet j = ij$ while $i \sqcup\!\sqcup j = ij + ji$, and $j \odot i = ij + ji$ while
$j \otimes i = ji$. A partially symmetrized core multiplies in the quotient, with
$\widehat{\mathbf{a}}\,\widehat{\otimes}\,\widehat{\mathbf{b}} = \widehat{\mathbf{a}\otimes\mathbf{b}}$ and its dual product
$\widehat{\sqcup\!\sqcup}$, or with $\widehat{\odot}$ and $\widehat{\bullet}$ in shear coordinates.

With two channels in $\mathcal{A}''$, partial symmetrization shrinks the $(1, 2)$ block from 12
to 10 coordinates. [`tensor_partially_symmetrize`][tensordev.core.universal.Universal.tensor_partially_symmetrize]
maps an ordered tensor to the quotient. The last check confirms that `tensor_to_ordered` is its
transpose:

```python
Y3 = jnp.concatenate([t, jnp.sin(3.0 * t), t**2], axis=-1)  # A = t, X = (sin(3t), t^2)

ordered = td.make_core(dims=(1, 2), max_trunc=(2, 2))
quotient = td.symmetrize_core(ordered)

sig_ordered = td.path_signature(Y3, core=ordered)
sig_quotient = td.path_signature(Y3, core=quotient)
print(sig_ordered[1, 2].shape, sig_quotient[1, 2].shape)  # (12,) (10,)

symmetrized = quotient.tensor_partially_symmetrize(sig_ordered)
np.testing.assert_allclose(
    quotient.tensor_to_flat(symmetrized), quotient.tensor_to_flat(sig_quotient), atol=1e-12
)

width = quotient.tensor_to_flat(sig_quotient).shape[-1]
alpha = quotient.tensor_from_flat(jax.random.normal(jax.random.PRNGKey(2), (width,)))
np.testing.assert_allclose(
    quotient.tensor_inner_product(alpha, sig_quotient),
    ordered.tensor_inner_product(quotient.tensor_to_ordered(alpha), sig_ordered),
)
```

--8<-- "README.md:shear_pairing"

## Backends

--8<-- "README.md:backends"

Developments scan over time with a sequential core,
[`JaxSequentialCore`][tensordev.JaxSequentialCore]. `set_default_core` installs the matching
pair, and [`get_default_core_pair`][tensordev.get_default_core_pair] returns both cores.

## Migrating from 0.0.3

Version 0.1.0 replaces the separate shuffle core with optional shuffle plans on every bounded
core:

| 0.0.3 | 0.1.0 |
|---|---|
| `JaxShuffleCore(d, trunc)`, `shuffle_core(d, trunc)` | `make_core(dims=d, max_trunc=trunc, precompute_shuffle=True)` |
| the same, as background default | `set_default_core(dims=d, max_trunc=trunc, precompute_shuffle=True)` |
| `shuffle_core_expected_memory` | `core_expected_memory(dims=d, max_trunc=trunc, precompute_shuffle=True)` |

`core_expected_memory` reports MiB by default. `tensordev` 0.1.0 requires Python 3.11 or newer and `jax>=0.10.0,<0.12`. `JaxSequentialCore`
remains public, and developments resolve it automatically for cores returned by `make_core`.
The [changelog](https://github.com/hagerpa/tensordev/blob/main/CHANGELOG.md) lists all changes.
