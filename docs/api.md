# API reference

Every object on this page is rendered from its docstring. Names exported at the top level are
used as `tensordev.NAME` (`import tensordev as td`); objects that only a subpackage exports are
shown with their import path, for example `tensordev.sss.JordanLambda`.

## Package structure

The top-level `tensordev` namespace re-exports the main objects of `core`, `development`,
`volterra`, `sss` and `kernel`; `tensordev.util` is imported on its own. [Home](index.md) has
the installation and a first example.

--8<-- "README.md:package_structure"

## Cores

A core implements the truncated tensor algebra. [`make_core`][tensordev.make_core] builds one,
and [`set_default_core`][tensordev.set_default_core] installs one as the process default, which
the module-level operations use and functions that accept `core=` use when it is omitted.
[Tensor algebra](tensor-algebra.md) explains the configurations with examples.

| Class | Selected by `make_core` and `set_default_core` with |
|---|---|
| [`Jax`][tensordev.Jax] | integer `dims` and integer `max_trunc`; an unbounded `Jax()` is the initial default |
| [`JaxBigraded`][tensordev.JaxBigraded] | pair `dims` and pair `max_trunc` |
| [`JaxPartiallySymmetrizedBigraded`][tensordev.JaxPartiallySymmetrizedBigraded] | pair `dims`, pair `max_trunc`, `partially_symmetrized=True` |
| [`JaxShearTotal`][tensordev.core.JaxShearTotal] | pair `dims`, integer `max_trunc`, `coordinates="shear"` |
| [`JaxShearBigraded`][tensordev.core.JaxShearBigraded] | pair `dims`, pair `max_trunc`, `coordinates="shear"` |
| [`JaxPartiallySymmetrizedShearBigraded`][tensordev.JaxPartiallySymmetrizedShearBigraded] | pair `dims`, pair `max_trunc`, `partially_symmetrized=True`, `coordinates="shear"` |
| [`JaxSequentialCore`][tensordev.JaxSequentialCore] | paired with every built-in core as its sequential core |

Most code selects a class through `make_core` or `set_default_core` rather than calling its
constructor. The class entries show only the constructor; the operations every core provides
are under [Tensor operations](#tensor-operations).

::: tensordev.make_core

::: tensordev.set_default_core

::: tensordev.reset_default_core

::: tensordev.get_default_core

Return the active algebra core, the first entry of `get_default_core_pair()`.

::: tensordev.get_default_seq_core

Return the active sequential core, the second entry of `get_default_core_pair()`.

::: tensordev.get_default_core_pair

::: tensordev.core_expected_memory

::: tensordev.shear_core

::: tensordev.symmetrize_core

::: tensordev.Jax
    options:
      members: false
      show_bases: false
      show_signature_annotations: false

Dense total-degree JAX core. `Jax()` without arguments is dimension-free and accepts any
total-degree truncation; `make_core(dims=d, max_trunc=N)` returns a `Jax` with `d=d` and
`max_trunc=N`.

::: tensordev.JaxSequentialCore
    options:
      members: false
      show_bases: false

::: tensordev.JaxBigraded
    options:
      members: false
      show_bases: false
      show_signature_annotations: false

::: tensordev.JaxPartiallySymmetrizedBigraded
    options:
      members: false
      show_bases: false
      show_signature_annotations: false

::: tensordev.JaxPartiallySymmetrizedShearBigraded
    options:
      members: false
      show_bases: false
      show_signature_annotations: false

::: tensordev.core.JaxShearTotal
    options:
      members: false
      show_bases: false
      show_root_full_path: true
      show_signature_annotations: false

::: tensordev.core.JaxShearBigraded
    options:
      members: false
      show_bases: false
      show_root_full_path: true
      show_signature_annotations: false

::: tensordev.BigradedSpec

::: tensordev.BigradedTensor

## Tensor operations

The module-level functions `td.tensor_*` are bound methods of the installed default core:
`tensordev` binds each name to the methods of the core it installs, and
[`set_default_core`][tensordev.set_default_core] and
[`reset_default_core`][tensordev.reset_default_core] rebind them. The initial default is an
unbounded [`Jax`][tensordev.Jax] core, so `td.tensor_product(A, B)` calls
`td.get_default_core().tensor_product(A, B)`. Every core object has the same 36 methods; see
[Tensor algebra](tensor-algebra.md) for examples.

The entries below are the generic implementations on `Universal`, the base class of every core,
and describe the total-degree layout: a tuple of levels `(X_0, X_1, ..., X_N)` in which level
`k` has shape `batch + (d**k,)`, or `(X_1, ..., X_N)` in the first-on format that flags such as
`a_first_on` and `first_on_out` select. Bidegree cores take and return a
[`BigradedTensor`][tensordev.BigradedTensor] and override some operations, for example
`tensor_to_flat`, `tensor_from_flat` and `tensor_densify`; shear cores override
`tensor_product`, `tensor_logarithm` and `tensor_fmexp`. Only partially symmetrized cores
implement `tensor_partially_symmetrize`, `tensor_partially_symmetrize_homogeneous` and
`tensor_to_ordered`; other cores raise `RuntimeError`.

| Group | Operation |
|---|---|
| Sum, scaling | [`tensor_summation`][tensordev.core.universal.Universal.tensor_summation] |
| | [`tensor_scalar_multiply`][tensordev.core.universal.Universal.tensor_scalar_multiply] |
| | [`tensor_dilation`][tensordev.core.universal.Universal.tensor_dilation] |
| Tensor product | [`tensor_product`][tensordev.core.universal.Universal.tensor_product] |
| | [`tensor_product_homogeneous`][tensordev.core.universal.Universal.tensor_product_homogeneous] |
| Shuffle product | [`tensor_shuffle_vector`][tensordev.core.universal.Universal.tensor_shuffle_vector] |
| | [`tensor_shuffle_vector_homogeneous`][tensordev.core.universal.Universal.tensor_shuffle_vector_homogeneous] |
| | [`tensor_shuffle_product`][tensordev.core.universal.Universal.tensor_shuffle_product] |
| | [`tensor_shuffle_product_homogeneous`][tensordev.core.universal.Universal.tensor_shuffle_product_homogeneous] |
| Exp, log | [`tensor_exponential`][tensordev.core.universal.Universal.tensor_exponential] |
| | [`tensor_logarithm`][tensordev.core.universal.Universal.tensor_logarithm] |
| | [`tensor_fmexp`][tensordev.core.universal.Universal.tensor_fmexp] |
| Pairing, adjoint | [`tensor_inner_product`][tensordev.core.universal.Universal.tensor_inner_product] |
| | [`tensor_inner_product_homogeneous`][tensordev.core.universal.Universal.tensor_inner_product_homogeneous] |
| | [`tensor_shear_pairing`][tensordev.core.universal.Universal.tensor_shear_pairing] |
| | [`tensor_shear_pairing_homogeneous`][tensordev.core.universal.Universal.tensor_shear_pairing_homogeneous] |
| | [`tensor_adjoint_product`][tensordev.core.universal.Universal.tensor_adjoint_product] |
| | [`tensor_adjoint_left_homogeneous`][tensordev.core.universal.Universal.tensor_adjoint_left_homogeneous] |
| | [`tensor_adjoint_right_homogeneous`][tensordev.core.universal.Universal.tensor_adjoint_right_homogeneous] |
| Matrix-valued | [`tensor_matrix_product`][tensordev.core.universal.Universal.tensor_matrix_product] |
| | [`tensor_matrix_product_homogeneous`][tensordev.core.universal.Universal.tensor_matrix_product_homogeneous] |
| | [`tensor_matrix_product_left`][tensordev.core.universal.Universal.tensor_matrix_product_left] |
| | [`tensor_matrix_product_left_homogeneous`][tensordev.core.universal.Universal.tensor_matrix_product_left_homogeneous] |
| | [`tensor_matrix_product_right`][tensordev.core.universal.Universal.tensor_matrix_product_right] |
| | [`tensor_matrix_product_right_homogeneous`][tensordev.core.universal.Universal.tensor_matrix_product_right_homogeneous] |
| Layout | [`tensor_stack`][tensordev.core.universal.Universal.tensor_stack] |
| | [`tensor_moveaxis`][tensordev.core.universal.Universal.tensor_moveaxis] |
| | [`tensor_densify`][tensordev.core.universal.Universal.tensor_densify] |
| | [`tensor_from_flat`][tensordev.core.universal.Universal.tensor_from_flat] |
| | [`tensor_to_flat`][tensordev.core.universal.Universal.tensor_to_flat] |
| | [`tensor_slice`][tensordev.core.universal.Universal.tensor_slice] |
| | [`tensor_from_standard_coordinates`][tensordev.core.universal.Universal.tensor_from_standard_coordinates] |
| | [`tensor_to_standard_coordinates`][tensordev.core.universal.Universal.tensor_to_standard_coordinates] |
| | [`tensor_partially_symmetrize`][tensordev.core.universal.Universal.tensor_partially_symmetrize] |
| | [`tensor_partially_symmetrize_homogeneous`][tensordev.core.universal.Universal.tensor_partially_symmetrize_homogeneous] |
| | [`tensor_to_ordered`][tensordev.core.universal.Universal.tensor_to_ordered] |

`td.tensor_flatten` is also bound, as an alias of `tensor_to_flat`.

!!! note
    `from tensordev import tensor_product` binds the method of the core that is the default when
    the import runs. A later `set_default_core` does not change that name; call
    `td.tensor_product` instead.

::: tensordev.core.universal.Universal
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false
      members:
        - tensor_summation
        - tensor_scalar_multiply
        - tensor_dilation
        - tensor_product
        - tensor_product_homogeneous
        - tensor_shuffle_vector
        - tensor_shuffle_vector_homogeneous
        - tensor_shuffle_product
        - tensor_shuffle_product_homogeneous
        - tensor_exponential
        - tensor_logarithm
        - tensor_fmexp
        - tensor_inner_product
        - tensor_inner_product_homogeneous
        - tensor_shear_pairing
        - tensor_shear_pairing_homogeneous
        - tensor_adjoint_product
        - tensor_adjoint_left_homogeneous
        - tensor_adjoint_right_homogeneous
        - tensor_matrix_product
        - tensor_matrix_product_homogeneous
        - tensor_matrix_product_left
        - tensor_matrix_product_left_homogeneous
        - tensor_matrix_product_right
        - tensor_matrix_product_right_homogeneous

::: tensordev.core.universal.Universal
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false
      members:
        - tensor_stack

Stack a list of equally structured elements level-wise along `axis`.

::: tensordev.core.universal.Universal
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false
      members:
        - tensor_moveaxis

Move the axis `source` to `destination` in every level.

::: tensordev.core.universal.Universal
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false
      members:
        - tensor_densify
        - tensor_from_flat
        - tensor_to_flat
        - tensor_slice
        - tensor_from_standard_coordinates
        - tensor_to_standard_coordinates
        - tensor_partially_symmetrize
        - tensor_partially_symmetrize_homogeneous
        - tensor_to_ordered

## Signatures and developments

[`path_signature`][tensordev.path_signature] computes the truncated signature of a path and
[`free_development`][tensordev.free_development] the free development of a tensor-valued path.
[`Signature`][tensordev.Signature] and [`FreeDevelopment`][tensordev.FreeDevelopment] bind
`trunc`, `core` and `seq_core` for repeated calls. Examples are on [Signatures](signatures.md).

::: tensordev.path_signature

::: tensordev.Signature

::: tensordev.free_development

::: tensordev.FreeDevelopment

## Volterra signatures

[`vsig`][tensordev.vsig] computes the truncated Volterra signature of a path for a
[`ConvolutionKernel`][tensordev.ConvolutionKernel], and
[`VolterraSignature`][tensordev.VolterraSignature] binds a kernel and a truncation. The
iteration functions implement the schemes `"quadratic"`, `"fft"` and `"adams"` of `vsig` and
take increments on the final grid. Examples are on [Volterra signatures](volterra.md).

::: tensordev.vsig

The parameters of `vsig` are described in
[Computing a Volterra signature](volterra.md#computing-a-volterra-signature). The method
[`VolterraSignature.vsig`][tensordev.VolterraSignature.vsig] takes the same arguments except
`kernel`, `trunc`, `core` and `seq_core`, which the object binds, and its entry lists them.

::: tensordev.volterra.vsig_fft
    options:
      show_root_full_path: true

Alias for `vsig` with `scheme="fft"` preset.

::: tensordev.VolterraSignature

::: tensordev.ConvolutionKernel

::: tensordev.FractionalKernel

::: tensordev.GammaKernel

::: tensordev.FSSKConvolutionKernel

::: tensordev.volterra.VolterraCoefficients
    options:
      show_root_full_path: true

::: tensordev.volterra.quadratic_iteration
    options:
      show_root_full_path: true

::: tensordev.volterra.fft_iteration
    options:
      show_root_full_path: true

::: tensordev.volterra.pc_iteration
    options:
      show_root_full_path: true

## State-space signatures

[`StateSpaceSignature`][tensordev.StateSpaceSignature] binds a finite state-space kernel
([`FSSK`][tensordev.FSSK]), a truncation and a hidden state, and computes Volterra signatures by
a state recursion; [`fssk_vsig`][tensordev.fssk_vsig] and [`fssk_state`][tensordev.fssk_state]
are the functional forms. [`Lambda`][tensordev.sss.Lambda] is the state operator, with the dense
and real-Jordan implementations [`DenseLambda`][tensordev.sss.DenseLambda] and
[`JordanLambda`][tensordev.sss.JordanLambda]. Examples are on
[State-space signatures](state-space.md).

::: tensordev.StateSpaceSignature

::: tensordev.FSSK

::: tensordev.fssk_vsig

::: tensordev.fssk_state

::: tensordev.sss.Lambda
    options:
      show_root_full_path: true

::: tensordev.sss.DenseLambda
    options:
      show_root_full_path: true

::: tensordev.sss.JordanLambda
    options:
      show_root_full_path: true

::: tensordev.sss.FSSKCoefficients
    options:
      show_root_full_path: true

## Signature kernels

Every signature kernel class derives from [`BaseKernel`][tensordev.kernel.BaseKernel]: a kernel
object is callable as `kernel(X, Y, ...)` and inherits the empirical methods `compute_kernel`,
`compute_Gram`, `compute_mmd`, `compute_scoring_rule` and `compute_expected_scoring_rule`. The
lower-case functions are the functional forms, and the static kernels are values for the
`static_kernel` argument. Examples are on [Signature kernels](kernels.md).

::: tensordev.kernel.BaseKernel
    options:
      show_root_full_path: true
      show_if_no_docstring: true
      members:
        - __call__
        - compute_kernel
        - compute_Gram
        - compute_mmd
        - compute_scoring_rule
        - compute_expected_scoring_rule

::: tensordev.SigKernel

::: tensordev.FreeKernel

::: tensordev.free_kernel

::: tensordev.HigherOrderKernel

::: tensordev.higher_order_kernel

::: tensordev.FSSKSigKernel

::: tensordev.fssk_sigkernel

::: tensordev.kernel.StaticKernel
    options:
      show_root_full_path: true

::: tensordev.LinearKernel

::: tensordev.RBFKernel

::: tensordev.kernel.RBF_CEXP_Kernel
    options:
      show_root_full_path: true

::: tensordev.kernel.RBF_SQR_Kernel
    options:
      show_root_full_path: true

## Utilities

Path generators and path preprocessing helpers, imported from `tensordev.util`.
[`bucket_pad_ragged_paths`][tensordev.util.bucket_pad_ragged_paths] and
[`velocity_to_increments`][tensordev.util.velocity_to_increments] are also exported by
`tensordev.kernel`. See [Example paths](signatures.md#example-paths) and
[Batching](kernels.md#batching).

::: tensordev.util
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_object_full_path: true
