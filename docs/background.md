# Background

`tensordev` implements published algorithms and learned from other open-source signature
libraries. This page lists both, describes the technical note in the repository, and states the
license.

## Acknowledgements

--8<-- "README.md:acknowledgements"

On this site, the Horner-style development is the streaming step of
[path signatures](signatures.md#arguments) and the operation of the
[native CPU kernels](performance.md#native-cpu-kernels); the
signature-kernel API, the finite-difference stencil and the higher-order schemes are on
[Signature kernels](kernels.md); the word-parallel approach is the alpha
[wordwise GPU execution](performance.md#wordwise-gpu-execution).

## References

--8<-- "README.md:references"

## Technical note

The technical note by P. P. Hager and L. Pelizzari,
[*A Technical Note on Signature Computations with Bidegree Truncation, Coordinate Changes, and
Symmetrization*](https://github.com/hagerpa/tensordev/blob/main/academia/bidegree/main.pdf)
(2026), documents the data layouts and algorithms behind the bidegree, shear and partially
symmetrized cores. Its LaTeX source is in
[`academia/bidegree`](https://github.com/hagerpa/tensordev/tree/main/academia/bidegree). The note
covers

- split alphabets $\mathcal{A} = \mathcal{A}' \,\dot\cup\, \mathcal{A}''$, the bidegree
  $(|w|_{\mathcal{A}'}, |w|_{\mathcal{A}''})$ of a word $w$, and bidegree truncation;
- the shear coordinate transformation and partial symmetrization, under which adjacent letters
  of $\mathcal{A}''$ commute, both from Hager and Pelizzari,
  [*Expected signatures via partial integration, coordinate change and
  symmetrization*](https://arxiv.org/abs/2607.29534);
- a contiguous array layout: a tensor of bidegree $(n, m)$ is stored as dense blocks
  $(\mathbb{R}^{d'})^{\otimes n} \otimes (\mathbb{R}^{d''})^{\otimes m}$, one for each placement
  of the $n$ letters from $\mathcal{A}'$ among the $n + m$ positions, with $d' = |\mathcal{A}'|$,
  $d'' = |\mathcal{A}''|$ and the placements ranked colexicographically along an extra axis.
  For partially symmetrized tensors, one colexicographically ranked axis of multiset placements
  replaces the placement axis and the $\mathcal{A}''$ axes, and only
  $(\mathbb{R}^{d'})^{\otimes n}$ stays dense. Concatenation and shuffle products then act on the
  rank axis and permute the dense axes;
- these operations in standard and shear coordinates, for ordered and partially symmetrized
  tensors;
- how the word-parallel signature computation of `pathsig` works with the ordered and partially
  symmetrized bidegree layouts.

The cores are described on [Tensor algebra](tensor-algebra.md), the word-parallel kernels under
[Wordwise GPU execution](performance.md#wordwise-gpu-execution).

## License

--8<-- "README.md:license"

The native CPU kernels and the wheels that contain them are under the same license.
