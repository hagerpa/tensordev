import numpy as np
import numba as nb

from typing import Tuple

from shuffle_basics import get_tensor_algebra_size, factorial_nb, get_digits_of_word, shuffle_nb


@nb.njit(nb.types.Tuple((nb.int32[:], nb.int32[:]))(nb.int32[:], nb.int32[:]), fastmath=True)
def remove_duplicate_columns(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Views input arrays a and b of same length n as a (2,n) array and removes duplicate columns.
    Primarly used for partition calculation as part of shuffle precomputation.

    Parameters
    ----------
    a : np.ndarray
        First row of length (n,).
    b : np.ndarray
        Second row of length (n,), i.e. same shape as `a`.

    Returns
    -------
    out : np.ndarray
        Two rows of same length, less or equal to n, such that there are no duplicate columns.

    Notes
    -----
    - This function is compiled with Numba (`@njit`).
    - Input arrays must be contiguous and have supported dtypes.

    Examples
    --------
    >>> a = np.array([1.0, 2.0, 1.0])
    >>> b = np.array([4.0, 5.0, 4.0])
    >>> remove_duplicate_columns(a, b)
    array([1.0, 2.0]), array([4.0, 5.0])
    """

    # pack the two rows into one np.uint64 array, each one using 32 bits
    packed = np.array([(np.uint64(a[j]) << 32) | np.uint64(b[j]) for j in range(len(a))])

    # numbers are equal in combined array iff the respective columns are equal
    unique_packed = np.unique(packed)

    # decompose the combined array back into two columns
    row0 = np.array([np.int32(unique_packed[j] >> 32) for j in range(len(unique_packed))])
    row1 = np.array([np.int32(unique_packed[j] & 0xFFFFFFFF) for j in range(len(unique_packed))])
    return row0, row1


@nb.njit(nb.types.Tuple((nb.int32[:], nb.int32[:]))(nb.int64, nb.int64, nb.int64, nb.int64), fastmath=True)
def partition(k: int, n: int, m: int, d: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each integer k (representing a length n+m word using d letters), we find all order-respecting subwords of 
    sizes n and m respectively. After removing duplicate word combinations we are left with all possibilities such 
    that the shuffle of the length n and m words yield at least one component equal to k. This is the core function 
    used to precompute shuffle products.

    Parameters
    ----------
    k : int
        Represents word of length `n` + `m`.
    n : np.ndarray
        Represents length of first word.
    m : np.ndarray
        Represents length of second word.
        Must be smaller than `n`. This is sufficient since the shuffle product is symmetric.
    d : int
        Basis for the tensor algebra.

    Returns
    -------
    out : Tuple[np.ndarray, np.ndarray]
        See description above.

    Notes
    -----
    - This function is compiled with Numba (`@njit`).
    - Only Numba-supported features are allowed.
    """

    # Since the shuffle product is commutative, we only want to compute it for n \geq m
    assert n >= m, f"Shuffle only needs to be computed for {n} >= {m}."

    # if m is not zero, then n cannot be zero either by the above
    if m == 0:
        return np.array([k], dtype=np.int32), np.array([0], dtype=np.int32) # only k itself and the empty word work
    
    # from now we can assume n >= m > 0
    else:
        # retrieve digits and word length of word k
        digits = get_digits_of_word(k, d)
        len_k = len(digits)

        # if n + m != len_k we would know for sure that no words i and j work 
        assert n + m == len_k, f"Respective word lengths {n} and {m} must add to length of k, namely {len_k}."

        # apriori there are only len_k nChr n possibilieties, allocate apropriate sizes
        max_size = factorial_nb(len_k) // (factorial_nb(n) * factorial_nb(len_k-n))
        row0 = np.empty(max_size, dtype=np.int32)
        row1 = np.empty(max_size, dtype=np.int32)

        # we need to perform a weighted sum by powers of d, since n is greater than m, the following is enough
        pow_d = np.array([d**(n-l-1) for l in range(n)])

        # We start Gosper's Hack (http://programmingforinsomniacs.blogspot.com/2018/03/gospers-hack-explained.html)
        set = (1 << n) - 1 
        limit = (1 << len_k)
        idx = 0
        while set < limit:
            # convert current valid binary number to boolean array
            binary_arr = np.array([(set >> (len_k - j - 1)) & 1 for j in range(len_k)], dtype=np.bool)

            # update rows; ~ converts to complementary binary number
            row0[idx] = sum(digits[binary_arr] * pow_d)
            row1[idx] = sum(digits[~binary_arr] * pow_d[m:])
            idx += 1

            # update current binary number according to Gosper's Hack
            c = set & - set
            r = set + c
            set = (((r ^ set) >> 2) // c) | r

        # return unique columns (when stacked)
        return remove_duplicate_columns(np.int32(get_tensor_algebra_size(d, n-1)) + row0, np.int32(get_tensor_algebra_size(d, m-1)) + row1)
    

@nb.njit(
     nb.types.Tuple((nb.uint32[:], nb.uint32[:], nb.uint32[:], nb.float64[:]))(nb.int64, nb.int64, nb.int64),
     fastmath=True   
)
def assemble_shuffle_tuple(n: int, m: int, d: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Precomputes the shuffle product from (R^d)^{otimes n} x (R^d)^{otimes m} -> (R^d)^{otimes (n+m)} for n >= m.
    We store the result into four arrays corresponding to a 3d generalization of a csr format. The first array
    stores the sizes of the matrix slices, the second and third array represent the row and column indices of each
    slice and the last array stores the respective data.

    Parameters
    ----------
    n : np.ndarray
        Represents length of first word.
    m : np.ndarray
        Represents length of second word.
        Must be smaller than `n`. This is sufficient since the shuffle product is symmetric.
    d : int
        Basis for the tensor algebra.

    Returns
    -------
    out : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Returns 3d generalization of csr format.

    Notes
    -----
    - This function is compiled with Numba (`@njit`).
    - Only Numba-supported features are allowed.
    """
    

    # Since the shuffle product is commutative, we only want to compute it for n \geq m
    assert n >= m, f"Shuffle only needs to be computed for {n} >= {m}."

    # in tensor algebra there are tensor_shift many tensors before length m+n tensors
    tensor_shift = get_tensor_algebra_size(d, n+m-1)

    # there are d**(n+m) target tensors and maximally n+m nChr n words per level with non-zero contribution
    N = int(d**(n+m))
    max_size = N * factorial_nb(n+m) // (factorial_nb(n) * factorial_nb(m))

    # if m is not zero, then n cannot be zero either by the above
    if m == 0:
        return np.ones(N, dtype=np.uint32), np.arange(N, dtype=np.uint32), np.zeros(N, dtype=np.uint32), np.ones(N, dtype=np.float64),
    
    # from now we can assume n >= m > 0
    else:
        # preallocate memory
        all_size = np.empty(N, dtype=np.uint32)
        all_rows = np.empty(max_size, dtype=np.uint32)
        all_cols = np.empty(max_size, dtype=np.uint32)
        all_data = np.empty(max_size, dtype=np.float64)

        idx = 0
        idy = 0
        for k in range(tensor_shift, tensor_shift+N):
            p1, p2 = partition(k, n, m, d)
            for j in range(len(p1)):
                all_rows[idx] = p1[j]
                all_cols[idx] = p2[j]
                all_data[idx] = shuffle_nb(p1[j], p2[j], k, d)
                idx += 1
            all_size[idy] = len(p1)
            idy += 1
        sn = np.uint32(get_tensor_algebra_size(d, n-1))
        sm = np.uint32(get_tensor_algebra_size(d, m-1))
        return all_size, all_rows[:idx] - sn, all_cols[:idx] -  sm, all_data[:idx]
    
