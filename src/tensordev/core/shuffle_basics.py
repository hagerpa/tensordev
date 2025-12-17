import numpy as np
import numba as nb

from typing import Tuple


@nb.njit(nb.int64(nb.int64), fastmath=True)
def factorial_nb(n: int) -> int:
    """
    Basic factorial function usable by numba.
    """
    if n == 0:
        return 1
    else:
        return n * factorial_nb(n-1)
    

@nb.njit(nb.int64(nb.int64, nb.int64), fastmath=True, cache=True)
def get_tensor_algebra_size(d: int, n: int) -> int:
    """
    Returns dim(T^n(R^d)).
    """
    if d != 1:
        return (d**(n+1) - 1) // (d - 1)
    else: 
        return n+1
    

@nb.njit(nb.boolean(nb.int64, nb.int64, nb.int64), fastmath=True, cache=True)
def check_if_last_letter_agrees(i: int, j: int, d: int) -> bool:
    """
    For given words I (=i) and J (=j), return bool according to I[-1] == J[-1].
    """
    assert i != 0 and j != 0, f"Empty words have no last letters."
    return (i-1) % d == (j-1) % d


@nb.njit(nb.int64(nb.int64, nb.int64), fastmath=True, cache=True)
def remove_letter_from_word(i: int, d: int) -> int:
    """
    For given word I (=i), return I' as long as I is non-empty.
    """
    assert i != 0, f"Cannot remove a letter from the empty word."
    return (i-1) // d # write j = 1 + d + ... + d^{len(I)-1} + sum_{i=0}^{len(I)-1} i_n d^n
    

@nb.njit(nb.int64(nb.int64, nb.int64), fastmath=True, cache=True)
def get_length_of_word(k: int, d: int) -> int:
    """
    Determines the length of the word representation of `k` in base `d` within the tensor algebra.
    """

    # If k is zero it represents the empty word and we return zero length
    if k == 0:
        return 0
    
    # If k is less than or equal to d, it fits in one digit
    if k <= d:
        return 1
    
    # Initialize counter to keep track of word length
    counter = 1
    while True: 
        # Increment counter until k fits within the tensor algebra size
        if k <= get_tensor_algebra_size(d, counter)-1:
            return counter
        counter += 1 # Increase word length
    

@nb.njit(nb.int64[:](nb.int64, nb.int64), fastmath=True, cache=True)
def get_digits_of_word(k: int, d: int) -> np.ndarray:
    """
    Converts an integer `k` into its digit representation in base `d` in tensor algebra.

    Parameters
    ----------
    k : int
        Integer repesentation of input word.
    d : int
        Basis of tensor algebra.

    Returns
    -------
    out : np.ndarray
        Digits of word `k` in base `d`.

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

    Example:
        >>> get_digits_of_word(42, 3)
        array([0, 0, 0, 2], dtype=unt64)

    Explanation:
        - The function first determines how many digits `k` has in the tensor algebra with base `d` using `get_length_of_word(k, d)`.
        - It then computes the corresponding digits by iteratively extracting the highest power of `d`.
    """

    # Compute the number of digits required to represent `k` in base `d`
    length_k = get_length_of_word(k, d)

    # Find the size of the tensor algebra below the given word length
    tensor_size_below = get_tensor_algebra_size(d, length_k-1)

    # Compute the remainder of `k` after subtracting the lower bound
    remainder_k = k - tensor_size_below

    # Initialize an array to store the digits
    digits = np.zeros(length_k, np.int64)

    # Extract the digits of `k` in base `d` iteratively
    for i in range(length_k):
        if remainder_k >= d**(length_k-i-1): 
            digits[i] = remainder_k // d**(length_k-i-1) # Get the current digit with integer division 
            remainder_k -= digits[i]*d**(length_k-i-1) # Subract used portion from remainder
    return digits


@nb.njit(nb.int64(nb.int64, nb.int64, nb.int64, nb.int64), fastmath=True)
def shuffle_nb(i: int, j: int, k: int, d: int) -> int:
    """
    For given words I (=i) and J (=j) and K (=k), return <I shuffle J, K>.

    REMARK: This could be done symbolically if needed. Using numba we do not have a cache in the recursion. Note that we also did not cache the compiled function itself (Issue with numba).
    """
    # base cases
    if k == 0:
        return 1
    elif i == 0: # if J = K, return 1, else 0
        if j == k:
            return 1
        else:
            return 0
    elif j == 0: # if I = K, return 1, else 0
        if i == k:
            return 1
        else:
            return 0
    # actual recursion
    else:
        val = 0
        if check_if_last_letter_agrees(i, k, d):
            val = val + shuffle_nb(remove_letter_from_word(i, d), j, remove_letter_from_word(k, d), d)
        if check_if_last_letter_agrees(j, k, d):
            val = val + shuffle_nb(i, remove_letter_from_word(j, d), remove_letter_from_word(k, d), d)
        return val