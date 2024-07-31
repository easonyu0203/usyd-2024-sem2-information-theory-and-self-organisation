import numpy as np
from typing import Union

def infocontent(p: Union[float, np.ndarray]) -> np.ndarray:
    """
    Calculate the Shannon information content for a given probability or array of probabilities.

    The Shannon information content (also known as self-information) is defined as:
    I(p) = -log2(p)

    Parameters:
    p (Union[float, np.ndarray]): A probability or an array of probabilities. Each probability must be in the range [0, 1].

    Returns:
    np.ndarray: The Shannon information content for the given probability or array of probabilities.

    Raises:
    AssertionError: If any element in p is not in the range [0, 1].
    """
    p = np.asarray(p)
    assert np.all((0 <= p) & (p <= 1)), "All elements in p must be in the range [0, 1]"
    p = np.where(p == 0, np.nextafter(0, 1), p)
    return -np.log2(p)


def entropy(p: Union[np.ndarray, list]) -> np.ndarray:  
    """
    Calculate the Shannon entropy for a given probability distribution.

    The Shannon entropy is defined as:
    H(p) = -sum(p * log2(p))

    Parameters:
    p (Union[np.ndarray, list]): A probability distribution or a list of probability distributions. 
                                 Each distribution must sum to 1, and all probabilities must be in the range [0, 1].

    Returns:
    np.ndarray: The Shannon entropy for the given probability distribution.

    Raises:
    AssertionError: If the sum of any distribution is not equal to 1 or if any element in p is not in the range [0, 1].
    """
    # First make sure the array is now a numpy array
    if not isinstance(p, np.ndarray):
        p = np.array(p)

    # check if the sum of the array is equal to 1 and if all elements are in the range [0, 1]
    assert np.isclose(np.sum(p, axis=-1), 1).all(), "The sum of the array must be equal to 1"
    assert np.all((0 <= p) & (p <= 1)), "All elements in p must be in the range [0, 1]"

    # We need to take the expectation value over the Shannon info content at
    # p(x) for each outcome x:
    # Alter the equation below to provide the correct entropy:
    return np.sum( p * infocontent(p), axis=-1)