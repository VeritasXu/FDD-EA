import numpy as np
from pyDOE import lhs


def lhsamp(n, d):
    """
    :param n:
    :param d:
    :return:
    """
    S = np.zeros((n, d))
    for i in range(d):
        S[:, i] = (np.random.uniform(0, 1, n) + np.random.permutation(n)) / n

    return S


def initialize_pop(n, d, lb, ub):
    """
    :param n: number of samples
    :param d: number of the decision variable
    :param lb: lower bound
    :param ub: upper bound
    :return:sample
    """

    result = lhs(d, samples=n)
    # result = lhsamp(n, d)

    POP = result * (ub - lb) + lb

    return POP
