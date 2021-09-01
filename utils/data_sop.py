# coding=utf-8
import numpy as np
from utils.initialize import initialize_pop


def init_samples(func, num_b4_d, d, xlb, xub):
    """
    :param func: real function
    :param num_b4_d: number before d, e.g. 5d, 11d
    :param d: dimension of the decision variable
    :param xlb: lower bound of x
    :param xub: up bound of x
    :return:
    """
    number = int(num_b4_d * d)

    X_init = initialize_pop(number, d, xlb, xub)

    Y_init = np.zeros(number)
    for i in range(number):
        Y_init[i] = func(X_init[i, :])

    return np.array(X_init), np.array(Y_init).reshape(-1, 1)
