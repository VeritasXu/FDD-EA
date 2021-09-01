import math
import numpy as np


def index_bootstrap(num_data, prob):
    """
    :param num_data: the index matrix of input, int
    :param prob: the probability for one index sample to be chose, >0
    return: index of chose samples, bool

    example:
    a=np.array([[1,2,3,4],[0,0,0,0]]).T
    rand_p = np.random.rand(4)
    b=np.greater(rand_p,0.5)
    b is the output, and we can use a[b] to locate data
    """
    rand_p = np.random.rand(num_data)

    out = np.greater(rand_p, 1 - prob)

    if True not in out:
        out = index_bootstrap(num_data, prob)

    return out


def mini_batches(input_x, input_y, distance, batch_size=64, seed=0):
    """
    return random batch indexes for a list
    """
    np.random.seed(seed)
    num_samp = input_x.shape[0]
    batches = []
    permutation = list(np.random.permutation(num_samp))
    num_batch = math.floor(num_samp / batch_size)
    iter_list = [i for i in range(num_batch)]

    for k in iter_list:
        batch_index = permutation[k * batch_size:(k + 1) * batch_size]
        batches.append((input_x[batch_index], input_y[batch_index], distance[batch_index]))
    if num_samp % batch_size != 0:
        batch_index = permutation[batch_size * num_batch:]
        batches.append((input_x[batch_index], input_y[batch_index], distance[batch_index]))

    return batches


def sort_data(pop, pop_obj, num_select):
    """
    sort the data (x, y) according to the descending sequence and pick first num_select points
    for single objective problem
    :param pop: [N, d]
    :param pop_obj: [N, 1]
    :param num_select:
    :return:
    """
    data_index = np.argsort(pop_obj.flatten())
    return pop[data_index[0:num_select], :], pop_obj[data_index[0:num_select], :]

