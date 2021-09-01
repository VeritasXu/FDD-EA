import copy
import numpy as np
from functools import partial
from utils.initialize import initialize_pop


def _obj_wrapper(func, args, kwargs, x):
    return func(x, *args, **kwargs)


def _power(mat1, mat2):
    # To solve the problem: Numpy does not seem to allow fractional powers of negative numbers
    return np.sign(mat1) * np.power(np.abs(mat1), mat2)


def RCGA(func, lb, ub, args=(), kwargs=None, pop_size=100, max_iter=100, particle_output=False):
    if kwargs is None:
        kwargs = {}

    lb = np.array(lb)
    ub = np.array(ub)

    d = len(lb)
    k_tour = 2
    obj = partial(_obj_wrapper, func, args, kwargs)

    pop_rand = initialize_pop(pop_size, d, lb[0], ub[0])

    pop_fitness = obj(pop_rand)

    generation = 0

    while generation < max_iter:
        temp1 = np.random.randint(0, pop_size, pop_size)
        temp2 = np.random.randint(0, pop_size, pop_size)

        pop_parent = np.zeros((pop_size, d))
        for i in range(pop_size):
            if pop_fitness[temp1[i]] <= pop_fitness[temp2[i]]:
                pop_parent[i] = pop_rand[temp1[i]]
            else:
                pop_parent[i] = pop_rand[temp2[i]]

        # crossover(simulated binary crossover)
        # dic_c is the distribution index of crossover
        dis_c = 1
        mu = np.random.rand(int(pop_size / 2), d)
        idx1 = [i for i in range(0, pop_size, 2)]
        idx2 = [i + 1 for i in range(0, pop_size, 2)]
        parent1 = pop_parent[idx1, :]
        parent2 = pop_parent[idx2, :]
        element_min = np.minimum(parent1, parent2)
        element_max = np.maximum(parent1, parent2)
        tmp_min = np.minimum(element_min - lb, ub - element_max)
        beta = 1 + 2 * tmp_min / np.maximum(abs(parent2 - parent1), 1e-6)
        alpha = 2 - beta ** (-dis_c - 1)
        betaq = _power(alpha * mu, 1 / (dis_c + 1)) * (mu <= 1 / alpha) + \
                _power(1. / (2 - alpha * mu), 1 / (dis_c + 1)) * (mu > 1. / alpha)
        # the mutation is performed randomly on each variable
        betaq = betaq * _power(-1, np.random.randint(0, 2, (int(pop_size / 2), d)))
        betaq[np.random.rand(int(pop_size / 2), d) > 0.5] = 1
        offspring1 = 0.5 * ((1 + betaq) * parent1 + (1 - betaq) * parent2)
        offspring2 = 0.5 * ((1 - betaq) * parent1 + (1 + betaq) * parent2)
        pop_crossover = np.vstack((offspring1, offspring2))

        # mutation (ploynomial mutation)
        # dis_m is the distribution index of polynomial mutation
        dis_m = 1
        pro_m = 1 / d
        rand_var = np.random.rand(pop_size, d)
        mu = np.random.rand(pop_size, d)
        deta = np.minimum(pop_crossover - lb, ub - pop_crossover) / (ub - lb)
        detaq = np.zeros((pop_size, d))
        # use dot multiply to replace matrix & in matlab
        position1 = (rand_var <= pro_m) * (mu <= 0.5)
        position2 = (rand_var <= pro_m) * (mu > 0.5)
        tmp1 = 2 * mu[position1] + (1 - 2 * mu[position1]) * _power(1 - deta[position1], (dis_m + 1))
        detaq[position1] = _power(tmp1, 1 / (dis_m + 1)) - 1
        tmp2 = 2 * (1 - mu[position2]) + 2 * (mu[position2] - 0.5) * _power(1 - deta[position2], (dis_m + 1))
        detaq[position2] = 1 - _power(tmp2, 1 / (dis_m + 1))
        pop_mutation = pop_crossover + detaq * (ub - lb)

        # fitness calculation
        pop_mutation_fitness = obj(pop_mutation)
        # ------------------------------------ environment selection
        pop_rand_iter = np.vstack((pop_rand, pop_mutation))
        pop_fitness_iter = np.concatenate((pop_fitness, pop_mutation_fitness))
        sorted_fit_index = np.argsort(pop_fitness_iter)
        pop_rand = pop_rand_iter[sorted_fit_index[0:pop_size], :]
        pop_fitness = pop_fitness_iter[sorted_fit_index[0:pop_size]]
        generation += 1
        # print(pop_fitness)

    p_min = pop_rand[0, :].flatten()
    temp_best = pop_fitness[0].flatten()
    # temp_best = obj(p_min)
    if particle_output:
        return p_min, temp_best, pop_rand, pop_fitness
    else:
        return p_min, temp_best

# def myfunc(x, *arg):
#     c1 = arg[0]
#     c2 = arg[1]
#     n = x.shape[0]
#     fitness = np.zeros(n)
#     for i in range(n):
#         x1 = x[i, 0]
#         x2 = x[i, 1]
#         fitness[i] = x1**c1 - 2*x2*x1**2 + x2**2 + x1**2 - 2*x1 + c2
#     return fitness
# #
# lb = [-3, -1]
# ub = [2, 6]
# print(RCGA(myfunc, lb, ub, args=(4, 6)))
