#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:01, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from SOP.cec2005.root import Root
from numpy import sum, dot, sqrt, abs, array, cos, pi, exp, e, ones, max
from numpy.random import normal


class Model(Root):
    def __init__(self, f_name="Rotated Hybrid Composition Function 1 with Noise in Fitness", f_shift_data_file="data_hybrid_func1",
                 f_ext='.txt', f_bias=120, f_matrix=None):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)
        self.f_matrix = f_matrix

    def __f12__(self, solution=None):
        return sum(solution ** 2 - 10 * cos(2 * pi * solution) + 10)

    def __f34__(self, solution=None, a=0.5, b=3, k_max=20):
        result = 0.0
        for i in range(len(solution)):
            result += sum([a ** k * cos(2 * pi * b ** k * (solution + 0.5)) for k in range(0, k_max)])
        return result - len(solution) * sum([a ** k * cos(2 * pi * b ** k * 0.5) for k in range(0, k_max)])

    def __f56__(self, solution=None):
        result = sum(solution ** 2) / 4000
        temp = 1.0
        for i in range(len(solution)):
            temp *= cos(solution[i] / sqrt(i + 1))
        return result - temp + 1

    def __f78__(self, solution=None):
        return -20 * exp(-0.2 * sqrt(sum(solution ** 2) / len(solution))) - exp(sum(cos(2 * pi * solution)) / len(solution)) + 20 + e

    def __f910__(self, solution=None):
        return sum(solution ** 2)

    def __fi__(self, solution=None, idx=None):
        if idx == 0 or idx == 1:
            return self.__f12__(solution)
        elif idx == 2 or idx == 3:
            return self.__f34__(solution)
        elif idx == 4 or idx == 5:
            return self.__f56__(solution)
        elif idx == 6 or idx == 7:
            return self.__f78__(solution)
        else:
            return self.__f910__(solution)

    def _main__(self, solution=None):
        problem_size = len(solution)
        if problem_size > 100:
            print("CEC 2005 not support for problem size > 100")
            return 1
        if problem_size == 10 or problem_size == 30 or problem_size == 50:
            self.f_matrix = "hybrid_func1_M_D" + str(problem_size)
        else:
            print("CEC 2005 F17 function only support problem size 10, 30, 50")
            return 1
        num_funcs = 10
        C = 2000
        xichma = ones(problem_size)
        lamda = array([1, 1, 10, 10, 5.0 / 60, 5.0 / 60, 5.0 / 32, 5.0 / 32, 5.0 / 100, 5.0 / 100])
        bias = array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
        y = 5 * ones(problem_size)
        shift_data = self.load_matrix_data(self.f_shift_data_file)
        shift_data = shift_data[:, :problem_size]
        matrix = self.load_matrix_data(self.f_matrix)

        weights = ones(num_funcs)
        fits = ones(num_funcs)
        for i in range(0, num_funcs):
            w_i = exp(-sum((solution - shift_data[i]) ** 2) / (2 * problem_size * xichma[i] ** 2))
            z = dot((solution - shift_data[i]) / lamda[i], matrix[i * problem_size:(i + 1) * problem_size, :])
            fit_i = self.__fi__(z, i)
            f_maxi = self.__fi__(dot((y / lamda[i]), matrix[i * problem_size:(i + 1) * problem_size, :]), i)
            fit_i = C * fit_i / f_maxi

            weights[i] = w_i
            fits[i] = fit_i

        sw = sum(weights)
        maxw = max(weights)

        for i in range(0, num_funcs):
            if weights[i] != maxw:
                weights[i] = weights[i] * (1 - maxw ** 10)
            weights[i] = weights[i] / sw

        fx = sum(dot(weights, (fits + bias)))
        return fx * (1 + 0.2 * abs(normal(0, 1))) + self.f_bias


print("""F17. Rotated Hybrid Composition Function 1 with Noise in Fitness, 
        f(x*) = 120""")

x_lb = -5
x_up = 5


def domf():
    return x_lb, x_up


def domfok(x):
    x = array(x)
    if all(x >= x_lb) and all(x <= x_up):
        return True
    else:
        return False


f17 = Model()


def CEC_F17(x):
    x = array(x)
    return f17._main__(x)

# if __name__ == "__main__":
#     print(domf())
#     a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     print(CEC_F17(a))