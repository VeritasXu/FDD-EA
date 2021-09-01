#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:07, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
# -------------------------------------------------------------------------------------------------------%

from SOP.cec2005.root import Root
from numpy import sum, array


class Model(Root):
    def __init__(self, f_name="Shifted Schwefel's Problem 1.2", f_shift_data_file="data_schwefel_102", f_ext='.txt',
                 f_bias=-450):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)

    def _main__(self, solution=None):
        problem_size = len(solution)
        if problem_size > 100:
            print("CEC 2005 not support for problem size > 100")
            return 1
        shift_data = self.load_shift_data()[:problem_size]

        result = 0
        for i in range(0, problem_size):
            result += (sum((solution[:i] - shift_data[:i]))) ** 2
        return result + self.f_bias


print("""F2.  Shifted Schwefel’s Problem 1.2, 
        f(x*) = -450""")

x_lb = -100
x_up = 100


def domf():
    return x_lb, x_up


def domfok(x):
    x = array(x)
    if all(x >= x_lb) and all(x <= x_up):
        return True
    else:
        return False


f2 = Model()


def CEC_F2(x):
    x = array(x)
    return f2._main__(x)

# if __name__ == "__main__":
#     print(domf())
#     a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     print(CEC_F2(a))
