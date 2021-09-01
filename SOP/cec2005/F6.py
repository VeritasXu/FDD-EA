#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 15:21, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from SOP.cec2005.root import Root
from numpy import array


class Model(Root):
    def __init__(self, f_name="Shifted Rosenbrock's Function", f_shift_data_file="data_rosenbrock",
                 f_ext='.txt', f_bias=390):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)

    def _main__(self, solution=None):
        problem_size = len(solution)
        if problem_size > 100:
            print("CEC 2005 not support for problem size > 100")
            return 1
        shift_data = self.load_shift_data()[:problem_size]
        z = solution - shift_data + 1
        result = 0
        for i in range(0, problem_size-1):
            result += (100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2)
        return result + self.f_bias


print("""F6. Shifted Rosenbrock's Function, 
        f(x*) = 390""")

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


f6 = Model()


def CEC_F6(x):
    x = array(x)
    return f6._main__(x)

# if __name__ == "__main__":
#     print(domf())
#     a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     print(CEC_F6(a))