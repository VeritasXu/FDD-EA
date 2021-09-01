#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:29, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
# -------------------------------------------------------------------------------------------------------%

from SOP.cec2005.root import Root
from numpy import sum, array


class Model(Root):
    def __init__(self, f_name="Shifted Sphere Function", f_shift_data_file="data_sphere", f_ext='.txt', f_bias=-450):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)

    def _main__(self, solution=None):
        problem_size = len(solution)
        if problem_size > 100:
            print("CEC 2005 not support for problem size > 100")
            return 1
        shift_data = self.load_shift_data()[:problem_size]
        result = sum((solution - shift_data) ** 2) + self.f_bias
        return result


print("""F1.  Shifted Sphere Function, 
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


f1 = Model()


def CEC_F1(x):
    x = array(x)
    return f1._main__(x)

# if __name__ == "__main__":
#     print(domf())
#     a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     print(CEC_F1(a))
