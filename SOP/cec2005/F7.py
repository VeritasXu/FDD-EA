#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 15:29, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from SOP.cec2005.root import Root
from numpy import sum, dot, cos, sqrt, array


class Model(Root):
    def __init__(self, f_name="Shifted Rotated Griewank's Function without Bounds", f_shift_data_file="data_griewank",
                 f_ext='.txt', f_bias=-180, f_matrix=None):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)
        self.f_matrix = f_matrix

    def _main__(self, solution=None):
        problem_size = len(solution)
        if problem_size > 100:
            print("CEC 2005 not support for problem size > 100")
            return 1
        if problem_size == 10 or problem_size == 30 or problem_size == 50:
            self.f_matrix = "griewank_M_D" + str(problem_size)
        else:
            print("CEC 2005 F8 function only support problem size 10, 30, 50")
            return 1

        shift_data = self.load_shift_data()[:problem_size]
        matrix = self.load_matrix_data(self.f_matrix)

        z = dot((solution - shift_data), matrix)

        vt1 = sum(z**2) / 4000 + 1
        result = 1.0
        for i in range(0, problem_size):
            result *= cos(z[i] / sqrt(i+1))
        return vt1 - result + self.f_bias

print("""F7. Shifted Rotated Griewank's Function without Bounds, 
        f(x*) = -180""")

x_lb = 0
x_up = 600


def domf():
    return x_lb, x_up


def domfok(x):
    x = array(x)
    if all(x >= x_lb) and all(x <= x_up):
        return True
    else:
        return False


f7 = Model()


def CEC_F7(x):
    x = array(x)
    return f7._main__(x)

# if __name__ == "__main__":
#     print(domf())
#     a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     print(CEC_F7(a))