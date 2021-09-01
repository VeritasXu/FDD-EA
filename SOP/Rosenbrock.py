import numpy as np

print("""Rosenbrock's function
        global minimum:
        f(x)=f_bias; x(i)=O[i], i=1:n.
  """)

x_lb = -2.048
x_up = 2.048


def domf():
    return x_lb, x_up


def domfok(x):
    x = np.array(x)
    if np.all(x >= x_lb) and np.all(x <= x_up):
        return True
    else:
        return False


def Rosenbrock(x):
    """
        function [y] = Rosenbrock(x)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % F6: Rosenbrock’s Function
        %
        % For function details and reference information, see:
        % A Gaussian Process Surrogate Model Assisted Evolutionary Algorithm for Medium Scale Expensive Optimization Problems
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % INPUT:
        %
        % xx = [x1, x2, ..., xd]
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    """
    try:
        x = np.array(x)

        if np.isrealobj(x) and domfok(x):

            out = 0
            d_x = len(x)

            for i in range(0, d_x - 1):
                tmp = 100 * np.power(x[i] ** 2 - x[i + 1], 2) + np.power(x[i] - 1, 2)
                out += tmp
            # F += f_bias[2]
            return out

        else:
            raise Exception("Error")
    except Exception as inst:
        if inst.__str__() == "Error":
            print("input out of domain: " + str(domf()))
            print("input must be vector: [1,2,...]")
        else:
            print(inst)

# if __name__ == "__main__":
#     print(domf())
#     a = [-1, 2, -1.86]
#     print(Rosenbrock(a))
