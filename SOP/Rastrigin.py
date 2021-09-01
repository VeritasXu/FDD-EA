import numpy as np

print("""Rastrigin function
        global minimum:
        f(x)=f_bias; x(i)=O[i], i=1:n.
  """)

x_lb = -5
x_up = 5


def domf():
    return x_lb, x_up


def domfok(x):
    x = np.array(x)
    if np.all(x >= x_lb) and np.all(x <= x_up):
        return True
    else:
        return False


def Rastrigin(x):
    """
        function [y] = Rastrigin(x)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % F6: Rastrigin’s Function
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
            F = 0
            d_x = len(x)
            for i in range(0, d_x):
                # z = x[i] - rastrigin[i]
                z = x[i]
                F = F + (z ** 2 - 10 * np.cos(2 * np.pi * z) + 10)
            return F

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
#     a = [-2, -4, 2.8]
#     print(Rastrigin(a))
