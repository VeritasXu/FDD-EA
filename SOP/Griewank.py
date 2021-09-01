import numpy as np

print("""Griewank’s function
        global minimum:
        f(x)=f_bias; x(i)=O[i], i=1:n.
  """)

x_lb = -600
x_up = 600


def domf():
    return x_lb, x_up


def domfok(x):
    x = np.array(x)
    if np.all(x >= x_lb) and np.all(x <= x_up):
        return True
    else:
        return False


def Griewank(x):
    """
        function [y] = Griewank(x)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % F6: Griewank’s Function
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
            F1 = 0
            F2 = 1
            d_x = len(x)
            for i in range(0, d_x):
                # z = x[i] - griewank[i]
                z = x[i]
                F1 = F1 + (z ** 2 / 4000)
                F2 = F2 * (np.cos(z / np.sqrt(i + 1)))
            return F1 - F2 + 1

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
#     a = [-2.80, 100, -18]
#     print(Griewank(a))
