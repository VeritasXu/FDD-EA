import numpy as np

print("""Ackley function
        global minimum:
        f(x)=f_bias; x(i)=O[i], i=1:n.
  """)

x_lb = -32.768
x_up = 32.768


def domf():
    return x_lb, x_up


def domfok(x):
    x = np.array(x)
    if np.all(x >= x_lb) and np.all(x <= x_up):
        return True
    else:
        return False


def Ackley(x):
    """
        function [y] = Ackley(x)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % F8: Ackley Function
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
            sum1 = 0
            sum2 = 0
            d_x = len(x)
            # M =1 no rotated
            M = 1
            for i in range(0, d_x):
                # z = M * (x[i] - ackley[i])
                z = x[i]
                sum1 = sum1 + z ** 2
                sum2 = sum2 + np.cos(2 * np.pi * z)
            out = -20 * np.exp(-0.2 * np.sqrt(sum1 / d_x)) - np.exp(sum2 / d_x) + 20 + np.e
            # F += f_bias[5]
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
#     a = [-8, -19, 25]
#     print(Ackley(a))
