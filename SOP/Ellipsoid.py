import numpy as np

print("""Ellipsoid function
        global minimum:
        f(x)=0; x(i)=0, i=1:n.
  """)

x_lb = -5.12
x_up = 5.12


def domf():
    return x_lb, x_up


def domfok(x):
    x = np.array(x)
    if np.all(x >= x_lb) and np.all(x <= x_up):
        return True
    else:
        return False


def Ellipsoid(x):
    """
        function [y] = Ellipsoid(x)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % ELLIPSOID FUNCTION
        %
        % For function details and reference information, see:
        % A Gaussian Process Surrogate Model Assisted Evolutionary Algorithm for
        % Medium Scale Expensive Optimization Problems
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
            out = 0.0
            d = len(x)
            d_L = [i for i in range(1, d + 1)]
            for i_ in d_L:
                out += i_ * x[i_ - 1] ** 2

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
#     a = [-4.35594927713958, -4.47381636626738, 0.118284816286218, 2.34500241717270,
#          3.29837941060865, 0.778773792847334, 4.87406181691801, -3.01232260406203,
#          1.10090270495220, -2.45517915479811]
#     print(Ellipsoid(a))
