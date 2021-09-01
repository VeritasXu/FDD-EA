import numpy as np

print("""Schwefel function
        global minimum:
        f(x)=f_bias; x(i)=O[i], i=1:n.
  """)

x_lb = -500
x_up = 500


def domf():
    return x_lb, x_up


def domfok(x):
    x = np.array(x)
    if np.all(x >= x_lb) and np.all(x <= x_up):
        return True
    else:
        return False


def Schwefel(x):
    """
        function [y] = Schwefel(x)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % F8: Generalized Schwefel’s Problem 2.26
        %
        % For function details and reference information, see:
        % Evolutionary computation with biogeography-based optimization
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % INPUT:
        %
        % xx = [x1, x2, ..., xd]
        % x* = [420.9687, ..., 420.9687]
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    """
    try:
        x = np.array(x)
        if np.isrealobj(x) and domfok(x):
            out = 0
            d_x = len(x)
            for i in range(d_x):
                out += x[i] * np.sin(np.sqrt(abs(x[i])))
            return -out

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
#     a = 420.9687* np.ones(50)
#     print(Schwefel(a))
