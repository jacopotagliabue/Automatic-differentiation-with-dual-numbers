from math import cos, sin, isclose
import mygrad as mg
from random import randint


class DualNumber:
    
    def __init__(self, real: float, infinitesimal: float =0.0):
        self.real = real  # the standard or real part
        self.infinitesimal = infinitesimal  # the infinitesimal or dual part
        
        return
        
    def __add__(self, other):
        return DualNumber(self.real + other.real, self.infinitesimal + other.infinitesimal)
    
    def __mul__(self, other):
        return DualNumber(self.real * other.real, self.real * other.infinitesimal + self.infinitesimal * other.real)
    
    def __str__(self):
        return str("{} + {}".format(self.real, self.infinitesimal) if self.infinitesimal else self.real)
    

def derivative_with_my_grad(x: float):
    """
    
    Use the mygrad package to compute the derivative of x^2 at x
    
    """
    x = mg.tensor(x)
    fx = x ** 2
    fx.backward()
    
    return float(x.grad)


def dual():
    print("Welcome to differentiation with Dual Numbers!\n")
    # start with a simple example in the reals
    _sum = DualNumber(2.0) + DualNumber(3.5)
    print("Let's start with a simple example, printing out reals: {}\n".format(_sum))
    # let's define our test function: f(x) = (x + 2)(x + 1)
    test_f = lambda x: (x + DualNumber(2.0)) * (x + DualNumber(1.0))
    # let's define a derivative function, f'(x), taking a function and a value
    f_prime = lambda f, x: f(DualNumber(x) + DualNumber(0.0, 1.0)).infinitesimal
    # let's test it out
    x = 3.0
    print("Let's test our code with x = {}".format(x))
    # for the full calculation, check the blog post
    print("f({}) = {}, expected: {}".format(x, test_f(DualNumber(3.0)), 20.0))
    print("f'({}) = {}, expected: {}\n".format(x, f_prime(test_f, 3.0), 9.0))
    # let's add rules for sin and cos 
    dualCos = lambda x: DualNumber(cos(x.real), -sin(x.real))
    dualSin = lambda x: DualNumber(sin(x.real), cos(x.real))
    # let's test it out by mixing and matching on a function
    # (sin(x) * sin(x)) +  (3.0 * x^2) + (4.0 * x)
    sin_test_f = lambda x: (dualSin(x) * dualSin(x)) +  (DualNumber(3.0) * (x * x)) + (DualNumber(4.0) * x)
    # let's evaluate it at x = 4-11
    print([(_, f_prime(sin_test_f, float(_))) for _ in range(4, 11)])
    # let's check it at x = 6.0 now
    # https://www.wolframalpha.com/input?i=derivative+of+%28sin%28x%29+*+sin%28x%29%29+%2B++%283.0+*+x%5E2%29+%2B+%284.0+*+x%29+at+x%3D6.0
    x = 6.0
    print("f'({}) = {}, expected: {}\n".format(x, round(f_prime(sin_test_f, 6.0), 4), 39.4634))
    print("Now compare the results with the autograd function in myGrad:\n")
    # our test function is x^2
    grad_test_f = lambda x: x * x
    # pick a random range to compare the two implementations
    rand_start = randint(10, 50)
    rand_end = randint(50, 100)
    dual_results = [f_prime(grad_test_f, float(_)) for _ in range(rand_start, rand_end)]
    mygrad_results = [derivative_with_my_grad(float(_)) for _ in range(rand_start, rand_end)]
    top_k = 10
    print("Dual results (top {}): {}".format(top_k, dual_results[:top_k]))
    print("MyGrad results (top {}): {}".format(top_k, mygrad_results[:top_k]))
    # check if the results are close
    assert all(isclose(dual, mygrad) for dual, mygrad in zip(dual_results, mygrad_results))
    print("\n\nAll done! See you, space cowboys")
    return


if __name__ == "__main__":
    dual()