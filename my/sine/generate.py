from math import sin
from math import pi
from math import exp
from random import randint
from random import uniform
from numpy import array
# generate damped sine wave in [0,1]

def generate_sequence(length, period, decay):
    return [5 * sin(2 * pi * i / period) * exp(-0.1 * decay * i) for i in range(length)]
# generate input and output pairs of damped sine waves
def generate_examples(length, n_patterns, output):
    X, y = list(), list()
    for _ in range(n_patterns):
        p = randint(10, 20)
        d = uniform(0.01, 0.1)
        sequence = generate_sequence(length + output, p, d)
        X.append(sequence[:-output])
        y.append(sequence[-output:])
    X = array(X).reshape(n_patterns, length, 1)
    y = array(y).reshape(n_patterns, output)
    return X, y, p, d