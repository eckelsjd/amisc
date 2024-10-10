import numpy as np


def f1(x):
    y1 = x * np.sin(np.pi * x)
    return y1


def f2(y1):
    y2 = 1 / (1 + 25*y1**2)
    return y2


def f3(x, y2):
    y3 = x * np.cos(np.pi * y2)
    return y3


def f3_dict(inputs):
    return {'y3': inputs['x'] * np.cos(np.pi * inputs['y2'])}
