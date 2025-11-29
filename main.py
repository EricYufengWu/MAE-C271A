import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# True acceleration
# a(t) = 10 * sin (w*t)
w = 0.2 # rad/sec

# Accelarometer noise parameters: additive white Gaussian noise w with zero mean and variance V
V = 0.0004 # (meters/sec^2)^2


