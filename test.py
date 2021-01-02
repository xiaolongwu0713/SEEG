import kf_book.kf_internal as kf_internal
from kf_book.kf_internal import DogSimulation
import numpy as np
import filterpy.stats as stats
from collections import namedtuple
from filterpy.kalman import predict

x = np.array([10.0, 4.5])
P = np.diag([500, 49])
F = np.array([[1, dt], [0, 1]])
# Q is the process noise
x, P = predict(x=x, P=P, F=F, Q=0)
print('x =', x)