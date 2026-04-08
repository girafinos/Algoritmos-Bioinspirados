# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# -----------------------------------------------------------------------------------

import numpy as np
import random
import matplotlib.pyplot as plt

VAR_RANGE = 100000
x = [random.uniform(0, 1) for i in range (VAR_RANGE)]
y = []

def f2x (x):
    return np.sin(5 * np.pi * x) ** 6

for i in range(VAR_RANGE):
    y.append(f2x(x[i]))

xy = sorted(zip(x, y))
x_sorted, y_sorted = zip(*xy)
plt.plot(x_sorted, y_sorted)
plt.xlabel('index')
plt.ylabel('f2x(x)')    
plt.title('CEC@2013 F2 approximation')
plt.grid(True)

# plt.ylim(0.99995, 1)

plt.savefig('cec_2013_f2.png', dpi=150, bbox_inches='tight')
plt.close()


