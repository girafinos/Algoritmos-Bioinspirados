# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# -----------------------------------------------------------------------------------

import random

VAR_RANGE = 5000000
z = []
x1 = [random.uniform(-1, 1) for i in range (VAR_RANGE)]
x2 = [random.uniform(-1, 1) for i in range (VAR_RANGE)]

def g1(x1, x2):
    return x2 - x1**2

def brute_force_g06(x1, x2):
    fx = x1**2 + (x2 - 1)**2
    return fx

for i in range(VAR_RANGE):
    xi = x1[i]
    xj = x2[i]

    result = brute_force_g06(xi, xj)
    g1result = g1(xi, xj)

    if abs(g1result) <= 0.000001:
        z.append((result, xi, xj))

melhor = min(z, key=lambda t: t[0])

print("\nMenor valor encontrado:\n")
print(melhor[0])
print(f"x1 = {melhor[1]}\nx2 = {melhor[2]}")