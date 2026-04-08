# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# -----------------------------------------------------------------------------------

import random

VAR_RANGE = 50000
z = []
x1 = [random.uniform(13, 100) for i in range (VAR_RANGE)]
x2 = [random.uniform(0, 100) for i in range (VAR_RANGE)]
x3 = []
x4 = []

def g1(x1, x2):
    return -(x1 - 5)**2 - (x2 - 5)**2 + 100
 
def g2(x1, x2):
    return (x1 - 6)**2 + (x2 - 5)**2 -82.81

def brute_force_g06(x1, x2):
    fx = (x1 - 10)**3 + (x2 - 20)**3
    return fx

for i in range(VAR_RANGE):
    result = (brute_force_g06(x1[i],x2[i]))
    g1result = g1(x1[i], x2[i])
    g2result = g2(x1[i], x2[i])

    if g1result > 0:
        result = result+ (g1result*2000)
    if g2result > 0:
        result = result+ (g2result*2000)

    z.append(result)
    x3.append(g1result)
    x4.append(g2result)



print(z)
melhor_resultado = min(z)
indiceMelhor = z.index(melhor_resultado)

print("\nMenor valor encontrado: \n")
print(melhor_resultado)
print(f"g1 = {x3[indiceMelhor]}\ng2 = {x4[indiceMelhor]}")
print(f"x1 = {x1[indiceMelhor]}\nx2 = {x2[indiceMelhor]}")
