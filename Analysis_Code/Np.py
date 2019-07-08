import numpy as np
from sympy.solvers import solve
from sympy import Symbol
from decimal import Decimal
I = Symbol('I')

I_unit = 3.5*pow(10, 16)
w_4 = 0.114 
w_8 = 0.057

N_p = np.arange(8, 11+0.1, 0.1)
Energy = [0.499995, 0.44443945, 0.374995]

Intensity = []

r = 3.2
U_p = 0.5
# soln = solve(I*r + 4*I - 16*U_p*pow(w_8, 2.0), I)
# soln = solve(I*pow(w_4, 2.0) + I*pow(w_8, 2.0) - 4*pow(w_4, 2.0)*pow(w_8, 2.0)*(N*w_8 - E), I)

E = Energy[2]
for N in N_p:
    # soln = solve(I*pow(w_4, 2.0) + I*pow(w_8, 2.0) - 4*pow(w_4, 2.0)*pow(w_8, 2.0)*(N*w_8 - E), I)
    soln = solve(I - 4*pow(w_8, 2.0)*(N*w_8 - E), I)
    soln = soln[0] * I_unit
    soln = '%.5E' % Decimal(str(soln))
    Intensity.append((N, soln))


print(Intensity)