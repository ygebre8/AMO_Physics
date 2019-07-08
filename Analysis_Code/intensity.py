import numpy as np
omega = 0.057 * 8.0

# new
E_i = -0.5
E_f = -0.5 / 5.0**2

for i in range(1, 10):
    intensity = (i * omega + E_i - E_f) * 4 * omega * omega
    intensity_cor = (i * omega + (-0.5) - E_f) * 4 * omega * omega
    if intensity > 0:
        print (i, intensity * 3.50944758e16, (intensity - intensity_cor) * 3.50944758e16)
