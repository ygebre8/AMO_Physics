import numpy as np
import matplotlib.pyplot as plt



### Ratio1

Counter_Ratio_1 =  np.array([2.16, 21.9, 82.53, 99.24])
Counter_Ratio_1_x = np.array([0.5, 1, 2, 3])

Co_Rotating_Ratio_1 = np.array([0.9, 11.25, 79.94, 99.18])
Co_Rotating_Ratio_1_x = np.array([0.5, 1, 2, 3])

plt.plot(Counter_Ratio_1_x, Counter_Ratio_1)
plt.title("Corotating Ratio 1")
plt.savefig("Corotating_Ratio_1.png")