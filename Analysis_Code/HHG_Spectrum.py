import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import Module

try:
    number_of_files = int(sys.argv[1])
except:
    print("First argument should be the number of files")


for i in range(2, number_of_files + 2):
    file_name = sys.argv[i]
    try:
        x_axis = np.loadtxt(file_name + "/Harmonic.txt") 
        y_axis = np.loadtxt(file_name + "/HHG_Spectrum.txt")
    except:
        print("Making the HHG Spectrum for " + file_name)
        TDSE_file = h5py.File(file_name +  "/TDSE.h5","r")
        Pulse_file = h5py.File(file_name +  "/Pulse.h5", "r")
        Module.HHG_Spectrum(TDSE_file, Pulse_file, file_name)


x_axis = []
y_axis = []
print("Plotting the HHG Spectrums for Comparison")
for i in range(2, int(sys.argv[1]) + 2):
    file_name = sys.argv[i]
    x_axis.append(np.loadtxt(file_name + "/Harmonic.txt"))
    y_axis.append(np.loadtxt(file_name + "/HHG_Spectrum.txt"))


# # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
f, (ax1) = plt.subplots(1, 1,sharex='col', sharey='row')


for i in range(0, int(sys.argv[1])):
    ax1.semilogy(x_axis[i], y_axis[i])


x_min = 0
x_max = 25
ax1.set_xlim([x_min, x_max])
ax1.set_ylim(ymin=1e-8)
ax1.set_title("Co Rotating-base case")

for i in np.arange(3, 25, 3):
    plt.axvline(x=i, color="r")
plt.xticks(np.arange(1, 25, 1)) 
plt.savefig("base_case_co.png")
plt.show()
