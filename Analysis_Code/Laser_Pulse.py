import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from scipy import fftpack

TDSE_File = h5py.File("TDSE.h5", "r")
Pulse_File = h5py.File("Pulse.h5", "r")


pulses = Pulse_File["Pulse"]
p_time = pulses["time"][:]
num_dims = TDSE_File["Parameters"]["num_dims"][0]
num_electrons = TDSE_File["Parameters"]["num_electrons"][0]
num_pulses = TDSE_File["Parameters"]["num_pulses"][0]
checkpoint_frequency = TDSE_File["Parameters"]["write_frequency_observables"][0]
energy = TDSE_File["Parameters"]["energy"][0]


print("Plotting E Field")
fig = plt.figure()
for dim_idx in range(num_dims):
    plt.plot(
        p_time,
        -1.0 * np.gradient(pulses["field_" + str(dim_idx)][:],
                           TDSE_File["Parameters"]["delta_t"][0]) * 7.2973525664e-3,
        label="field " + str(dim_idx))
plt.xlabel("Time (a.u.)")
plt.ylabel("Field (a.u.)")
plt.title("E Field")
plt.legend()
plt.savefig("Pulse_total_E_field.png")


E_x = -1.0 * np.gradient(pulses["field_" + str(0)][:], TDSE_File["Parameters"]["delta_t"][0]) * 7.2973525664e-3
E_y = -1.0 * np.gradient(pulses["field_" + str(1)][:], TDSE_File["Parameters"]["delta_t"][0]) * 7.2973525664e-3

E_Field = np.power(np.power(E_x, 2.0) + np.power(E_y, 2.0), 0.5)

sample_rate = int(np.size(p_time) / p_time[-1])

FFT = fftpack.fft(E_Field)
freq = fftpack.fftfreq(len(E_Field)) * sample_rate


fig, ax = plt.subplots()

ax.stem(freq, np.abs(FFT))
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_xlim(sample_rate / 2, sample_rate / 2)
# ax.set_ylim(-5, 110)

plt.savefig("test.png")


