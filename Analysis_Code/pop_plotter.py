import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import Module


Populations, n_l_pops, l_m_pops, n_l_pop_fixed_ms, TDSE_files, Target_files, file_names = Module.File_Organizer(sys.argv)

Ion_Rate = Module.Ionization_Rate(Populations)
Intensities = []
for f in file_names:
    Intensities.append(float(f[27 + 5:])) 

Energy = [0.499995, 0.44443945, 0.374995]
N_Photon = np.arange(8, 12, 1)

Resonance_N_2 = Module.Resonance_Condition(N_Photon, Energy[2], w=0.057)
Resonance_N_3 = Module.Resonance_Condition(N_Photon, Energy[1], w=0.057)
Resonance_I_P = Module.Resonance_Condition(N_Photon, Energy[0], w=0.057)
Resonances = [Resonance_N_2, Resonance_N_3, Resonance_I_P]

print(Resonance_N_2)

Module.Ionization_Rate_Plotter(Ion_Rate, Intensities, Resonances)
# for i, I in zip(Ion_Rate, Intensities):
#     print((np.log10(i), np.log10(I)))

# for i in np.arange(1, 15):
#     States = [[i],np.arange(0, 14)]
#     Module.State_Population(Populations, Intensities, States)

# States = [[2,3],np.arange(0, 14)]
# Module.State_Population(Populations, Intensities, States)
States = [[1],np.arange(0, 14)]
Module.State_Population(Populations, Intensities, States)
States = [[2],np.arange(0, 14)]
Module.State_Population(Populations, Intensities, States)
States = [[3],np.arange(0, 14)]
Module.State_Population(Populations, Intensities, States)
States = [[4],np.arange(0, 14)]
Module.State_Population(Populations, Intensities, States)
States = [np.arange(5, 15),np.arange(0, 14)]
Module.State_Population(Populations, Intensities, States)


def commented_out():
    comment = "organizational reasons"
    # Module.N_L_Population_Plotter(n_l_pops[0], TDSE_files[0], Target_files[0], file_name = "1x14_Corotating_a=1.png")

    #Module.N_L_Population_Fixed_M(n_l_pop_fixed_ms[0], TDSE_files[0], Target_files[0])


    # Plot_data = []
    # for TDSE, Target, n_l_pop in zip(TDSE_files, Target_files, n_l_pops):
    #     Plot_data.append(Module.Organize_For_PP1D(TDSE, Target, n_l_pop))

    # plotname = ""
    # for i in file_names:
    #     plotname += i[3:]
    # Module.Population_Plotter_1D(Plot_data, pow(10, -12), "Comparison.png", plotname)

    # Pop = T_P_WFs[1]
    # for k in Pop.keys():
    #     print((k, np.vdot(Pop[k], Pop[k])))

    # print(total)