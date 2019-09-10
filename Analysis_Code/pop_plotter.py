import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import Module


Populations, n_l_pops, l_m_pops, n_m_pops, n_l_pop_fixed_ms, n_m_pop_fixed_ls, TDSE_files, Target_files, file_names = Module.File_Organizer(sys.argv)
Module.N_L_Population_Plotter(n_l_pops[0], TDSE_files[0], Target_files[0], file_name = "Co_Rotating.png")
Module.N_M_Population_Plotter(n_m_pops[0], TDSE_files[0], Target_files[0], file_name = "N_M_Population_Co_Rotating.png")
Module.N_L_Population_Fixed_M(n_l_pop_fixed_ms[0], TDSE_files[0], Target_files[0])
# Module.N_M_Population_Fixed_L(n_m_pop_fixed_ls[0], TDSE_files[0], Target_files[0])

# pop = n_m_pop_fixed_ls[0][1]
# # print(pop.keys())
# diff = {}
# for n in np.arange(2, 15):
#     diff[n] = pop[n, 1] / pop[n, -1]
# print(diff.values())
# Plot_data = []
# for TDSE, Target, n_l_pop in zip(TDSE_files, Target_files, n_l_pops):
#     Plot_data.append(Module.Organize_For_PP1D(TDSE, Target, n_l_pop))
# Module.Population_Plotter_1D(Plot_data, pow(10,-10), "pic.png", "plot")

Ion_Rate = Module.Ionization_Rate(Populations)
print(Ion_Rate)

# Intensities = []
# for f in file_names:
#     Intensities.append(float(f[27:])) 
    

# Energy = [0.499995, 0.44443945, 0.374995]
# N_Photon = np.arange(5, 12, 1)

# w = 0.057 * 1.230769231

# Resonance_N_2 = Module.Resonance_Condition(N_Photon, Energy[2], w)
# Resonance_N_3 = Module.Resonance_Condition(N_Photon, Energy[1], w)
# Resonance_I_P = Module.Resonance_Condition(N_Photon, Energy[0], w)
# Resonances = [Resonance_N_2, Resonance_N_3, Resonance_I_P]

# # # # print(Resonances)
# print(Resonances[0])
# print(Resonances[1])
# print(Resonances[2])

# Module.Ionization_Rate_Plotter(Ion_Rate, Intensities, Resonances)
# for i, I in zip(Ion_Rate, Intensities):
#     print((np.log10(i), np.log10(I)))

# for i in np.arange(1, 15):
#     States = [[i],np.arange(0, 14)]
#     Module.State_Population(Populations, Intensities, States)

# States = [[2,3],np.arange(0, 14)]
# # Module.State_Population(Populations, Intensities, States)
# States = [[1],np.arange(0, 14)]
# Module.State_Population(Populations, Intensities, States)
# States = [[2],np.arange(0, 14)]
# Module.State_Population(Populations, Intensities, States)
# States = [[3],np.arange(0, 14)]
# Module.State_Population(Populations, Intensities, States)
# States = [[4],np.arange(0, 14)]
# Module.State_Population(Populations, Intensities, States)
# States = [np.arange(4, 15),np.arange(0, 14)]
# Module.State_Population(Populations, Intensities, States)


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