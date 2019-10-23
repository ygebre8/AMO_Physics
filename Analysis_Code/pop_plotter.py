import numpy as np
import sys
import h5py
import Module
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg') 

Populations, n_l_pops, l_m_pops, n_m_pops, n_l_pop_fixed_ms, n_m_pop_fixed_ls, l_m_pop_fixed_ns, TDSE_files, Target_files, Pulse_files, file_names = Module.File_Organizer(sys.argv)
# Module.N_L_Population_Plotter(n_l_pops[0], TDSE_files[0], Target_files[0], file_name = "Counter_Rotating.png")
# Module.N_M_Population_Plotter(n_m_pops[0], TDSE_files[0], Target_files[0], file_name = "N_M_Population_Counter_Rotating.png")
# Module.N_L_Population_Fixed_M(n_l_pop_fixed_ms[0], TDSE_files[0], Target_files[0])
# Module.L_M_Population_Fixed_N(l_m_pop_fixed_ns[0], TDSE_files[0], Target_files[0])


def M_Distribution(Population):
    m_array = {}
    m_values = np.arange(-13, 14, 1)
    for m in m_values:
        if m == 0:
            continue
        m_array[m] = 0.0

    for k in Population.keys():
        m = k[2]
        if m == 0:
            continue
        m_array[m] += Population[k]
    
    plt.semilogy(m_array.keys(), m_array.values())
    plt.title("Distribution of m quantum number")
    plt.xlabel("M Quantum Number")
    plt.ylabel("Population")
    plt.savefig("M_dist_line.png")

def M_Distribution_N(Population):
    n_array_pos = {}
    n_array_neg = {}
    n_values = np.arange(1, 15)
    for n in n_values:
        n_array_pos[n] = 0.0
        n_array_neg[n] = 0.0

    for k in Population.keys():
        n = k[0]
        m = k[2]
        if m > 0:
            n_array_pos[n] += Population[k]
        if m < 0:
            n_array_neg[n] += Population[k]

    ratio = {}
    for n in n_values:
        if n == 1:
            continue
        ratio[n] = n_array_neg[n] / n_array_pos[n]
    # plt.plot(n_array_pos.keys(), n_array_pos.values())
    plt.semilogy(ratio.keys(), ratio.values(), 'o')
    plt.axhline(y=1, color='r')
    plt.title("Ratio of negative m / positive m")
    plt.xlabel("N Quantum Number")
    plt.ylabel("Ratio")
    plt.savefig("N_dist_ratio05.png")

M_Distribution_N(Populations[0])

# n_l_pop = n_l_pops[0]
# n_values = np.arange(1, 15)
# l_values = np.arange(1, 15)

# l_pop = {}

# for l in l_values:
#     l_pop[l] = 0.0
# for n in n_values:
#         for l  in np.arange(1, n):
#             l_pop[l] += n_l_pop[(n,l)]

# for l in l_values:
#     print(l, np.log10(l_pop[l]))

# labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]
# fig, ax = plt.subplots()
# ax.bar(l_values ,l_pop.values(), log=True,ec="k", align="center")
# ax.axvline(x=4.2, color="r")
# ax.set_xticks(l_values)
# ax.set_xticklabels(labels)


# fig = plt.figure()
# ax = plt.gca()
# ax.bar(l_values ,l_pop.values(), log=True,)

# ax.set_yscale('log')

# plt.scatter(l_values, l_pop.values(), logy=True)
# plt.yscale('log')
# plt.title(r'$I_{400} = 5x10^{12}$ $\mathrm{and}$ $I_{800} = 5x10^{13}$')
# plt.savefig("I400-5x10^12_I800-5x10^13.png")
# m_value = None
# l_max = np.amax(l_values) + 1
# n_max = np.amax(n_values) + 1
# heat_map = np.zeros(shape=(n_max + 1,l_max))

# for n in n_values:
#         for l  in l_values:
#             try:
#                 heat_map[n][l] = n_l_pop[(n, l)]
#                 if m_value != None:
#                     if l < abs(m_value):
#                         heat_map[n][l] = None
#             except:
#                 heat_map[n][l] = None
   
    
# heat_map[0][:] = None


# for n in np.arange(3, 12):
#     idx = n - 2
#     eval("plt.subplot(33" + str(idx) + ")")
#     l_values = np.arange(0, n)
#     plt.xticks(l_values)
#     plt.title("population for n = " + str(n))
#     plt.semilogy(l_values, heat_map[n][:n])
   
# fig, axes = plt.subplots(7)
# fig.suptitle('L Population for different n')


#     axes[n - 5].semilogy(heat_map[n][:])


# for i,k in enumerate(n_l_pop.keys()):
#     axes[i].plot(x, y)
# plt.subplots_adjust(left=0.075, bottom=0.05, right=0.975, top=.95, wspace=0.4, hspace=0.5)
# plt.savefig("plot.png")

# Module.N_M_Population_Fixed_L(n_m_pop_fixed_ls[0], TDSE_files[0], Target_files[0])

# for p in (Populations[0]).keys():
#     print(p, Populations[0][p])


## pop = n_m_pop_fixed_ls[0][1]
# # print(pop.keys())
# diff = {}
# for n in np.arange(2, 15):
#     diff[n] = pop[n, 1] / pop[n, -1]
# print(diff.values())
# Plot_data = []
# for TDSE, Target, n_l_pop in zip(TDSE_files, Target_files, n_l_pops):
#     Plot_data.append(Module.Organize_For_PP1D(TDSE, Target, n_l_pop))
# Module.Population_Plotter_1D(Plot_data, pow(10,-10), "pic.png", "plot")


# Module.HHG_Spectrum(TDSE_files[0], Pulse_files[0], "base_case")

# Ion_Rate = Module.Ionization_Rate(Populations)
# print(Ion_Rate)

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
