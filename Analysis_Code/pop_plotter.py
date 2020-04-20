import numpy as np
import sys
import h5py
import Module
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
plt.switch_backend('agg') 

Populations, n_l_pops, l_m_pops, n_m_pops, n_l_pop_fixed_ms, n_m_pop_fixed_ls, l_m_pop_fixed_ns, TDSE_files, Target_files, Pulse_files, file_names = Module.File_Organizer(sys.argv)
# Module.N_L_Population_Plotter(n_l_pops[0], TDSE_files[0], Target_files[0], file_name = "N_L_Population_Joel_Length_35.png")
# Module.N_M_Population_Plotter(n_m_pops[0], TDSE_files[0], Target_files[0], file_name = "N_M_Population_Joel_Length_35.png")
# Module.N_L_Population_Fixed_M(n_l_pop_fixed_ms[0], TDSE_files[0], Target_files[0])
# Module.L_M_Population_Fixed_N(l_m_pop_fixed_ns[0], TDSE_files[0], Target_files[0])

pop = Populations[0]
ion = 0

for k in pop.keys():
    ion += pop[k]
    print(k, pop[k])
    
print((1.0 - ion)*100)
# error_1 = {}
# error_2 = {}
# count = 0
# ion = 0.0
# for k in pop.keys():
#     ion += pop[k]
#     m = k[2]
#     n = k[0]
    
#     if n > 1 and Populations[0][k] > pow(10,-10):  
    
#         error_1[k] = abs(Populations[0][k] - Populations[1][k]) / Populations[0][k]
#         # error_2[k] = Populations[1][k]
        
#     # if m > 0 and n > 1:  
#     #     error_1[k] = Populations[0][k]
#     # if m <= 0 and n > 1:  
#     #     error_2[k] = Populations[0][k]

# # for k in error_1.keys():
# #     print(k, error_1[k])
# # plt.semilogy(error.values(), '.')
# plt.semilogy(error_1.values(), 'r.')
# plt.semilogy(error_2.values(), 'b.')

# plt.semilogy(Populations[0].values(), '.r')
# plt.semilogy(Populations[1].values(), '.b')

# plt.savefig("Conv_Exp.png")

def M_Distribution(Population):
    m_array = {}
    m_values = np.arange(-13, 14, 1)
    for m in m_values:
        # if m == 0:
        #     continue
        m_array[m] = 0.0

    for k in Population.keys():
        m = k[2]
        n = k[0]
        if n == 1:
            continue
        m_array[m] += Population[k]
    
    plt.bar(m_array.keys(), m_array.values(), align='center', alpha=1, log = True, color = 'darkblue')
    plt.title("Distribution of m quantum number")
    plt.xlabel("M Quantum Number")
    plt.ylabel("Population")
    
    plt.ylim(pow(10,-5), 0.5*pow(10, -2))

    xticks = m_values
    xlabel = m_values
    plt.xticks(xticks)
    plt.xlim(-7.5, 7.5)

    plt.savefig("M_dist_line.png")

def M_Distribution_N(Population):
    n_array_pos = {}
    n_array_neg = {}
    n_values = np.arange(1, 15)
    excit = 0.0
    for n in n_values:
        n_array_pos[n] = 0.0
        n_array_neg[n] = 0.0

    for k in Population.keys():
        n = k[0]
        if n != 1:
            excit += Population[k]
    for k in Population.keys():
        n = k[0]
        m = k[2]
        if m > 0:
            n_array_pos[n] += Population[k] 
        if m < 0:
            n_array_neg[n] += Population[k] 

    ratio = {}
    white = {}
    blue_bar= {}
    white_two = {}

    for n in n_values:
        if n == 1:
            continue
        ratio[n] = (n_array_neg[n] - n_array_pos[n]) / (n_array_neg[n] + n_array_pos[n]) 
       
        white[n] = pow(10,0)
        if ratio[n] < pow(10, 0):
            blue_bar[n] = pow(10, 0)
            white_two[n] = ratio[n]
        else:
            blue_bar[n] = 0.0
    
   
    # plt.plot(list(ratio.keys())[0:], list(ratio.values())[0:],  'o')
    plt.bar(list(ratio.keys())[0:], list(ratio.values())[0:], align='center', alpha=1, color = 'darkblue')
    # plt.bar(list(white.keys())[0:], list(white.values())[0:], align='center', alpha=1, color = 'white')
    # plt.bar(list(blue_bar.keys())[0:], list(blue_bar.values())[0:], align='center', alpha=1, color = 'darkblue')
    # plt.bar(list(white_two.keys())[0:], list(white_two.values())[0:], align='center', alpha=1, color = 'white')
    plt.axhline(y=0, color='r')
    
    # plt.title("negative m / positive m")
    
    plt.xlabel("N Quantum Number")
    plt.ylabel("Ratio")
    # plt.yscale('symlog')

    xticks = n_values
    plt.xticks(xticks)
    plt.xlim(2.5, 15)
    
    plt.ylim(-1.1, 1.1)
    plt.savefig("N_Distribution_Ratio05.png")

def Weighted_M_distribution(Population):
    n_array = {}
    n_array_total = {}
    n_values = np.arange(1,15)

    for n in n_values:
        n_array[n] = 0.0
        n_array_total[n] = 0.0

    for k in Population.keys():
        n = k[0]
        n_array_total[n] += Population[k]
    
    for k in Population.keys():
        n = k[0]
        m = k[2]
        n_array[n] += m*Population[k] / n_array_total[n]

    plt.bar(list(n_array.keys())[0:], list(n_array.values())[0:], align='center', alpha=1, color = 'darkblue')
    plt.axhline(y=0, color='r')
    
    plt.title("Weighted M Distribution")
    
    plt.xlabel("N Quantum Number")
    plt.ylabel("Weighted M")
    # plt.yscale('symlog')

    xticks = n_values
    plt.xticks(xticks)
    plt.xlim(2.5, 15)
    
    # plt.ylim(-1.1, 1.1)
    plt.savefig("N_Distribution_1_13.png")

def L_Distribution(Population):
    excit = 0.0
    for k in Population.keys():
        if k[0] != 1:
            excit += Population[k]
        
    l_array = {}
    l_array_pos_m = {}
    l_array_neg_m = {}

    l_values = np.arange(1,15)

    for l in l_values:
        l_array[l] = 0.0
        l_array_neg_m[l] = 0.0
        l_array_pos_m[l] = 0.0

    for k in Population.keys():
        n = k[0]
        l = k[1]
        m = k[2]
        if l==0:
            continue
        if m > 0:
            l_array_pos_m[l] += Population[k]/ excit
        if m < 0:
            l_array_neg_m[l] += Population[k]/ excit

        l_array[l] += Population[k] / excit
    

    plt.bar(l_array.keys(), l_array.values(), align='center', alpha=1, log=True, color = 'black')
    
    # plt.bar(l_array_neg_m.keys(), l_array_neg_m.values(), align='center', alpha=1, log=True, color = 'red')
    
    label = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]
    plt.xlabel("l Quantum Number")
    plt.ylabel("Population")
    plt.xticks(l_values, label)
    v=list(l_array.values())
    k=list(l_array.keys())

    # plt.axvline(x=3, color='red', linewidth=3.0)

    plt.ylim(pow(10,-4), pow(10, 0))
    plt.savefig("5_13-1_13.png")
    # plt.clf()

    # plt.bar(l_array_pos_m.keys(), l_array_pos_m.values(), align='center', alpha=1, log=True, color = 'darkblue')
    # plt.xlabel("l Quantum Number")
    # plt.ylabel("Population")
    # plt.xticks(l_values, label)
    # v=list(l_array.values())
    # k=list(l_array.keys())

    # # plt.axvline(x=3, color='red', linewidth=3.0)

    # plt.ylim(pow(10,-4), pow(10, 0))
    # plt.savefig("5_12-5_13_L_Dist_Pos.png")

# Weighted_M_distribution(Populations[0])

# L_Distribution(Populations[0])
