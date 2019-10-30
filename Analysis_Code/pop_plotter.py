import numpy as np
import sys
import h5py
import Module
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
plt.switch_backend('agg') 

Populations, n_l_pops, l_m_pops, n_m_pops, n_l_pop_fixed_ms, n_m_pop_fixed_ls, l_m_pop_fixed_ns, TDSE_files, Target_files, Pulse_files, file_names = Module.File_Organizer(sys.argv)
# Module.N_L_Population_Plotter(n_l_pops[0], TDSE_files[0], Target_files[0], file_name = "Co_Rotating.png")
# Module.N_M_Population_Plotter(n_m_pops[0], TDSE_files[0], Target_files[0], file_name = "N_M_Population_Counter_Rotating_Rescaled.png")
# Module.N_L_Population_Fixed_M(n_l_pop_fixed_ms[0], TDSE_files[0], Target_files[0])
# Module.L_M_Population_Fixed_N(l_m_pop_fixed_ns[0], TDSE_files[0], Target_files[0])


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
    plt.savefig("N_Distribution_2o75-13_5-13.png")

def L_Distribution(Population):
    excit = 0.0
    for k in Population.keys():
        if k[0] != 1:
            excit += Population[k]
        
    l_array = {}
    l_values = np.arange(1,15)
    for l in l_values:
        l_array[l] = 0.0
    for k in Population.keys():
        l = k[1]
        n = k[0]
        if l==0:
            continue
        l_array[l] += Population[k] / excit
    

    plt.bar(l_array.keys(), l_array.values(), align='center', alpha=1, log=True, color = 'darkblue')
    # plt.title()
    label = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]
    plt.xlabel("l Quantum Number")
    plt.ylabel("Population")
    plt.xticks(l_values, label)
    v=list(l_array.values())
    k=list(l_array.keys())

    plt.axvline(x=3, color='red', linewidth=3.0)

    plt.ylim(pow(10,-4), pow(10, 0))
    plt.savefig("5_13-5_12.png")


M_Distribution_N(Populations[0])

# L_Distribution(Populations[0])
