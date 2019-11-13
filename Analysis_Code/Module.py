if True:
    import numpy as np
    import matplotlib
    # matplotlib.use('agg')
    import matplotlib.pyplot as plt
    plt.switch_backend('agg') 
    import matplotlib.colors as colors
    import matplotlib as mpl
    import seaborn as sns
    from sympy.solvers import solve
    from sympy import Symbol
    from decimal import Decimal
    import os
    import sys
    import h5py
    import json


def Time_Propagated_Wavefunction_Reader(TDSE_file):
	T_P_WF = {}

	time = np.array(TDSE_file['Wavefunction']['time'])
	psi = TDSE_file['Wavefunction']['psi'][time.size - 1]
	psi = psi[:,0] + 1.0J*psi[:,1] 

	l_values = np.array(TDSE_file['Wavefunction']['l_values'])
	m_values = np.array(TDSE_file['Wavefunction']['m_values'])
	r_ind_max = TDSE_file['Wavefunction']['x_value_2'].size
	r_ind_lower = 0
	r_ind_upper = r_ind_max 

	for i in range(len(l_values)):
		T_P_WF[(l_values[i], m_values[i])] = np.array(psi[r_ind_lower: r_ind_upper])
		r_ind_lower = r_ind_upper
		r_ind_upper = r_ind_upper + r_ind_max
        
	norm = 0.0
	for k in T_P_WF.keys():
		norm += np.vdot(T_P_WF[k], T_P_WF[k])
    	
	print(norm)

	return T_P_WF

def Field_Free_Wavefunction_Reader(Target_file):
    F_F_WF = {}
    n_max = int(Target_file['Energy_l_0'].size / 2)
    n_values = np.arange(1, n_max + 1)
    
    for i in range(n_max):
        group_name = "psi_l_" + str(i)
        for k in range(n_max - i):
            F_F_WF[(k + 1 + i, i)] = Target_file[group_name]['psi'][k]
            F_F_WF[(k + 1 + i, i)] = np.array(F_F_WF[(k + 1 + i, i)][:,0] + 1.0J*F_F_WF[(k + 1 + i, i)][:,1])
	
    return(F_F_WF)

def Population_Calculator(TDSE_file, Target_file, T_P_WF, F_F_WF):
    Population = {}
    l_m_pop = {}
    n_l_pop = {}
    n_m_pop = {}
    n_l_pop_fixed_m = {}
    n_m_pop_fixed_l = {}
    l_m_pop_fixed_n = {}

    l_values = np.array(TDSE_file['Wavefunction']['l_values'])
    m_values = np.array(TDSE_file['Wavefunction']['m_values'])
    m_max = np.amax(m_values)
    n_max = int(Target_file['Energy_l_0'].size / 2)
    n_values = np.arange(1, n_max + 1)
    
    
    for l, m  in zip(l_values, m_values):
        l_m_pop[(l, m)] = pow(10, -20)


    for l in range(n_max):
        for n in range(l + 1, n_max + 1):
            n_l_pop[(n, l)] = pow(10, -20)
    
    for l in range(0, np.amax(l_values) + 1):
        for m in range(-1 * l, l + 1):
            for n in range(l + 1, n_max + 1):
                    n_m_pop[(n, m)] = pow(10, -20)


    for m in range(-1 * m_max, m_max + 1):
        n_l_pop_fixed_m[m] = {}
        for l in range(n_max):
            for n in range(l + 1, n_max + 1):
                n_l_pop_fixed_m[m][(n, l)] = pow(10, -20)
    
    for l in range(0, np.amax(l_values) + 1):
        n_m_pop_fixed_l[l] = {}
        for n in range(l + 1, n_max + 1):
            for m in range(-1 * l, l + 1):        
                n_m_pop_fixed_l[l][(n, m)] = pow(10, -20)

    
    for n in range(1, n_max + 1):
        l_m_pop_fixed_n[n] = {}
        for l in range(0, n):
            for m in range(-1 * l, l + 1):  
                l_m_pop_fixed_n[n][(l, m)] = pow(10, -20)
    


    for l, m  in zip(l_values, m_values):
        for n in range(l + 1, n_max + 1):
            Population[(n, l, m)] = np.vdot(F_F_WF[(n, l)], T_P_WF[(l, m)])
            Population[(n, l, m)] = np.power(np.absolute(Population[(n, l, m)]),2.0)
            
            n_l_pop[(n, l)] = n_l_pop[(n, l)] + Population[(n,l,m)]
            l_m_pop[(l, m)] = l_m_pop[(l, m)] + Population[(n,l,m)]
            n_m_pop[(n, m)] = n_m_pop[(n, m)] + Population[(n,l,m)]

            n_l_pop_fixed_m[m][(n, l)] = n_l_pop_fixed_m[m][(n, l)] + Population[(n,l,m)]
            n_m_pop_fixed_l[l][(n, m)] = n_m_pop_fixed_l[l][(n, m)] + Population[(n,l,m)]
            
            l_m_pop_fixed_n[n][(l, m)] = l_m_pop_fixed_n[n][(l, m)] + Population[(n,l,m)]

    # Pop_json = {}
    # for k in Population.keys():
    #     Pop_json[str(k[0]) + str(k[1]) + str(k[2])] = Population[k]
    # j = json.dumps(Pop_json)
    # file = open("Population_co_rotating.json", "w")
    # file.write(j)
    # file.close()
    return (Population, n_l_pop, l_m_pop, n_m_pop, n_l_pop_fixed_m, n_m_pop_fixed_l, l_m_pop_fixed_n)

def N_L_Population_Plotter(n_l_pop, TDSE_file, Target_file, m_value = None, vmax = None, file_name = "N_L_Population.png"):
    l_values = np.array(TDSE_file['Wavefunction']['l_values'])
    l_max = np.amax(l_values) + 1
    n_max = int(Target_file['Energy_l_0'].size / 2)
    n_values = np.arange(1, n_max + 1)
    heat_map = np.zeros(shape=(n_max + 1,l_max))
    
    for n in n_values:
        for l  in l_values:
            try:
                heat_map[n][l] = n_l_pop[(n, l)]
                if m_value != None:
                    if l < abs(m_value):
                        heat_map[n][l] = None
            except:
                heat_map[n][l] = None
   
    
    heat_map[0][:] = None
    
    figure, axes = plt.subplots()
   
    xticks = np.arange(0.5, n_max, 2)
    yticks = np.arange(1.5, n_max + 1, 2)
    xlabel = np.arange(0, n_max, 2)
    ylabel = np.arange(1, n_max + 1, 2)
    
    
    if vmax == None:
        max_elements = Max_Elements(n_l_pop)
        if max_elements[1] == 0:
            vmaxlog = -20
        else:
            vmaxlog = int(np.log10(max_elements[1]))
    else:
        if vmax == 0:
            vmaxlog = -20
        else:
            vmaxlog = int(np.log10(vmax))
    
    # vmaxlog = -2 
    print(vmaxlog)
    label = [pow(10, i) for i in range(vmaxlog - 5, vmaxlog)]
    vmax_num = pow(10, vmaxlog)
    vmin_num = pow(10, vmaxlog - 5)
    
    
    
    axes = sns.heatmap(heat_map, norm=colors.LogNorm(), yticklabels=ylabel, xticklabels=xlabel, linewidths=.5, 
    cmap="viridis", annot=False, cbar_kws={"ticks":label},  vmin=vmin_num, vmax=vmax_num,)
    #"viridis""
    
    plt.xticks(xticks, fontsize=14, rotation=90)
    plt.yticks(yticks, fontsize=14, rotation='vertical')

    plt.ylim(1, n_max + 1)
    plt.xlim(-0.5,n_max)

    plt.axvline(x=-0.5)
    plt.axhline(y=1)
    plt.xlabel('$\it{l}$', fontsize=20)
    plt.ylabel('$\it{n}$', fontsize=20)


    if m_value == None:
        plt.title("N and L states population for Co-Rotating", fontsize=11)
    else:
        plt.title("N and L states population for m = " + str(m_value))
    
    for tick in axes.get_xticklabels():
        tick.set_rotation(360)
    
    for tick in axes.get_yticklabels():
        tick.set_rotation(360)

    

    plt.savefig(file_name)
    plt.show()

def N_L_Population_Fixed_M(n_l_pop_fixed_m, TDSE_file, Target_file):
    n_max = int(Target_file['Energy_l_0'].size / 2)
    vmax = pow(10, -20)
    for m in n_l_pop_fixed_m.keys():
        vmax_current = Max_Elements(n_l_pop_fixed_m[m])[1]
        if(vmax_current > vmax):
            vmax = vmax_current
    for m in n_l_pop_fixed_m.keys():
        if abs(m) > 10: #n_max - 1:
            continue
        if m >= 0:
            file_name = "Population_Co-Rotating_For_M=" + str(m).zfill(2) + ".png" 
        else:
            file_name = "Population_Co-Rotating_For_M=" + str(m).zfill(3) + ".png"
        N_L_Population_Plotter(n_l_pop_fixed_m[m], TDSE_file, Target_file, m, vmax, file_name)

def N_M_Population_Plotter(n_m_pop, TDSE_file, Target_file, l_value = None, vmax = None, file_name = "N_M_Population_Counter_Rotating.png"):
    
    n_max = int(Target_file['Energy_l_0'].size / 2)
    n_values = np.arange(2, n_max + 1)
    m_values = np.array(TDSE_file['Wavefunction']['m_values']) 
    m_max = 2 * np.amax(m_values) + 1
    heat_map = np.zeros(shape=(n_max + 1, m_max))

    figure, axes = plt.subplots()
    xticks = np.arange(-1 * np.amax(m_values), np.amax(m_values) + 1)
    yticks = np.arange(0, n_max + 1)
    ylabel = np.arange(0, n_max + 1)
    xlabel = np.arange(-1 * np.amax(m_values), np.amax(m_values) + 1)

    plt.xticks(xticks)
    plt.yticks(yticks)

    if vmax == None:
        max_elements = Max_Elements(n_m_pop)
        if max_elements[1] == 0:
            vmaxlog = -20
        else:
            vmaxlog = int(np.log10(max_elements[1]))
    else:
        if vmax == 0:
            vmaxlog = -20
        else:
            vmaxlog = int(np.log10(vmax))
    
    label = [pow(10, i) for i in range(vmaxlog - 10, vmaxlog)]
    vmax_num = pow(10, vmaxlog)
    vmin_num = pow(10,  -10)

    for n in n_values:
        for m in m_values:
            try:
                heat_map[n][m + np.amax(m_values)] = n_m_pop[(n, m)]
            except:
                if(l_value != None and l_value >= n):
                    heat_map[n][m + np.amax(m_values)] = None
                


    axes = sns.heatmap(heat_map, norm=colors.LogNorm(), yticklabels=ylabel, xticklabels=xlabel, linewidths=.5, 
    cmap="viridis", annot=False, cbar_kws={"ticks":label},  vmin=vmin_num, vmax=vmax_num)

    plt.ylim(2, n_max + 1)
    # if(l_value == None):
    plt.xlim(np.amax(m_values) - n_max + 1, np.amax(m_values) + n_max)
    # else:
    #     plt.xlim(np.amax(m_values) - l_value + 1, np.amax(m_values) + l_value)
    plt.xlabel('m', fontsize=12)
    plt.ylabel('n', fontsize=12)

    if l_value == None:
        plt.title("N and M states population Co_Rotating", fontsize=12)
    else:
        plt.title("N and M states population Co_Rotating for l = " + str(l_value), fontsize=12)

    plt.savefig(file_name)
    plt.show()

def N_M_Population_Fixed_L(n_m_pop_fixed_l, TDSE_file, Target_file):
    n_max = int(Target_file['Energy_l_0'].size / 2)
    vmax = pow(10, -20)
    for l in n_m_pop_fixed_l.keys():
        vmax_current = Max_Elements(n_m_pop_fixed_l[l])[1]
        if(vmax_current > vmax):
            vmax = vmax_current
    for l in n_m_pop_fixed_l.keys():
        if l > 5:#n_max - 1:
            continue
        file_name = "N_M_Population_Counter_Rotating_For_L=" + str(l).zfill(4) + ".png"    
        N_M_Population_Plotter(n_m_pop_fixed_l[l], TDSE_file, Target_file, l, vmax, file_name)

def L_M_Population_Fixed_N(l_m_pop_fixed_n, TDSE_file, Target_file):
    n_max = int(Target_file['Energy_l_0'].size / 2)
    vmax = pow(10, -20)
    for n in l_m_pop_fixed_n.keys():
        vmax_current = Max_Elements(l_m_pop_fixed_n[n])[1]
        if(vmax_current > vmax):
            vmax = vmax_current
    for n in l_m_pop_fixed_n.keys():
        file_name = "Population_Co-Rotating_For_N=" + str(n).zfill(2) + ".png"
        L_M_Population_Plotter(l_m_pop_fixed_n[n], TDSE_file, Target_file, n, vmax, file_name)
 
def L_M_Population_Plotter(l_m_pop, TDSE_file, Target_file, n_value = None, vmax = None, file_name = "L_M_Population.png"):
    
    n_max = int(Target_file['Energy_l_0'].size / 2)
    n_values = np.arange(1, n_max + 1)
    l_values = np.array(TDSE_file['Wavefunction']['l_values'])
    m_values = np.array(TDSE_file['Wavefunction']['m_values']) 
    l_max = np.amax(l_values) + 1
    m_max = 2 * np.amax(m_values) + 1
    heat_map = np.zeros(shape=(l_max, m_max))
    

    figure, axes = plt.subplots()
    xticks = np.arange(-1 * np.amax(m_values), np.amax(m_values) + 1)
    yticks = np.arange(0, l_max)
    ylabel = np.arange(0, l_max)
    xlabel = np.arange(-1 * np.amax(m_values), np.amax(m_values) + 1)

    plt.xticks(xticks)
    plt.yticks(yticks)

    if vmax == None:
        max_elements = Max_Elements(l_m_pop)
        if max_elements[1] == 0:
            vmaxlog = -20
        else:
            vmaxlog = int(np.log10(max_elements[1]))
    else:
        if vmax == 0:
            vmaxlog = -20
        else:
            vmaxlog = int(np.log10(vmax))
    
    label = [pow(10, i) for i in range(vmaxlog - 5, vmaxlog)]
    vmax_num = pow(10, vmaxlog)
    vmin_num = pow(10, -10)

    for l, m  in zip(l_values, m_values):
        try:
            heat_map[l][m + np.amax(m_values)] = l_m_pop[l, m]
            # print((l,m,l_m_population[l, m]))
        except:
            heat_map[l][m + np.amax(m_values)] = pow(10, -20)

    axes = sns.heatmap(heat_map, norm=colors.LogNorm(), yticklabels=ylabel, xticklabels=xlabel, linewidths=.5, 
    cmap="cool", annot=False, cbar_kws={"ticks":label},  vmin=vmin_num, vmax=vmax_num)

    plt.ylim(-0.5, n_max)
    plt.xlim(np.amax(m_values) - n_max + 1, np.amax(m_values) + n_max)
    plt.xlabel('m_values', fontsize=12)
    plt.ylabel('l_values', fontsize=12)

    if n_value == None:
        plt.title("L and M states population for Counter-Rotating", fontsize=11)
    else:
        plt.title("L and M states population for N = " + str(n_value))

    plt.savefig(file_name)
    plt.show()

def M_Population_Fixed_L(l_m_pop, TDSE_file, Target_file):
    l_values = np.array(TDSE_file['Wavefunction']['l_values'])
    m_values = np.array(TDSE_file['Wavefunction']['m_values'])
    m_max = np.amax(m_values)
    l_max = np.amax(l_values)
    n_max = int(Target_file['Energy_l_0'].size / 2)
    n_values = np.arange(1, n_max + 1)

    m_pop = {}
    for l in range(0, l_max + 1):
        m_pop[l] = []
        for m in range(-1 * l, l + 1):
            m_pop[l].append(np.power(10.0, -20))
    
    for l, m in zip(l_values, m_values):
        # print(l_m_pop[(l, m)])
        m_pop[l][m + l] = m_pop[l][m + l] + np.log10(l_m_pop[(l, m)])    

    figure, axes = plt.subplots()
    plt.style.use('seaborn')
    for l in m_pop.keys():
        if(l == n_max):
            break
        y_axis = m_pop[l]
        x_axis = [m for m in range(-1 * l, l + 1)]
        plt.title("M Distribution for l = " + str(l))
        file_name = "M_Distribution_l=" + str(l).zfill(3) + ".png"
        plt.scatter(x_axis, y_axis)
        plt.savefig(file_name)
        plt.clf()
        
def L_Population_Plotter(TDSE_file_List):
    Population = []
    for i in TDSE_file_List:
        Population.append({})
    for TDSE_file, pop in zip(TDSE_file_List, Population):
        time = np.array(TDSE_file['Wavefunction']['time'])
        psi = TDSE_file["Wavefunction"]["psi"][time.size - 1]
        shape = TDSE_file["Wavefunction"]["num_x"][:]
        psi_cooridnate_values = []
        
        for dim_idx in np.arange(shape.shape[0]):
            psi_cooridnate_values.append(TDSE_file["Wavefunction"]["x_value_" + str(dim_idx)][:])
            
        l_values = TDSE_file["/Wavefunction/l_values"][:]
        m_values = TDSE_file["/Wavefunction/m_values"][:]
        psi = psi[:, 0] + 1j * psi[:, 1]
        
        psi.shape = shape
        psi_norm = np.sqrt((psi * psi.conjugate()).sum())
        for l in np.arange(0, psi_cooridnate_values[1].shape[0]):
            pop[l] = np.log10((np.sqrt((psi[0, l, :] * psi[0, l, :].conjugate()).sum()) / psi_norm).real)
        # plt.ylim(-10, 0)
        plt.plot(pop.keys(), pop.values())
        plt.savefig("pic.png")
    
def Organize_For_PP1D(TDSE_file, Target_file, n_l_population):
    l_values = np.array(TDSE_file['Wavefunction']['l_values'])
    n_max = int(Target_file['Energy_l_0'].size / 2)
    n_values = np.arange(1, n_max + 1)
    
    l_n_population = {}
    for l in range(n_max):
        for n in range(l + 1, n_max + 1):
            l_n_population[(n, l)] = n_l_population[(n, l)]


    l_value_index = [0]
    l_value_label = [str(0)]
    l_value = 0
    x_axis = []
    y_axis = []
    for i, k in enumerate(l_n_population.keys()):
        x_axis.append(i)
        y_axis.append(l_n_population[k])
        if(k[1] > l_value):
            l_value_index.append(i)
            l_value_label.append(str(k[1]))
            l_value = k[1]

    Plot_data = (x_axis, y_axis, l_value_index, l_value_label)

    return(Plot_data)

def Population_Plotter_1D(Plot_data_List, y_min_limit, file_name, plot_name):
    # fig, axes = plt.subplots(1, 2)
    locs, labels = plt.xticks()
    l_value_index = Plot_data_List[0][2]
    l_value_label = Plot_data_List[0][3]

    plt.xticks(l_value_index, l_value_label)
    
    for i, p in enumerate(Plot_data_List):
        plt.plot(p[0], p[1], label=str(i))
        # axes[i].plot(p[0], p[1], label=str(i))
        # axes[i].set_yscale('log')
    
    plt.yscale('log')
    plt.ylim(y_min_limit, 1)
    plt.title(plot_name)
    plt.savefig(file_name)
    plt.show()

def Max_Elements(input_dict):
    max_element = 0.0
    max_element_second = 0.0

    for k in input_dict.keys():
        if(input_dict[k] > max_element):
            max_element = input_dict[k]
    for k in input_dict.keys():
        if(input_dict[k] > max_element_second and input_dict[k] < max_element):
            max_element_second = input_dict[k]
    return (max_element, max_element_second)

def HHG_Spectrum(TDSE_file, Pulse_file, file_name):
    observables = TDSE_file["Observables"]
    pulses = Pulse_file["Pulse"]
    time = observables["time"][1:]
    num_dims = TDSE_file["Parameters"]["num_dims"][0]
    num_electrons = TDSE_file["Parameters"]["num_electrons"][0]
    checkpoint_frequency = TDSE_file["Parameters"]["write_frequency_observables"][0]

    energy = shifted_energy(TDSE_file, 0)
    HHG_spec = np.array([])
    Harmonic = np.array([])
    for elec_idx in range(num_electrons):
        for dim_idx in range(num_dims):
            if (not (dim_idx == 0
                 and TDSE_file["Parameters"]["coordinate_system_idx"][0] == 1)):
                data = observables[
                "dipole_acceleration_" + str(elec_idx) + "_" + str(dim_idx)][
                    1:len(pulses["field_" + str(dim_idx)]
                          [checkpoint_frequency::checkpoint_frequency]) + 1]
                data = data * np.blackman(data.shape[0])
                padd2 = 2**np.ceil(np.log2(data.shape[0] * 4))
                paddT = np.max(time) * padd2 / data.shape[0]
                dH = 2 * np.pi / paddT / energy
                if np.max(data) > 1e-19:
                    data = np.absolute(
                        np.fft.fft(
                            np.lib.pad(
                                data, (int(np.floor((padd2 - data.shape[0]) / 2)),
                                    int(np.ceil((padd2 - data.shape[0]) / 2))),
                                'constant',
                                constant_values=(0.0, 0.0))))
                    data /= data.max()
                    HHG_spec = data
                    Harmonic = np.arange(data.shape[0]) * dH

    np.savetxt("Harmonic.txt", Harmonic)
    np.savetxt("HHG_Spectrum.txt", HHG_spec)    

def shifted_energy(TDSE_file, pulse_idx=0):
    # central frequency of A field (hbar=1)
    energy = TDSE_file["Parameters"]["energy"][pulse_idx]
    # index of pulse shape, sin=0, gauss=1
    pulse_shape_idx = TDSE_file["Parameters"]["pulse_shape_idx"][pulse_idx]
    # power sin like pulses are raised to
    power_on = TDSE_file["Parameters"]["power_on"][pulse_idx]
    power_off = TDSE_file["Parameters"]["power_off"][pulse_idx]
    cycles_on = TDSE_file["Parameters"]["cycles_on"][pulse_idx]
    cycles_off = TDSE_file["Parameters"]["cycles_off"][pulse_idx]
    cycles_plateau = TDSE_file["Parameters"]["cycles_plateau"][pulse_idx]
    mu = None
    if pulse_shape_idx == 0 and power_on == 2 and power_off == 2 and np.abs(
            cycles_plateau) < 1e-10 and cycles_off == cycles_on:
        mu = 4.0 * (np.arcsin(np.exp(-1.0 / 4.0)))**2
    elif pulse_shape_idx == 1 and np.abs(
            cycles_plateau) < 1e-10 and cycles_off == cycles_on:
        mu = 4 * 2 * np.log(2.0) / np.pi**2
    else:
        print("WARNING: using uncorrected energy.")
        print("         The frequency shift is not implemented for this pulse shape.")
        return energy
    shift = (1.0 + np.sqrt(1 + mu / (cycles_off + cycles_on)**2)) / 2.0
    print("SHIFT:", energy, shift * energy)
    return energy * shift

def Ionization_Rate(Populations):
    Ion_Rate = []
    for Pop in Populations:
        ion_rate = 0.0
        for k in Pop.keys():
            ion_rate += Pop[k]

        Ion_Rate.append((1.00 - ion_rate)*100)

    return Ion_Rate

def File_Organizer(command_line_arg):
    num_of_files = int(command_line_arg[1])
    file_names = []
    TDSE_files = []
    Target_files = []
    Pulse_files = []
    T_P_WFs = []
    F_F_WFs = []
    Populations = []
    n_l_pops = []
    l_m_pops = []
    n_m_pops = []
    n_l_pop_fixed_ms = []
    n_m_pop_fixed_ls = []
    l_m_pop_fixed_ns = []

    for i in range(2, 2 + num_of_files):
        file_names.append(command_line_arg[i])

    for file in file_names:
        TDSE_files.append(h5py.File(file + "/TDSE.h5"))
        Target_files.append(h5py.File(file + "/Hydrogen.h5"))
        Pulse_files.append(h5py.File(file + "/Pulse.h5"))

    for TDSE, Target in zip(TDSE_files, Target_files):
        T_P_WFs.append(Time_Propagated_Wavefunction_Reader(TDSE))
        F_F_WFs.append(Field_Free_Wavefunction_Reader(Target))

    for TDSE, Target, T_P_WF, F_F_WF in zip(TDSE_files, Target_files, T_P_WFs, F_F_WFs):
        Population, n_l_pop, l_m_pop, n_m_pop, n_l_pop_fixed_m, n_m_pop_fixed_l, l_m_pop_fixed_n = Population_Calculator(TDSE, Target, T_P_WF, F_F_WF)
        Populations.append(Population)
        n_l_pops.append(n_l_pop)
        l_m_pops.append(l_m_pop)
        n_m_pops.append(n_m_pop)
        n_l_pop_fixed_ms.append(n_l_pop_fixed_m)
        n_m_pop_fixed_ls.append(n_m_pop_fixed_l)
        l_m_pop_fixed_ns.append(l_m_pop_fixed_n)
    return(Populations, n_l_pops, l_m_pops, n_m_pops, n_l_pop_fixed_ms, n_m_pop_fixed_ls, l_m_pop_fixed_ns, TDSE_files, Target_files, Pulse_files, file_names)

def Resonance_Condition(N_Photon, Energy, w=0.057):
    I = Symbol('I')
    I_unit = 3.5*pow(10, 16)
    Intensities = []
    E = Energy
    
    for N in N_Photon:
        soln = solve(I - 4*pow(w, 2.0)*(N*w - E), I)
        soln = soln[0] * I_unit
        soln = '%.5E' % Decimal(str(soln))
        Intensities.append((round(N, 2), soln))
    return Intensities

def Ionization_Rate_Plotter(Ion_Rate, Intensities, Resonances):
    plt.loglog(Intensities, Ion_Rate, label="Ion_Rate")
    colors = ['r', 'g', 'b']
    for c, Res in enumerate(Resonances):
        for i in Res:
            plt.axvline(x=float(i[1]), c=colors[c])
    plt.xlim(Intensities[0] * 0.975, Intensities[-1]*1.01)
    plt.text(Intensities[-1] * 0.25, pow(10, 0), "Vertical lines show resonant intensities \n n=2 red  n=3 green  I_pot =  blue")
    plt.savefig("Ionization_Rate.png")
    plt.show()

def State_Population(Populations, Intensities, States):
    state_pop = []
    for Pop in Populations:
        total = 0.0
        for k in Pop.keys():
            if (k[0] in States[0] and k[1] in States[1]):
                total += Pop[k]
        if 1 in States[0]:
            state_pop.append(1.0 - total)
        else:
            state_pop.append(total)
    if(len(States[0]) == 1):
        label = "n = " + str(States[0][0]) + " Population" 
    else:
        label = "n = " + str(States[0][0]) + " to " + str(States[0][-1])
    
    plt.semilogy(Intensities, state_pop, label=label)
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1.15), fancybox=True, framealpha=0.5)
    plt.savefig("State_Population.png")
    plt.show()

    # return state_pop 
