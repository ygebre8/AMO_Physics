import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import h5py

def Time_Propagated_Wavefunction_Reader(TDSE_file):
	T_P_Wavefunction = {}

	time = np.array(TDSE_file['Wavefunction']['time'])
	psi = TDSE_file['Wavefunction']['psi'][time.size - 1]
	psi = psi[:,0] + 1.0J*psi[:,1] 

	l_values = np.array(TDSE_file['Wavefunction']['l_values'])
	m_values = np.array(TDSE_file['Wavefunction']['m_values'])
	r_ind_max = TDSE_file['Wavefunction']['x_value_2'].size
	r_ind_lower = 0
	r_ind_upper = r_ind_max 

	for i in range(len(l_values)):
		T_P_Wavefunction[(l_values[i], m_values[i])] = np.array(psi[r_ind_lower: r_ind_upper])
		r_ind_lower = r_ind_upper
		r_ind_upper = r_ind_upper + r_ind_max

	return T_P_Wavefunction

def Field_Free_Wavefunction_Reader(Target_file):
	F_F_Wavefunction = {} 

	n_max = int(Target_file['Energy_l_0'].size / 2)
	n_values = np.arange(1, n_max + 1)

	for i in range(n_max):
		group_name = "psi_l_" + str(i)
		for k in range(n_max - i):
			# print(group_name, i, k)
			F_F_Wavefunction[(k + 1 + i, i)] = Target_file[group_name]['psi'][k]
			F_F_Wavefunction[(k + 1 + i, i)] = np.array(F_F_Wavefunction[(k + 1 + i, i)][:,0] + 1.0J*F_F_Wavefunction[(k + 1 + i, i)][:,1])
	
	
	return(F_F_Wavefunction)

def Fortran_Wavefunction_Reader(Target_file):
	F_F_Wavefunction = {}
	n_max = Target_file['Energy_l0'].size
	n_values = np.arange(1, n_max + 1)
	
	Wavefunction_size = Target_file['Psi_l0'][0].size

	for l in range(n_max):
		Dataset_name = "Psi_l" + str(l)
		current_l_value_array = Target_file[Dataset_name]
		for n in range(n_max  - l):
			F_F_Wavefunction[(n + 1 + l, l)] = current_l_value_array[n]

	# for i in F_F_Wavefunction.keys():
	# 	for j in F_F_Wavefunction.keys():
	# 		if(i[1] == j[1] and i[0] < 3):
	# 			print(np.vdot(F_F_Wavefunction[i], F_F_Wavefunction[j]))


	return(F_F_Wavefunction)

def Population_Calculator(TDSE_file, Target_file, T_P_Wavefunction, F_F_Wavefunction):

	Population = {}
	l_m_population = {}
	n_l_population = {}
	specific_n_population = []	

	l_values = np.array(TDSE_file['Wavefunction']['l_values']) 
	m_values = np.array(TDSE_file['Wavefunction']['m_values'])
	n_max = int(Target_file['Energy_l_0'].size / 2)
	n_values = np.arange(1, n_max + 1)

	print(n_max)
	for i in range(n_max + 1):
		specific_n_population.append({})
		for l, m  in zip(l_values, m_values):
			(specific_n_population[i])[l, m] = pow(10.0, -20)

	for l, m  in zip(l_values, m_values):
		l_m_population[l, m] = pow(10.0, -20)

	for n in n_values:
		for l in l_values:
			n_l_population[n, l] = pow(10.0, -20)



	for l, m  in zip(l_values, m_values):
		for n in range(l + 1, n_max + 1):
			Population[(n, l, m)] = np.vdot(F_F_Wavefunction[(n, l)], T_P_Wavefunction[(l, m)])
			Population[(n, l, m)] = np.power(np.absolute(Population[(n, l, m)]),2.0) + pow(10.0, -20)
	
			n_l_population[n, l] = n_l_population[n, l] + Population[(n,l,m)]
			l_m_population[l, m] = l_m_population[l, m] + Population[(n,l,m)]
			(specific_n_population[n])[l,m] = (specific_n_population[n])[l,m] + Population[(n,l,m)]
			
	
	for i in Population.keys():
		Population[i] = np.log10(Population[i])
		if(Population[i] <= -15):
			Population[i] = None
	for j in l_m_population.keys():
		l_m_population[j] = np.log10(l_m_population[j])
		# if(l_m_population[j] <= -15):
		# 	l_m_population[j] = None
	for j in n_l_population.keys():
		n_l_population[j] = np.log10(n_l_population[j])
		if(n_l_population[j] <= -21):
			n_l_population[j] = None
	for i in range(n_max + 1):
		for j in specific_n_population[i].keys():
			specific_n_population[i][j] = np.log10(specific_n_population[i][j])	
			if(specific_n_population[i][j] <= -15):
				specific_n_population[i][j] = None
	
	return (Population, n_l_population, l_m_population, specific_n_population)

def N_L_Population_Plotter(n_l_population, TDSE_file, Target_file, file_name = "N_L_Population.png"):
	
	l_values = np.array(TDSE_file['Wavefunction']['l_values']) 
	l_max = np.amax(l_values) + 1
	n_max = int(Target_file['Energy_l_0'].size / 2)
	n_values = np.arange(1, n_max + 1)
	heat_map_array = np.zeros(shape=(n_max + 1,l_max))

	for n in n_values:
		for l  in l_values:
			heat_map_array[n][l] = n_l_population[n, l]
			if(heat_map_array[n][l] <= -21):
				heat_map_array[n][l] = None

	n_l_max = -50.0
	n_l_second_max = -50.0
	for i in n_l_population.keys():
		if(n_l_population[i] !=None and n_l_population[i] > n_l_max):
			n_l_max = n_l_population[i]

	for i in n_l_population.keys():
		if(n_l_population[i] != None and n_l_population[i]> n_l_second_max and n_l_population[i] < n_l_max):
			n_l_second_max = n_l_population[i]

	xticks = np.arange(0, n_max)
	yticks = np.arange(0, n_max)		


	figure = plt.figure()
	plt.xticks(xticks)
	plt.yticks(yticks)
	plt.imshow(heat_map_array)
		
	plt.colorbar()
	# plt.clim(n_l_max)
	plt.clim(n_l_second_max - 5,n_l_second_max)

	print(n_l_max)
	print(n_l_second_max)
	plt.ylim(0.5, n_max + 0.5)
	plt.xlim(-0.5,n_max - 0.5)
	plot_text_one = "Intensity = 2*10^13, Wavelength = 571nm, 10 cycles"
	plot_text_tw0 = "6 photon process"
	# plt.text(1.5, 16, plot_text_one)
	# plt.text(4, 15, plot_text_tw0)
	figure.savefig(file_name)
	plt.show()

def L_M_Population_Plotter(l_m_population, TDSE_file, Target_file, file_name = "L_M_Population.png"):
	n_max = int(Target_file['Energy_l_0'].size / 2)
	n_values = np.arange(1, n_max + 1)
	l_values = np.array(TDSE_file['Wavefunction']['l_values'])
	m_values = np.array(TDSE_file['Wavefunction']['m_values']) 
	l_max = np.amax(l_values) + 1
	m_max = 2 * np.amax(m_values) + 1
	heat_map_array_l_m = np.zeros(shape=(l_max, m_max))

	for i in range(l_max):
		for j in range(m_max):
			heat_map_array_l_m[i][j] = None
	l_m_max = -50.0
	l_m_second_max = -50.0
	for l, m  in zip(l_values, m_values):
		if(l_m_population[l, m] != None and l_m_population[l, m] > l_m_max):
			l_m_max = l_m_population[l, m]

	for l, m  in zip(l_values, m_values):
		if(l_m_population[l, m] != None and l_m_population[l, m] > l_m_second_max and l_m_population[l, m] < l_m_max):
			l_m_second_max = l_m_population[l, m]


	for l, m  in zip(l_values, m_values):
		heat_map_array_l_m[l][m + np.amax(m_values)] = l_m_population[l, m]
		if(heat_map_array_l_m[l][m + np.amax(m_values)] != None and heat_map_array_l_m[l][m + np.amax(m_values)]<= -21):
			heat_map_array_l_m[l][m + np.amax(m_values)] = None

	figure = plt.figure()
	xticks = np.arange(-1 * np.amax(m_values), np.amax(m_values) + 1)
	yticks = np.arange(0, l_max)
	plt.xticks(np.arange(0,m_max), xticks)
	plt.yticks(yticks)

	plt.imshow(heat_map_array_l_m)

	plt.colorbar()
	plt.clim(l_m_second_max - 5,l_m_second_max)
	plt.ylim(-0.5, n_max - 0.5)
	plt.xlim(np.amax(m_values) - n_max, np.amax(m_values) + n_max,)
	plot_text_one = "Intensity = 2*10^13, Wavelength = 3112nm, 10 cycles"
	plot_text_tw0 = "1s - 3s, 3d 2 photon process"
	# plt.text(1.5, 16, plot_text_one)
	# plt.text(2, 15, plot_text_tw0)
	
	figure.savefig(file_name)
	plt.show()

def M_Distribution(l_m_population, TDSE_file, l_value, file_name):
	l_values = np.array(TDSE_file['Wavefunction']['l_values'])
	m_values = np.array(TDSE_file['Wavefunction']['m_values']) 
	m_max = np.amax(m_values)	
	m_dis = {}

	for l in range(np.amax(l_values) + 1):
		m_dis[l] = []
		for m in range(-1 * l, l + 1):
			m_dis[l].append(np.power(10.0, -15))

	
	for l, m in zip(l_values, m_values):
		if(l_m_population[(l,m)] != None):
			m_dis[l][m + l] = m_dis[l][m + l] + np.power(10.0, l_m_population[(l,m)])
		else:
			None
			# print(l,m, "check not populated" )

	print(m_dis[l_value])
	figure = plt.figure()
	plt.plot(range(-1*l_value, l_value + 1), np.log10(m_dis[l_value]), 'o')
	figure.savefig(file_name)

def state_name(state_number, l_max, m_max):
    name_list = []
    state_idx = 0
    n_val = 1
    while state_idx < state_number:
        for l_val in np.arange(0, min(n_val, l_max + 1)):
            m_range = min(l_val, m_max)
            for m_val in np.arange(-m_range, m_range + 1):
                name_list.append("(" + str(n_val) + "," + str(l_val) + "," +
                                 str(m_val) + ")")
                state_idx += 1
        n_val += 1
    return name_list

def population_comparison(TDSE_file, Target_file, Population=None):
	n_max = int(Target_file['Energy_l_0'].size / 2)
	n_values = np.arange(1, n_max + 1)
	l_values = np.array(TDSE_file['Wavefunction']['l_values'])
	m_values = np.array(TDSE_file['Wavefunction']['m_values']) 
	l_max = np.amax(l_values)
	m_max = np.amax(m_values)	
	state_name_list = state_name(len(Population.keys()), l_max, m_max)

	time = np.array(TDSE_file['Wavefunction']['time'])
	TDSE_file_population = TDSE_file['Wavefunction']['projections'][time.size - 1]	
	TDSE_file_population = np.array(TDSE_file_population[:,0] + 1.0J*TDSE_file_population[:,1])
	TDSE_file_population = np.power(np.absolute(TDSE_file_population), 2.0) + np.power(10, -20.0)
	TDSE_file_population = np.log10(TDSE_file_population)

	TDSE_file_population_dictionary = {}
	for i in range(len(state_name_list)):
		TDSE_file_population_dictionary[eval(state_name_list[i])] = TDSE_file_population[i] 
	
	
	for i in Population.keys():
		print(TDSE_file_population_dictionary[i] - Population[i], i)	

def l_pop(TDSE_file, TDSE_file2 = None):
	pop = {}
	f = TDSE_file
	time = np.array(TDSE_file['Wavefunction']['time'])
	psi = f["Wavefunction"]["psi"][time.size - 1]
	shape = f["Wavefunction"]["num_x"][:]
	psi_cooridnate_values = []
	
	for dim_idx in np.arange(shape.shape[0]):
		psi_cooridnate_values.append(f["Wavefunction"]["x_value_" + str(dim_idx)][:])

	l_values = f["/Wavefunction/l_values"][:]
	m_values = f["/Wavefunction/m_values"][:]
	psi = psi[:, 0] + 1j * psi[:, 1]
	
	psi.shape = shape
	psi_norm = np.sqrt((psi * psi.conjugate()).sum())
	for l in np.arange(0, psi_cooridnate_values[1].shape[0]):
		pop[l] = np.log10((np.sqrt((psi[0, l, :] * psi[0, l, :].conjugate()).sum()) / psi_norm).real)
		

	# pop2 = {}
	# f2 = TDSE_file2
	# time2 = np.array(TDSE_file2['Wavefunction']['time'])
	# psi2 = f2["Wavefunction"]["psi"][time.size - 1]
	# shape2 = f2["Wavefunction"]["num_x"][:]
	# psi_cooridnate_values2 = []
	
	# for dim_idx in np.arange(shape2.shape[0]):
	# 	psi_cooridnate_values2.append(f2["Wavefunction"]["x_value_" + str(dim_idx)][:])


	# l_values2 = f2["/Wavefunction/l_values"][:]
	# m_values2 = f2["/Wavefunction/m_values"][:]
	# psi2 = psi2[:, 0] + 1j * psi2[:, 1]
	
	# psi2.shape = shape2
	# psi_norm2 = np.sqrt((psi2 * psi2.conjugate()).sum())
	# for l in np.arange(0, psi_cooridnate_values2[1].shape[0]):
	# 	pop2[l] = np.log10((np.sqrt((psi2[0, l, :] * psi2[0, l, :].conjugate()).sum()) / psi_norm2).real)
	
	# plt.plot(pop2.keys(), pop2.values())
	plt.plot(pop.keys(), pop.values())
	plt.savefig("30vs25 pop.png")
	plt.show()