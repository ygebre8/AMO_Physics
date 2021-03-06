import numpy as np
from sympy.solvers import solve
from sympy import Symbol
from decimal import Decimal
import os
import json

#07.025', '7.74818E+14
### Define intensity
I = Symbol('I')
I_unit = 3.5*pow(10, 16)
Intensities = []
Intensities_old = []
### Range for the number of photon process
resolution = 0.025
N_photon = np.arange(5.5, 10.2  + 0 * resolution, resolution)
print(N_photon)
N_photon_old = np.arange(6, 11+ 0.1, 0.1)
Energy = [0.499995, 0.44443945, 0.374995] ## Ip, n=3 or n=2 in resonance


# ### Energy and frequency used for the calculation
E = Energy[2]
w = 0.07015384617

for N in N_photon:
    soln = solve(I - 4*pow(w, 2.0)*(N*w - E), I)
    soln = soln[0] * I_unit
    soln = '%.5E' % Decimal(str(soln))
    num_pho = "%.3f" % N 
    Intensities.append((num_pho.zfill(6) , soln))



input_par = {
 "alpha": 0.0,
 "coordinate_system": "Spherical",
 "delta_t": 0.025,
 "dimensions": [
  {
   "delta_x_max": 0.1,
   "delta_x_max_start": 4.0,
   "delta_x_min": 0.1,
   "delta_x_min_end": 4.0,
   "dim_size": 750.0,
   "l_max": 135,
   "m_max": 0
  }
 ],
 "ee_soft_core": 0.01,
 "field_max_states": 0,
 "free_propagate": 0,
 "gauge": "Length",
 "gobbler": 0.95,
 "laser": {
  "experiment_type": "default",
  "frequency_shift": 1,   
  "pulses": [
   {
    "cep": 0.0,
    "cycles_delay": 0.0,
    "cycles_off": 5.0,
    "cycles_on": 5.0,
    "cycles_plateau": 0.0,
    "ellipticity": 0.0,
    "energy": 0.07015384617,
    "gaussian_length": 5.0,
    "helicity": "left",
    "intensity": 1.0e13,
    "polarization_vector": [
     0.0,
     0.0,
     1.0
    ],
    "power_off": 2.0,
    "power_on": 2.0,
    "poynting_vector": [
     1.0,
     0.0,
     0.0
    ],
    "pulse_shape": "sin"
   }
  ]
 },
 "num_electrons": 1,
 "order": 4,
 "propagate": 1,
 "restart": 0,
 "sigma": 3.0,
 "start_state": {
  "amplitude": [
   1.0
  ],
  "l_index": [
   0
  ],
  "m_index": [
   0
  ],
  "n_index": [
   1
  ],
  "phase": [
   0.0
  ]
 },
 "state_solver": "File",
 "states": 14,
 "target": {
  "name": "Hydrogen",
  "nuclei": [
   {
    "exponential_amplitude": [
     0.0
    ],
    "exponential_decay_rate": [
     0.0
    ],
    "exponential_r_0": [
     0.0
    ],
    "gaussian_amplitude": [
     0.0
    ],
    "gaussian_decay_rate": [
     0.0
    ],
    "gaussian_r_0": [
     0.0
    ],
    "location": [
     0.0,
     0.0,
     0.0
    ],
    "square_well_amplitude": [
     0.0
    ],
    "square_well_r_0": [
     0.0
    ],
    "square_well_width": [
     0.0
    ],
    "yukawa_amplitude": [
     0.0
    ],
    "yukawa_decay_rate": [
     0.0
    ],
    "yukawa_r_0": [
     0.0
    ],
    "z": 1.0
   }
  ]
 },
 "tol": 1e-10,
 "write_frequency_checkpoint": 20000,
 "write_frequency_eigin_state": 100,
 "write_frequency_observables": 1
}

count = 0
for i in Intensities:
    if i in Intensities_old:
        #continue
        s = 1
    fold = "Run_" + i[0] + "_" + i[1]
    os.chdir('/home/becker/yoge8051/AMO_Physics/Intensity_Sweep/')
    os.mkdir(fold)
    os.chdir('/home/becker/yoge8051/AMO_Physics/Intensity_Sweep/' + fold)
    input_par["laser"]["pulses"][0]["intensity"] = float(i[1])
    with open("input.json", 'w') as f:
        f.write(json.dumps(input_par, indent=2))
        os.system('cp /home/becker/yoge8051/Hydrogen.h5 .')
    
    # count += 1    
    # if count == 10:    
    #     break     
        
