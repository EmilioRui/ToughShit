import numpy as np
import pandas as pd
from abdb import *
import json

chip_name = 'TAR-0012-01'

var = Variation.query().get(f'{chip_name}')
sim = var.simulations.first()

#remove side junctions and their energies
idx = []
new_je = []
new_jj = []
for i, je in enumerate(sim.data.josephson_elements):
    if '_jct' in je:
        idx.append(i)
    else:
        new_je.append(je)
        new_jj.append(sim.data.josephson_energies[i])

# ZPF
tmp_zpf = []

for j, _zpf in enumerate(sim.data.phi_ZPF):
    tmp_zpf.append([])
    for i, _modezpf in enumerate(_zpf):
        if i not in idx:
            tmp_zpf[j].append(_modezpf)
    # tmp_zpf[j] = np.array(tmp_zpf[j])

zpf = tmp_zpf

# Energies
Ejs = []
Njs = []
for i, ej in enumerate(new_jj):
    if isinstance(ej,TransmonEnergy):
        # if '_jct' not in sim.data.josephson_elements[i]:
        Ejs.append(ej.EJ/1e9)
        Njs.append(1)
    if isinstance(ej,ArrayEnergy):
        Ejs.append(ej.EL/1e9)
        Njs.append(ej.n_array)
    if isinstance(ej,ATSEnergy):
        Ejs.append(ej.EL/1e9)
        Njs.append(ej.n_array)

# Freq
freq_c = np.array(sim.data.frequencies)
chispice = sim.data.chi

freq = []
for i, f in enumerate(freq_c):
    freq.append( (f + np.sum(chispice[i])) * 1e-9 )

data = {"zpf":zpf,"Ejs":Ejs,"Njs":Njs, "freq" : freq}

filename = 'Chip_Data/' + chip_name + '.json'
with open(filename,'w') as f:
    json.dump(f,data,indent=3)
