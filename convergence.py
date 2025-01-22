import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt

import safe 
import scm
import muller_rootfinding

freq = 4e3
discretization = np.arange(3, 50, 1)

L = 1
ALU = {'name': 'Aluminium', 'rho': 2700, 'cp': 6091, 'cs': 3134}
ALU['ld'] = ALU['rho']*(ALU['cp']**2 - 2*ALU['cs']**2)
ALU['mu'] = ALU['rho']*ALU['cs']**2

fig, ax = plt.subplots(1, 2, figsize=(9/2.54, 8/2.54), tight_layout=True)

mullerParams = {
    'dx':5e-2, 'eps1':1e-10, 'eps2':1e-5, 'nbIter':400, 
    'deflated':True, 'warnings':False
}

k_muller_sym = muller_rootfinding.compute_lamb_dispersion(
    muller_rootfinding.lamb_expr, wavenb_bounds={'realLimits': (0, 15), 'imagLimits':(-5, 5)}, 
    freqs=[freq], params=(ALU['cp'], ALU['cs'], L/2, True), 
    mullerParams=mullerParams)[0][0]
k_muller_antisym = muller_rootfinding.compute_lamb_dispersion(
    muller_rootfinding.lamb_expr, wavenb_bounds={'realLimits': (0, 15), 'imagLimits':(-5, 5)}, 
    freqs=[freq], params=(ALU['cp'], ALU['cs'], L/2, False), 
    mullerParams=mullerParams)[0][0]

k_muller = np.sort(np.unique([*k_muller_sym, *k_muller_antisym]))
print(k_muller, len(k_muller))


convergence = np.zeros((len(discretization), len(k_muller), 2))
for i, elems in enumerate(discretization):
    k_safe = safe.eigenproblem(freq, elems, L=L, params=(ALU['ld'], ALU['mu'], ALU['rho']))
    k_safe = np.sort(k_safe)

    k_scm = scm.lamb(
        [freq], materials_properties={'elastic': {'h':L, 'M':elems, 'params':ALU}}
    )[0][0]
    k_scm = np.sort(k_scm)

    for j, k in enumerate(k_muller):
        idx_sc = np.argmin(np.abs(k - k_scm))
        idx_fe = np.argmin(np.abs(k - k_safe))
        convergence[i, j, :] = [np.abs(k - k_scm[idx_sc]), np.abs(k - k_safe[idx_fe])]

output_dict = {
    'k_muller': k_muller, 'discretization': discretization, 
    'conv_scm':convergence[:, :, 0], 'conv_safe':convergence[:, :, 1]
}

np.save('convergence.npy', output_dict)