import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt

import safe 
import scm
import muller_rootfinding


carte_safe = False
spectral = True
semi_fe = True
rootfinding = False

export_data = False

freqs = np.linspace(1, 7e3, 50)
wavenb = np.linspace(1e-5, 12, 100)

L = 1
ALU = {'name': 'Aluminium', 'rho': 2700, 'cp': 6091, 'cs': 3134}
ALU['ld'] = ALU['rho']*(ALU['cp']**2 - 2*ALU['cs']**2)
ALU['mu'] = ALU['rho']*ALU['cs']**2

fig, ax = plt.subplots(1, 1, figsize=(9/2.54, 8/2.54), tight_layout=True)


# CARTE DET SAFE #################
if carte_safe:
    mapping = np.zeros((len(freqs), len(wavenb)), dtype=float)
    for ii, f in enumerate(freqs):
        mapping[ii, :] = np.array(safe.determinant(f, wavenb, 15, L, (ALU['ld'], ALU['mu'], ALU['rho'])))
    pcol = ax.pcolormesh(wavenb, freqs/1e3, np.log10(mapping), cmap='plasma_r', rasterized=True)
    fig.colorbar(pcol, ax=ax, label='log10(abs(det(matrice)))')

# SPECTRAL COLLOCATION
if spectral:
    val, vec = scm.lamb(freqs, materials_properties={'elastic': {'h':L, 'M':12, 'params':ALU}})
    freqs_tile = np.tile(freqs, (val.shape[1], 1)).T
    tri_k = np.abs(np.imag(val)) < 3
    ax.scatter(np.real(val[tri_k]), freqs_tile[tri_k]/1e3, c='k', s=10, label='SCM')
    if export_data: np.savetxt('disp_scm.csv', np.vstack((val[tri_k], freqs_tile[tri_k]/1e3)).T)
    # np.savetxt()

# SAFE ###########################
if semi_fe: 
    nb_elem    = 25
    nb_eigvals = 2*(2*nb_elem + 1)
    vals = np.zeros((2*nb_eigvals, len(freqs), 2), dtype=complex)
    for ii, f in enumerate(tqdm(freqs)):
        vals[:, ii, 0] = safe.eigenproblem(f, nb_elem, L, (ALU['ld'], ALU['mu'], ALU['rho']))
        vals[:, ii, 1] = [f]*vals.shape[0]
    tri_k = np.abs(np.imag(vals[:, :, 0])/np.real(vals[:, :, 0])) < 1
    ax.scatter(np.real(vals[:, :, 0][tri_k]), np.real(vals[:, :, 1][tri_k])/1e3, c='m', s=2, label='SAFE')
    if export_data: np.savetxt('disp_safe.csv', np.vstack((np.real(vals[:, :, 0][tri_k]), np.real(vals[:, :, 1][tri_k])/1e3)).T)

# ROOT FINDING ####################
if rootfinding:
    mullerParams = {'dx':1e-1, 'eps1':1e-10, 'eps2':1e-2, 'nbIter':2000, 'deflated':True, 'warnings':False}
    
    roots_sym, frequencies_sym = muller_rootfinding.compute_lamb_dispersion(
        muller_rootfinding.lamb_expr, wavenb_bounds={'realLimits': (0, 15), 'imagLimits':(-5, 5)}, freqs=freqs,
        params=(ALU['cp'], ALU['cs'], L/2, True), mullerParams=mullerParams)
    ax.scatter(np.real(roots_sym), np.real(frequencies_sym)/1e3, s=2, c='r', label='muller')

    roots, frequencies = muller_rootfinding.compute_lamb_dispersion(
        muller_rootfinding.lamb_expr, wavenb_bounds={'realLimits': (0, 15), 'imagLimits':(-5, 5)}, freqs=freqs,
        params=(ALU['cp'], ALU['cs'], L/2, False), mullerParams=mullerParams)
    ax.scatter(np.real(roots), np.real(frequencies)/1e3, s=2, c='r', label='')

    if export_data: np.savetxt('disp_roots.csv', np.vstack((
        np.hstack((roots.flatten(), roots_sym.flatten())),
        np.hstack(((frequencies/1e3).flatten(), (frequencies_sym/1e3).flatten()))
    )).T)


ax.legend(markerscale=3, draggable=True, facecolor='white', edgecolor='k')
ax.set(xlim=(0, np.max(wavenb)), ylim=(0, np.max(freqs)/1e3), xlabel='Wavenumber', ylabel='Frequency')
# plt.savefig('compare_methods.pdf', dpi=200)
plt.show()