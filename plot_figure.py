import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
from phd_lib.plotting import matplot_header

cycle = matplot_header(fontsize=12, colors='ibm')

bleu  = cycle[-1]
rouge = cycle[1]
jaune = cycle[0]
fig, ax = plt.subplots(1, 2, figsize=(18/2.54, 8/2.54))

disp = np.loadtxt('disp_roots.csv', dtype=complex)
ax[0].scatter(np.real(disp[:, 0]), np.real(disp[:, 1]), c='k', s=.5, label='analytical')
disp = np.loadtxt('disp_scm.csv', dtype=complex)
ax[0].scatter(np.real(disp[:, 0]), np.real(disp[:, 1]), c=rouge, marker='o', s=10, label='SCM')
disp = np.loadtxt('disp_safe.csv', dtype=complex)
ax[0].scatter(np.real(disp[:, 0]), np.real(disp[:, 1]), s=5, marker='o', facecolors='none', edgecolors=bleu, label='SAFE')

ax[0].set(xlim=(0, 8), ylim=(0, 5), xlabel=r'Wavenumber Re$(k_1)$ (rad.m$^{-1}$)', ylabel=r'Frequency $f$ (kHz)')

ax[0].plot([0, 10], [4, 4], 'k--')

conv = np.load('convergence.npy', allow_pickle=True).item()
print(conv.keys())
cmap = plt.get_cmap('viridis', len(conv['k_muller']))
for j in range(len(conv['k_muller'])):
    ax[1].semilogy(conv['discretization'], conv['conv_scm'][:, j], '-', color=rouge)
    ax[1].semilogy(conv['discretization'], conv['conv_safe'][:, j], '-', color=bleu)

ax[1].set(
    ylim=(1e-12, 1),
    xlim=(np.min(conv['discretization']), np.max(conv['discretization'])), 
    xlabel="Discretization $n$", ylabel='Absolute error / analytics')


h, l = ax[0].get_legend_handles_labels()
plt.legend(handles=h+\
[
    Line2D([0], [0], label='SCM', color='k', ls='-'), 
    Line2D([0], [0], label='SAFE', color='k', ls='--')
], 
ncol=5, draggable=True, facecolor='white', edgecolor='k', 
columnspacing=0.4, markerscale=2
)

plt.subplots_adjust(
top=0.865,
bottom=0.15,
left=0.065,
right=0.97,
hspace=0.29,
wspace=0.38)
plt.show()