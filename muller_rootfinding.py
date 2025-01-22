from numpy import sqrt, exp, cos, sin
import numpy as np 
from scipy.linalg import det
from tqdm import tqdm

def muller(
        f, args, rootSearch, dx=1, eps1=1e-5, 
        eps2=1e-8, nbIter=100, deflated=False, warnings=False, verbose=False):
    
    """
    Parameters
    --------
    f : function(x, args)
        complex-valued function to search roots of z on
    args : tuple (N, )
        additional arguments to use with the function
    rootSearch : dict
        Specify the different modes for root searching. 
        - rootSearch = {
            'mode': 'grid',
            'realAxisResolution': int,
            'imagAxisResolution': int,
            'realAxisBounds': tuple size 2,
            'imagAxisBounds': tuple size 2,
            'previousRoots': list, optional
        };
        - rootSearch = {
            'mode': 'guesses',
            'guesses': list
        }
    dx : float, optional
        calculation step. Muller method uses 3 points to perform 
        each iteration. The initial points are xr-dx, xr, xr+dx.
    eps1 : float 
        Stopping criterion
    eps2 : float 
        Tolerance for 2 close located roots
    nbIter : int, optional 
        max number of iterations to be performed
    
    Returns
    -------
    roots : array(dtype=complex)
        computed roots
    iters : ndarray 
        array with same shape as roots, iteration number for each converged root
    """
    
    
    def mullerIteration(p0, p1, p2, prevRoots, deflated=False, warnings=False):
        """
        Parameters
        ----------
        p0 : complex
            initial point 1 of the search
        p1 : complex
            initial point 2 of the search
        p2 : complex
            initial point 3 of the search
        prevRoots: array of complex
            previously found roots in order to compute the deflated function
        deflated : boolean, optional
            compute using deflated function fp or function f. Any way, 
            fp3 is always computed as it's a convergence criterion
        conv_warn : boolean, optional
            print a warning in the output is the root does not converge.
            Moslty for debugging purposes

        Returns
        -------
        p3 : complex
            the computed root value. If the method doesnt converge, 
            the last iteration is returned. It can be checked, since 
            in this case, i > nbIter
        """

        conv = False
        if verbose:
            print('pi', p0, p1, p2)
        f0, f1, f2 =  f(p0, args),  f(p1, args),  f(p2, args)
        # print('f', f0, f1, f2)
        abs_var = []

        for i in range(1,  nbIter+1):
            # Compute the coefficients of the classical Muller algorithm
            a = ((p1 - p2)*(f0 - f2) - (p0 - p2)*(f1 - f2))/((p0 - p2)*(p1 - p2)*(p0 - p1))
            b = ((p0 - p2)**2 *(f1 - f2)-(p1 - p2)**2*(f0 - f2))/((p0 - p2)*(p1 - p2)*(p0 - p1))
            c = f2

            p3 = p2 - 2*c/(b + np.sign(b)*np.sqrt(b**2 - 4*a*c))

            # abs_var.append(np.abs(p3 - p2))
            # Convergence criterias
            if np.abs((p3 - p2)/p3) < eps1:
                conv = True
                break 

            elif np.isnan(p3):
                # Keep the loop from making function calls until the end of the loop 
                # if the guess was computed as None 
                conv = False
                break       

            elif np.var([p0, p1, p2, p3]) > 1e6:
                # Test if the path will diverge
                conv = False
                break       

            else:
                f3 = f(p3, args)

            
            # Swap values for the next iteration
            p0, p1, p2 = p1, p2, p3
            # print('p', p0, p1, p2)
            f0, f1, f2 = f(p0, args), f(p1, args), f(p2, args)
            # print('f', f0, f1, f2)

            if deflated: 
                f0 /= np.prod(p0 - prevRoots)
                f1 /= np.prod(p1 - prevRoots)
                f2 /= np.prod(p2 - prevRoots)
        
        # # ax.semilogy(np.abs(abs_var))
        # if warnings: print(f"no root found after {nbIter} it., f(p3={p3})={f3}")

        return p3, i, conv

    def validRoot(root, roots, conv):
        return all((
            conv == True,
            np.min(np.abs(root - roots)) > eps2,
            (np.real(root) > rootSearch['interval'][0][0]),
            (np.real(root) < rootSearch['interval'][0][1]),
            (np.imag(root) > rootSearch['interval'][1][0]),
            (np.imag(root) < rootSearch['interval'][1][1])
        ))
    
    r = 0

    if rootSearch['mode'] == 'guesses':
        initial_points = rootSearch['guesses']

    elif rootSearch['mode'] == 'grid':
        KR, KI = np.meshgrid(
            np.linspace(*rootSearch['realAxisBounds'], rootSearch['realAxisResolution']),
            np.linspace(*rootSearch['imagAxisBounds'], rootSearch['imagAxisResolution'])
        )

        initial_points = (KR + 1j*KI).ravel()
        
        try: 
            initial_points = np.array([*rootSearch['previousRoots'], *initial_points])
        except KeyError:
            if warnings:
                print("! The 'previousRoots' key is empty")
            pass

    else: 
        raise ValueError('The rootSearch mode is invalid')


    roots, iters = [0+0j], [0]
    if verbose:
        print(f"Muller \t: iters./(total found roots)/total iters.")
    for jj, xr in enumerate(initial_points): 
        if verbose:
            print(f"Muller \t: {jj}({len(roots)})/{len(initial_points)}")
        root, it, conv = mullerIteration(xr - dx, xr, xr + dx, roots[:r], deflated=deflated, warnings=warnings)
        if validRoot(root, roots, conv) and r < len(initial_points):
            roots.append(root)
            iters.append(it)

    return roots[1:], iters[1:]

def local_minima(grid, det_map, minimas=True, maximas=False):
    if minimas:
        mapmin = (
            (det_map <= np.roll(det_map,  1, 0)) &
            (det_map <= np.roll(det_map, -1, 0)) &
            (det_map <= np.roll(det_map,  1, 1)) &
            (det_map <= np.roll(det_map, -1, 1))
        )
    if maximas:
        mapmax = (
            (det_map >= np.roll(det_map,  1, 0)) &
            (det_map >= np.roll(det_map, -1, 0)) &
            (det_map >= np.roll(det_map,  1, 1)) &
            (det_map >= np.roll(det_map, -1, 1))
        )

    if minimas and not maximas: return grid[mapmin]
    if minimas and maximas:     return grid[mapmin], grid[mapmax]
    if maximas and not minimas: return grid[mapmax]

def lamb_det(k, params, d=True):
    om, cp, cs, H = params 
    kl = sqrt(om**2/cp**2 - k**2);  kt = sqrt(om**2/cs**2 - k**2)
    expL = exp(-1j*kl*H);       expT = exp(-1j*kt*H)
    rl = 2*k*kl;  rt = 2*k*kt;     s = k**2 - kt**2 # Equations /mu
    m = np.array([
        [-rl*expL, rl,      s*expT,   s       ],
        [-rl,      rl*expL, s,        s*expT  ],
        [ s*expL,  s,       rt*expT, -rt      ],
        [ s,       s*expL,  rt,      -rt*expT ]
    ])
    if d: return det(m)
    else: return m


def lamb_expr(k, params):
    om, cp, cs, H, sym = params 
    kl = sqrt(om**2/cp**2 - k**2);  kt = sqrt(om**2/cs**2 - k**2)

    if sym:
        expr = np.tan(kt*H)/np.tan(kl*H) + (4*k**2*kt*kl)/(kt**2 - k**2)**2
    else:
        expr = np.tan(kt*H)/np.tan(kl*H) + (kt**2 - k**2)**2/(4*k**2*kt*kl)
    return expr

def compute_map(func, kr, ki, params):

    det_matrix = np.zeros((len(kr), len(ki)), dtype=complex)
    for i, ikr in enumerate((kr)):
        for j, iki in enumerate(ki):
            k = ikr + 1j*iki
            det_matrix[i, j] = func(k, params)

    return det_matrix

def list_of_lists_to_array(liste, pad_value=None, dtype=float):
    return np.array([[*i, *[pad_value]*(len(max(liste, key=len)) \
                - len(i))] for i in list(liste)], dtype=dtype)

def compute_lamb_dispersion(func, wavenb_bounds, freqs, params, mullerParams):
    previous_roots, roots, frequencies, wavenumbers = [], [], [], []
    rootSearch = {'mode':'guesses', 'interval': (), 'guesses': []}
    for ii, f in enumerate(tqdm(freqs[::-1])):
        om = 2*np.pi*f 

        kr = np.linspace(*wavenb_bounds['realLimits'], 200)
        ki = np.linspace(*wavenb_bounds['imagLimits'], 100)
        KR, KI = np.meshgrid(kr, ki)
        grid = {'kr': kr, 'ki': ki}
        rootSearch['interval'] = ((grid['kr'][0], grid['kr'][-1]), (grid['ki'][0], grid['ki'][-1]))

        maps = compute_map(lamb_expr, **grid, params=(om, *params))
        KR, KI = np.meshgrid(grid['kr'], grid['ki'])
        mins = local_minima(KR + 1j*KI, np.abs(maps.T), minimas=True)
        
        rootSearch['guesses'] = [*previous_roots, *mins]
        root, iters = muller(
            lamb_expr, args=(om, *params), 
            rootSearch=rootSearch, verbose=False,
            **mullerParams
        )

        previous_roots = root
        roots.append(root)
        frequencies.append([f]*len(root))

    return list_of_lists_to_array(roots, dtype=complex), list_of_lists_to_array(frequencies, dtype=complex)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    H = 1
    freqs  = np.linspace(1, 5e3, 100)
    ALU = {'name': 'Aluminium', 'rho': 2700, 'cp': 6091, 'cs': 3134}
    sym = True
    mullerParams = {'dx':1+1j, 'eps1':1e-5, 'eps2':1e-5, 'nbIter':800, 'deflated':False, 'warnings':False}
    # ALU['ld'] = ALU['rho']*(ALU['cp']**2 - 2*ALU['cs']**2)
    # ALU['mu'] = ALU['rho']*ALU['cs']**2
    roots, frequencies = compute_lamb_dispersion(
        lamb_det, wavenb_bounds={'realLimits': (0, 5), 'imagLimits':(-3, 3)}, freqs=freqs,
        params=(ALU['cp'], ALU['cs'], H, sym), mullerParams=mullerParams)

    fig, ax = plt.subplots()
    ax.scatter(np.real(roots), frequencies, s=2, c='k')

    plt.show()
