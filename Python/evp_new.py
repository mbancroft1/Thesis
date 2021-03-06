
import sys
import os
import numpy as np
from mpi4py import MPI
import time
from dedalus.tools.config import config
import time
import matplotlib.pyplot as plt
from docopt import docopt
from dedalus import public as de
from dedalus.extras import flow_tools
import logging
logger = logging.getLogger(__name__)
ax = plt.gca()
#CW = MPI.COMM_WORLD
# Unitless Dimensions of Beta Plane
Lx = 1020408.163
Ly = 510204.0816
# Resolution
nx = 128
ny = 128
zeta = 0.9 # control for sponge layer
# Constants all Unitless
D = 1 # Depth 
T = np.sqrt((2*98)/9.8)
#g  = (9.8*T**2)/(98) 
a = 1.157e-7/T # time const. 
b = 1.157e-7/T # time const.
Co = 1.4*(98/T) #speed
beta = 2.28e-11/(98*T) # beta plane const.
kx = 2*np.pi/Lx

# Function for building Solver with system of equations

def growth_rate(Lx, Ly, zeta, D, a, b, Co, T,  beta, kx, i):

    # Create Basis and domain

    y_basis = de.Chebyshev('y', 128, interval = (-Ly,Ly), dealias = 3/2)

    domain = de.Domain([y_basis], grid_dtype=np.complex128)

    # setting up parameters/equations

    bp = de.EVP(domain,variables=['u','v','h'],eigenvalue='sigma')
    bp.parameters['beta'] = beta
    bp.parameters['zeta'] = zeta
    bp.parameters['Lx'] = Lx
    bp.parameters['Ly'] = Ly
    bp.parameters['pi'] = np.pi
    bp.parameters['a'] = a
    bp.parameters['b'] = b
    bp.parameters['D'] = D
    bp.parameters['Co'] = Co
    bp.parameters['kx'] = i*2*np.pi/Lx
    bp.parameters['i'] = i
    bp.parameters['T'] = np.sqrt((2*98)/9.8)
    bp.substitutions['theta'] = "pi*y/Ly"
    bp.substitutions['Y(y)'] = "Ly/pi *((1+zeta)/zeta * arctan(zeta*sin(theta)/(1+zeta*cos(theta))))"
    bp.substitutions['Z(y)'] = "(1-zeta)**2/2 * (1-cos(theta))/(1+zeta**2+2*zeta*cos(theta))"
    bp.substitutions['f'] = "beta*y"
    bp.substitutions['vol_avg(A)'] = 'integ(A)/(Lx*Ly)'
    bp.substitutions['dx(A)'] = "1j*kx*A"
    

    #equations
    bp.add_equation("sigma*h+dx(u)+dy(v) + b*h = 0")
    bp.add_equation("sigma*u + a*u + 2*dx(h) - f*v = 0")
    bp.add_equation("sigma*v + a*v + 2*dy(h) - f*u = 0")

    #init cond
    bp.add_bc("left(v) = 0")
    bp.add_bc("right(v) = 0")
    # Building solver

    EVP = bp.build_solver()
    pencils = EVP.pencils
    EVP.solve(EVP.pencils[0])
    logger.info('Solver built')
 
    return EVP

#end of growth_rate definition

#create wave numbers for function to run over
kx_global = np.linspace(1, 1, 1)

#Running function over wavenumbers
freq = np.array([growth_rate(Lx, Ly, zeta, D, a, b, Co, T, beta, kx, i) for i in kx_global])
'''
# largest finite imaginary part
ev = freq[0].eigenvalues
ev = ev[np.isfinite(ev)]
gamma = ev.real
omega = ev.imag
'''

#Function to plot e-val unsorted 
def plot_eval(freq):
    for k in range(len(freq)):
        ev = freq[k].eigenvalues
        ev = ev[np.isfinite(ev)]
        # sorting real and imaginary parts of e-vals
        gamma = ev.real
        omega = ev.imag
        #plotting Imaginary parts 
        plt.plot(omega)
        plt.ylim(-2e-3,2e-3)
        ax.yaxis.labelpad = -5
        plt.xlabel("kx")
        plt.ylabel("evalues")
        plt.title("Eigenvalues")
    plt.savefig('evalues.png')
    plt.clf()
# end plot_eval
plot_eval(freq)


#plotting eigenvalues sorted
def plot_eval_sort(freq):
    for k in range(len(freq)):
        finite = np.isfinite(freq[k].eigenvalues)
        freq[k].eigenvalues = freq[k].eigenvalues[finite]
        freq[k].eigenvectors = freq[k].eigenvectors[:,finite]
        order = np.argsort(freq[k].eigenvalues)
        freq[k].eigenvalues = freq[k].eigenvalues[order]
        freq[k].eigenvectors = freq[k].eigenvectors[:, order]
        #logger.info(freq[k].eigenvalues)
        plt.plot(freq[k].eigenvalues)
        #plt.ylim(-2e-3,2e-3)
        ax.yaxis.labelpad = -5
        plt.xlabel("kx")
        plt.ylabel("evalues")
        plt.title("Eigenvalues")
    plt.savefig('evalues_sort.png')
    plt.clf()

plot_eval_sort(freq)



evp = growth_rate(Lx, Ly, zeta, D, a, b, Co, T, beta, kx, 1)

# Filter infinite/nan eigenmodes
finite = np.isfinite(evp.eigenvalues)
evp.eigenvalues = evp.eigenvalues[finite]
evp.eigenvectors = evp.eigenvectors[:, finite]

# Sort eigenmodes by eigenvalue
order = np.argsort(evp.eigenvalues)
evp.eigenvalues = evp.eigenvalues[order]
evp.eigenvectors = evp.eigenvectors[:, order]

#testing set_state
yy = evp.domain.grid(0)
plt.figure()
for i in range(3):
    evp.set_state(i)
    plt.plot(yy, evp.state['u']['g'].real, label = "mode {}".format(i))
             
plt.xlabel("y")
plt.ylabel("u")
plt.legend(loc="upper left").draw_frame(False)
plt.title("Eigenmodes")
plt.savefig('emodes.png')   

plt.clf()




