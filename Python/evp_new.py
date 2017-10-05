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
#CW = MPI.COMM_WORLD

Lx = 10000000
Ly = 5000000
nx = 128
ny = 128
zeta = 0.9 # control for sponge layer
D = 98. # Depth m
a = 1.157e-7 # time const. 1/s
b = 1.157e-7 # time const. 1/s
g  = .02 #reduced gravity?
Co = 1.4 #speed m/s
beta = 2.28e-11 # beta plane const. 1/m*s
kx = 2*np.pi/Lx
#y_basis = de.Chebyshev('y', 256, interval = (-Ly,Ly), dealias = 3/2)

def growth_rate(Lx, Ly, zeta, D, a, b, g, Co, beta, kx, i):

    # Create Basis and domain

    #x_basis = de.Fourier('x', 128, interval = (0,Lx), dealias = 3/2)

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
    bp.parameters['g'] = g
    bp.parameters['Co'] = Co
    bp.parameters['kx'] = i*2*np.pi/Lx
    bp.parameters['i'] = i
    bp.substitutions['theta'] = "pi*y/Ly"
    bp.substitutions['Y(y)'] = "Ly/pi *((1+zeta)/zeta * arctan(zeta*sin(theta)/(1+zeta*cos(theta))))"
    bp.substitutions['Z(y)'] = "(1-zeta)**2/2 * (1-cos(theta))/(1+zeta**2+2*zeta*cos(theta))"
    bp.substitutions['f'] = "beta*y"
    bp.substitutions['vol_avg(A)'] = 'integ(A)/(Lx*Ly)'
    bp.substitutions['dx(A)'] = "1j*kx*A"
    

    #equations
    bp.add_equation("sigma*h + D*(dx(u)+dy(v)) + b*h = 0")
    bp.add_equation("sigma*u + a*u + g*dx(h) - f*v = 0")
    bp.add_equation("sigma*v + a*v + g*dy(h) + f*u = 0")

    #init cond
    bp.add_bc("left(v) = 0")
    bp.add_bc("right(v) = 0")
    # solver

    EVP = bp.build_solver()
    pencils = EVP.pencils
    EVP.solve(EVP.pencils[0])
    logger.info('Solver built')
 
    return EVP

#end of growth_rate definition

#create wave numbers for function to run over
kx_global = np.linspace(1, 6, 6)

#Running function over wavenumbers
freq = np.array([growth_rate(Lx, Ly, zeta, D, a, b, g, Co, beta, kx, i) for i in kx_global])
'''
# largest finite imaginary part
ev = freq[0].eigenvalues
ev = ev[np.isfinite(ev)]
gamma = ev.real
omega = ev.imag
'''
for k in range(len(freq)):
    ev = freq[k].eigenvalues
    ev = ev[np.isfinite(ev)]
    gamma = ev.real
    omega = ev.imag
    #logger.info(gamma[0])
    for j in range(0,len(omega)):
        #logger.info(ev[j])
        plt.scatter(kx_global[k],omega[j])
        plt.ylim(-2e-3,2e-3)
        plt.xlabel("x")
        plt.ylabel("evalues")
        plt.title("Eigenvalues")
plt.savefig('evalues.png')



evp = growth_rate(Lx, Ly, zeta, D, a, b, g, Co, beta, kx, 1)

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
for i in range(2):
    evp.set_state(i)
    plt.plot(yy, evp.state['u']['g'].real, label = "mode {}".format(i))
             
plt.xlabel("y")
plt.ylabel("u")
plt.legend(loc="upper left").draw_frame(False)
plt.title("Eigenmodes")
plt.savefig('emodes.png')   

plt.clf()
'''
plt.plot(yy,evp.eigenvalues.real)
plt.xlabel("x")
plt.ylabel("evalues")
plt.title("Eigenvalues")
plt.savefig('evalues.png')
'''
#code below needs to be fixed for new growth_rate function
'''
# largest finite imaginary part
ev = evp.eigenvalues
ev = ev[np.isfinite(ev)]
gamma = ev.real
omega = ev.imag

#separating out real and imaginary parts of frequency
omega = []
gamma = []
for k in range(len(freq)):
    omega.append(freq[k][0])
    gamma.append(freq[k][1])

max_imag = []
for l in range(len(omega)):
    max_imag.append(np.max(omega[l]))
    
ax = plt.gca()

#plots only positive y values
for j in range(0,len(kx_global)):      
    for k in range(0,len(omega[0])):
        if omega[j][k] >= 0:
            plt.scatter(kx_global[j], omega[j][k])
            plt.ylim(-2e-3,2e-3)
    plt.title('Positive')
    ax.yaxis.labelpad = -5
    plt.xlabel('Wave Number (kx)')
    plt.ylabel('Frequency',labelpad = -5)
plt.savefig('positive.png')
plt.clf()

#plots every nth y value
def positive_freq(interval):
    save_info = 'positive_' + str(interval)
    for j in range(0,len(kx_global)):
        omega_1 = (omega[j])
        positive_omega = []
        for k in range(0,len(omega_1)):
            if omega_1[k] >= 0:
                positive_omega.append(omega_1[k])
                positive_x = positive_omega[::interval]
                for l in range(0,len(positive_x)):
                    plt.scatter(kx_global[j], positive_x[l])
                    plt.ylim(-2e-3,2e-3)
    ax.yaxis.labelpad = -5
    plt.title('Every '+str(interval)+' positive')
    plt.xlabel('Wave Number (kx)')
    plt.ylabel('Frequency',labelpad=-5)
    plt.savefig(save_info)
    plt.clf()
    
#creating the plots for every nth y value (only positive)
positive_freq(5)
positive_freq(10)
positive_freq(20)
positive_freq(100)

#function that plots every nth value whether it is positive or negative
def every_x(interval):
    save_info_1 = 'Every_' + str(interval)
    for j in range(0,len(kx_global)):
    
        every_freq = (omega[j])
        every_int = every_freq[::interval]
        #print(every_10)
        for k in range(0,len(every_int)):
            plt.scatter(kx_global[j], every_int[k])
            plt.ylim(-2e-3,2e-3)
    ax.yaxis.labelpad = -5
    plt.title('Every '+str(interval))
    plt.xlabel('Wave Number (kx)')
    plt.ylabel('Frequency',labelpad = -5)
    plt.savefig(save_info_1)
    plt.clf()

#creating plots for every nth y value
every_x(15)
every_x(5)
every_x(20)

#creating plots of real and imaginary frequencies
for j in range(0,len(kx_global)):      
    for k in range(0,len(omega[0])):
        plt.scatter(kx_global[j], omega[j][k])
        #plt.ylim(-2e-3,2e-3)
    ax.yaxis.labelpad = -5
    plt.title('kx vs Frequency')
    plt.xlabel('Wave Number (kx)')
    plt.ylabel('Frequency',labelpad = -5)
plt.savefig('imag_freq.png')
plt.clf()


for j in range(0,len(kx_global)):      
    for k in range(0,len(gamma[0])):
        plt.scatter(kx_global[j], gamma[j][k])
        #plt.ylim(-2e-3,2e-3)
    ax.yaxis.labelpad = -5
    plt.title('kx vs Frequency')
    plt.xlabel('Wave Number (kx)')
    plt.ylabel('Frequency',labelpad=-5)
plt.savefig('real_freq.png')
plt.clf()

plt.scatter(kx_global, max_imag)
#plt.ylim(1e-4,1.2e-4)
plt.title('kx vs Frequency')
plt.xlabel('Wave Number (kx)')
plt.ylabel('Frequency')
plt.savefig('max_imag.png')


'''
