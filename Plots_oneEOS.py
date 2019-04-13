#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Code to compute various plots given files with different quantities 
of rotating neutron stars, and considering one EOS

"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize = (9,7))

# File to read the data from
data = np.loadtxt('NS_data_eosL.txt', unpack=True)  #only one EOS

# Some constants
G = 6.67e-8
c = 2.99792458e10
Msun = 1

# Extracting columns from the data files
ec = data[0,:]*10**15   # central energy density
M = data[1,:]           # Total mass (in Msun)
M0 = data[2,:]          # Baryonic mass (in MSun)
Mstat = data[3,:]       # Mass when the NS is not rotating (in MSun)
Mmax = data[4,:]        # Maximum mass for a given EOS (in MSun)
R = data[5,:]           # Radius(in km)
Rratio = data[6,:]      # Ratio rp/re
Rstat = data[7,:]       # Radius when the NS is not rotating (in km)
freq = data[8,:]        # Rotational frequency (in Hz)
kfreq = data[9,:]       # Kepler limit (in Hz)
J = data[10,:]          # Angular momentum
T = data[11,:]          # Rotational kinetic energy
W = data[12,:]          # Gravitational binding energy
Rmax = data[13,:]       # Maximum radius of the non rotating star
Qmax = data[14,:]       # MaxMass / MaxRadius of the non rotating star

# Converting previous quantities to CGS
m = M * 1.9884e33   # Mass
r = R * 1.0e5       # Radius
rstat = Rstat * 1.0e5   # Radius of the non rotating NS
mstat = Mstat * 1.9884e33   # Mass of the non rotating NS
delta = freq*(2*np.pi) * ((rstat**3)/(G*mstat))**(0.5)   # Normalized angular velocity
m0 = M0 * 1.9884e33     # Baryonic mass
normJ = (c*J)/(G*m0**2)     # Normalized angular momentum

# Menu of the different plots
print('Choose the plot to generate (Choose only integer numbers)')
print('1. Sequences of Mass vs energy density')
print('2. Mass vs Radius')
print('3. Frequency vs Mass')
print('4. (M-M*)/M* vs frequency')
print('5. (M-M*)/M* vs normalized omega')
print('6. (M-M_0)/M_0 vs frequency')
print('7. (M-M_0)/M_0 vs normalized omega')
print('8. M/Mmax vs normalized omega')
print('9. (R-R*)/R* vs frequency')
print('10. (R-R*)/R* vs normalized omega')
print('11. Compactness vs frequency')
print('12. Compactness vs normalized omega')
print('13. Frequency vs Normalized angular momentum')
print('14. Frequency vs T/W')
print('15. 3D plot of M/Mmax, normalized omega and compactness')
print('16. 3D plot of (R-R*)/R*, normalized omega and compactness')
print('17. 3D plot of (M-M*)/M*, normalized omega and compactness')
print('18. 3D plot of (M-M_0)/M_0, normalized omega and compactness')

# Choose the number of the desired plot 
method = input()
#method = 1

if method == 1:
    plt.scatter(ec, M, s=3)
    plt.xlabel(r'$\varepsilon_c$ [g cm$^{-3}$]', fontsize='15')
    plt.ylabel(r'$M$ [M$_\odot$]', fontsize='15')
    plt.legend()
    plt.show()

elif method == 2:
    plt.scatter(R, M, s=3)
    plt.xlabel(r'$R$ [km]', fontsize='15')
    plt.ylabel(r'$M$ [M$_\odot$]', fontsize='15')
    plt.legend()
    plt.show()
    
elif method == 3:
    plt.scatter(M, freq, s=3)
    plt.xlabel(r'$M$ [M$_\odot$]', fontsize='15')
    plt.ylabel(r'$\nu$ [Hz]', fontsize='15')
    plt.legend()
    plt.show()
    
elif method == 4:
    plt.scatter(freq, (M-Mstat)/Mstat, s=3)
    plt.xlabel(r'$\nu$ [Hz]', fontsize='15')
    plt.ylabel(r'$(M-M_*)/M_*$', fontsize='15')
    plt.legend()
    plt.show()
    
elif method == 5:
    plt.scatter(delta, (M-Mstat)/Mstat, s=3)
    plt.xlabel(r'$\Omega (R_*^3 / GM_*)^{1/2}$', fontsize='15')
    plt.ylabel(r'$(M-M_*)/M_*$', fontsize='15')
    plt.legend()
    plt.show()
    
elif method == 6:
    plt.scatter(freq, (M0-M)/M0, s=3)
    plt.xlabel(r'$\nu$ [Hz]', fontsize='15')
    plt.ylabel(r'$(M_0-M)/M_0$', fontsize='15')
    plt.legend()
    plt.show()
    
elif method == 7:
    plt.scatter(delta, (M0-M)/M0, s=3)
    plt.xlabel(r'$\Omega (R_*^3 / GM_*)^{1/2}$', fontsize='15')
    plt.ylabel(r'$(M_0-M)/M_0$', fontsize='15')
    plt.legend()
    plt.show()
  
elif method == 8:
    plt.scatter(delta, M/Mmax, s=3)
    plt.xlabel(r'$\Omega (R_*^3 / GM_*)^{1/2}$', fontsize='15')
    plt.ylabel(r'$M/M_{max}$', fontsize='15')
    plt.legend()
    plt.show()
    
elif method == 9:
    plt.scatter(freq, (R-Rstat)/Rstat, s=3)
    plt.xlabel(r'$\nu$ [Hz]', fontsize='15')
    plt.ylabel(r'$(R-R_*)/R_*$', fontsize='15')
    plt.legend()
    plt.show()

elif method == 10:
    plt.scatter(delta, (R-Rstat)/Rstat, s=3)
    plt.xlabel(r'$\Omega (R_*^3 / GM_*)^{1/2}$', fontsize='15')
    plt.ylabel(r'$(R-R_*)/R_*$', fontsize='15')
    plt.legend()
    plt.show()

    
elif method == 11:
    plt.scatter(freq, (M/Msun)*(1.47/R), s=3)
    plt.xlabel(r'$\nu$ [Hz]', fontsize='15')
    plt.ylabel(r'$\zeta$', fontsize='15')
    plt.legend()
    plt.show()
    
elif method == 12:
    plt.scatter(delta, (M/Msun)*(1.47/R), s=3)
    plt.xlabel(r'$\Omega (R_*^3 / GM_*)^{1/2}$', fontsize='15')
    plt.ylabel(r'$\zeta$', fontsize='15')
    plt.legend()
    plt.show()
    
elif method == 13:
    plt.scatter(normJ, freq, s=3)
    plt.xlabel(r'$cJ/GM_0^2$', fontsize='15')
    plt.ylabel(r'$\nu$ [Hz]', fontsize='15')
    plt.legend()
    plt.show()
    
elif method == 14:
    plt.scatter(T/W, freq, s=3)
    plt.xlabel(r'$T/W$', fontsize='15')
    plt.ylabel(r'$\nu$ [Hz]', fontsize='15')
    plt.legend()
    plt.show()
    
elif method == 15:
    ax = fig.gca(projection='3d')
    ax.scatter(delta, G*mstat/(rstat*c**2), M/Mmax)
    ax.set_xlabel(r'$\Omega (R_*^3 / GM_*)^{1/2}$', fontsize='15')
    ax.set_ylabel(r'$GM_*/Rc^2$', fontsize='15')
    ax.set_zlabel(r'$M/M_{max}$', fontsize='15')
    plt.legend()
    plt.show()
    
elif method == 16:
    ax = fig.gca(projection='3d')
    ax.scatter(delta, G*mstat/(rstat*c**2), (R-Rstat)/Rstat)
    ax.set_xlabel(r'$\Omega (R_*^3 / GM_*)^{1/2}$', fontsize='15')
    ax.set_ylabel(r'$GM_*/Rc^2$', fontsize='15')
    ax.set_zlabel(r'$(R-R_*)/R_*$', fontsize='15')
    plt.legend()
    plt.show()
    
elif method == 17:
    ax = fig.gca(projection='3d')
    ax.scatter(delta, G*mstat/(rstat*c**2),(M-Mstat)/Mstat)
    ax.set_xlabel(r'$\Omega (R_*^3 / GM_*)^{1/2}$', fontsize='15')
    ax.set_ylabel(r'$GM_*/Rc^2$', fontsize='15')
    ax.set_zlabel(r'$(M-M_*)/M_*$', fontsize='15')
    plt.legend()
    plt.show()
    
elif method == 18:
    ax = fig.gca(projection='3d')
    ax.scatter(delta, G*mstat/(rstat*c**2), (M0-M)/M0)
    ax.set_xlabel(r'$\Omega (R_*^3 / GM_*)^{1/2}$', fontsize='15')
    ax.set_ylabel(r'$GM_*/Rc^2$', fontsize='15')
    ax.set_zlabel(r'$(M_0-M)/M_0$', fontsize='15')
    plt.legend()
    plt.show()
    
else:
    print('Enter a number from 1 to 18')
