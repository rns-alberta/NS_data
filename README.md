# NS_data

Output files from the sequences.c code that computes a series of constant 
baryonic mass tracks of neutron stars with changing spin velocity. Each column in the file represents:

Central energy density
Total mass
Baryonic mass
Mass when the NS is not rotating
Maximum mass of a non rotating NS for a given EOS 
Radius
Ratio r_p/r_e
Radius when the NS is not rotating
Spin frequency
Kepler limit
Angular momentum
Rotational kinetic energy
Gravitational binding energy
Maximum radius of the non rotating star for a given EOS 
Ratio of the maximum mass and the maximum radius of the non rotating star


The python (version 2.7) code Plots.py generates various plots combining the data from the ten files. 
To run it from the Terminal the following command line is used "python Plots.py" and by typing a number from 1 to 18 
and "enter" we get the chosen plot.

The code "Plot_oneEOS.py" can be run in the same way, but this only generates the plot using one equation of state, 
instead of the 10 in the code "Plots.py"
