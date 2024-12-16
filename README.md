# CFD library for Lattice Boltzmann Method
THe main goal is to write a library in order to exploit LBM method using the arguments of the course AMSC

## Overview
The physical approch of this is based on the discretization of the 2D Boltzmann equation
### Space discretization
For the space discretizaion we used a common equispaced Grid in 2D: $dx dy$

###Time discretization
For the time discretization we used equispaced time with distance $dt=frac{dx}/{c_s}$ where $c_s$ is the lattice sound speed

###ANgle discretization
In order to discretize the angle we followed the D2Q9 apporch that consider only 9 possible directions of the particles since the moving time step allows to move of only one square

##Physical interpretation
###Boltzmann equation

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program
