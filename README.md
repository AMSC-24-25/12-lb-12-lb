# CFD library for Lattice Boltzmann Method
THe main goal is to write a library in order to exploit LBM method using the arguments of the course AMSC

## Overview
The physical approch of this is based on the discretization of the 2D Boltzmann equation
### Space discretization
For the space discretizaion we used a common equispaced Grid in 2D: $dx, dy$

### Time discretization
For the time discretization we used equispaced time with distance $dt=\frac{dx}/{c_s}$ where $c_s$ is the lattice sound speed.

### ANgle discretization
In order to discretize the angle we followed the D2Q9 apporch that consider only 9 possible directions of the particles since the moving time step allows to move of only one square.
We have also added according to that model a weight specific of any direction.

## Physical interpretation

### Boltzmann equation
The Boltzmann equation describes the behaviour of thermodynaic system by the use of the probability density function. The resulting differential equation obtained in the general case is:
$\frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \mathbf{F} \cdot \nabla_{\mathbf{v}} f = \left( \frac{\partial f}{\partial t} \right)_{\text{coll}}$
