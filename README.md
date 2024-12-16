# CFD library for Lattice Boltzmann Method
THe main goal is to write a library in order to exploit LBM method using the arguments of the course AMSC

## Overview
The physical approch of this is based on the discretization of the 2D Boltzmann equation

### Space discretization
For the space discretizaion we used a common equispaced Grid in 2D: $\delta_x, \delta_y$

### Time discretization
For the time discretization we used equispaced time with distance $\delta_t=\frac{\delta_x}{c_s}$ where $c_s$ is the lattice sound speed. All the equation in the code are arleady computed for $c_s=\frac{1}{\sqrt{3}}$

### Angle discretization
In order to discretize the angle we followed the D2Q9 apporch that consider only 9 possible directions of the particles since the moving time step allows to move of only one square.
We have also added according to that model a weight specific of any direction: for the D2Q9 model 
$$
w_i = \begin{cases} 
\frac{4}{9} & \text{if } i = 0 \\ 
\frac{1}{9} & \text{if } i \in \{1, 2, 3, 4\} \\ 
\frac{1}{36} & \text{if } i \in \{5, 6, 7, 8\} 
\end{cases}
$$
The general approch is the DnQm where n is the number of dimensions and m is the number of speeds.
In order to use dimsension quantity like speed and position we need to convert them into lattice units so the height will become L->NY where NY is the number of points along y in the lattice

## Physical interpretation and mathematical development

### Boltzmann equation
The Boltzmann equation describes the behaviour of thermodynaic system by the use of the probability density function. The resulting differential equation obtained in the general case is:
$\frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \mathbf{F} \cdot \nabla_{\mathbf{v}} f = \left( \frac{\partial f}{\partial t} \right)_{\text{coll}}$
where:
* $f$ is the probability distribution function that in general is a function of postion $\mathbf{x}$, velocity $\mathbf{v}$ and time $t$. The behaviour on the velocity can be decomposed in the behaviour with energy $E$ and direction $\mathbf{\Omega}$: $f(\mathbf{x},E,\mathbf{\Omega},t)$. It is defined as $dN=f(\mathbf{x},\mathbf{v},t)d^3\mathbf{x}d^3\mathbf{v}$ with N number of particles.
* $\mathbf{F}$ is the Force field acting on the particles. In the following the force will be assumed zero over all the grid.
* $\left(\frac{\partial f}{\partial t} \right)_{\text{coll}}$ is the term that describes collision and can be modelled in many different ways depending on our goal. It can be even neglected leading to a collisionless description.

### Discretized equation
After the discretization in time , space and angle we obtain an equation of the kind:
$f_i(\mathbf{x}+\mathbf{e}_i\delta_t,t+delta_t)-f_i(\mathbf{x},t)+F_i=C(f)$
where:
* $\mathbf{e}_i$ is one of the directions considered in the model
* $C(f)$ is the collision term

### Collision term
THe collision term can be treated in many different ways, the approch that we followed is to use the Bhatnagar Gross Krook model for relaxation equilibrium:
$C(f)=f_i(\mathbf{x},t)+\frac{f_i^{eq}(\mathbf{x},t)-f_i(\mathbf{x},t)}{\tau}$
where:
* $f_i^{eq}$ is the equiliobrium distribution function obtained after a truncation of a Taylor expansion from the complete equation $f^{eq}=\frac{\rho}{(2\pi RT)^{D/2}} e^{-\frac{(\mathbf{e} - \mathbf{u})^2}{2RT}}$ where D is the dimnesion, R the universal gas costant and T absolute temperature related to the sound velocity by $c_s=3RT$. After the trunccation we obtain $f_i^{eq}=w_i\rho(1+\frac{3\mathbf{e} \cdot \mathbf{u}}{c_s^2}+\frac{9(\mathbf{e} \cdot \mathbf{u})^2}{2c_s^4}-\frac{3(\mathbf{u})^2}{2c_s^2})$
* $\tau$ is related to the kinematic viscosity $\nu$ by $\tau = \frac{\nu}{c_s^2}+0.5$ and $nu% can be obtained from the Reynolds number $Re=\frac{u_lidL}{\nu}$ where $u_lid$ is the lid velocity in lattice units and $$L is the height of the cavity in lattice units (so $L=NY$)

### Boundry conditions
The problem requested a lid driven cavity so the boundry condition for 3 of the four walls can be chosen arbitartly. For a simple description our approch was to describe all the collision with the borse as elastic and assume perfect reflection at borders. In order to account to the driven top boundry condition we used Dirchlet boundry condition $f_{opp(i)}=f_i-2w_i\rho\frac{\mathbf{e_i} \cdot \mathbf{u}_{lid}}{c_s^2}$ with $\rho$ local density

## Code structure
### Initialization
In this process we perfrom the initialization of the quantity in particular we start from a full null velocity, a uniform and equal density (that in lattice units it's 1) and a distribution function based only on the weights: $f_i(\mathbf{x},t=0)=w_i$. Here the equilibrium distribution function can be calculated with the formula from the Taylor expansion above or, since we are in a static initial case, as a copi of $f$

### Collision
THe collision term is a simple result of 

### Streaming and boundry conditions

### Calculation of macroscopi quantities

### Printing result and videomaking

## Key feature
