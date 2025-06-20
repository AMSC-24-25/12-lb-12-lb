# CFD library for Lattice Boltzmann Method
The main goal is to write a library in order to exploit LBM method using the arguments of the course AMSC

## Compiling
In order not to have problems in the compilation we need to be sure that all the packets needed are correctly installed: <br /> 
1. sudo apt update <br />
2. sudo apt install pkg-config <br />
3. sudo apt-get install libopencv-dev <br />
4. sudo apt-get install libc6-dev
5. sudo apt-get install gcc-10 g++-10 <br />
6. sudo apt install ffmpeg <br />

At this point we need to locate the file "opencv4.pc": <br />
dpkg -L libopencv-dev <br /><br />
Now we need to re-configure the path through this command: <br />
export PKG_CONFIG_PATH=<insert/your/path/>/pkgconfig:$PKG_CONFIG_PATH <br /> 
You should type something like: <br />
export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH <br /> <br />

To check add the line: <br />
pkg-config --modversion opencv4 <br />
If we get something like "4.6.0", then we're okay.

## Runnning
In order to run the code we need these lines: <br /> <br />
cd code <br />
chmod +x compile_and_run.sh <br />
./compile_and_run.sh <number_of_cores> <br />
<br />
and sobstitute <number_of_cores> with the number of cores with which you want to run the program. If you don't insert it it will simply run with the maximum number of threads in your pc
The first one is just to move in the folder of the code while the second compile the file with all the information needed to compute the program and the third one run that file.  

## Overview
The physical approch of this is based on the discretization of the 2D Boltzmann equation. <br />

### Space discretization
For the space discretizaion we used a common equispaced Grid in 2D: $\delta_x, \delta_y$ <br />

### Time discretization
For the time discretization we used equispaced time with distance $\delta_t=\frac{\delta_x}{c_s}$ where $c_s$ is the lattice sound speed. All the equation in the code are arleady computed for $c_s=\frac{1}{\sqrt{3}}$

### Angle discretization

In order to discretize the angle, we followed the D2Q9 approach that considers only 9 possible directions of the particles, since the moving time step allows movement of only one square. <br />
We have also added, according to that model, a weight specific to each direction. For the D2Q9 model:

$$ w_i = 
\begin{cases} 
\frac{4}{9}, & \text{if } i = 0; \\
\frac{1}{9}, & \text{if } i \in \{1, 2, 3, 4\}; \\
\frac{1}{36}, & \text{if } i \in \{5, 6, 7, 8\}.
\end{cases} $$

The general approach is the DnQm model, where `n` is the number of dimensions and `m` is the number of speeds. <br />
In order to use dimensional quantities like speed and position, we need to convert them into lattice units. For example, the height will become `L -> NY`, where `NY` is the number of points along `y` in the lattice. <br />

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
$C(f)=\frac{f_i^{eq}(\mathbf{x},t)-f_i(\mathbf{x},t)}{\tau}$
where:
* $f_i^{eq}$ is the equiliobrium distribution function obtained after a truncation of a Taylor expansion from the complete equation $f^{eq}=\frac{\rho}{(2\pi RT)^{D/2}} e^{-\frac{(\mathbf{e} - \mathbf{u})^2}{2RT}}$ where D is the dimnesion, R the universal gas costant and T absolute temperature related to the sound velocity by $c_s=3RT$. After the trunccation we obtain $f_i^{eq}=w_i\rho(1+\frac{3\mathbf{e} \cdot \mathbf{u}}{c_s^2}+\frac{9(\mathbf{e} \cdot \mathbf{u})^2}{2c_s^4}-\frac{3(\mathbf{u})^2}{2c_s^2})$
* $\tau$ is related to the kinematic viscosity $\nu$ by $\tau = \frac{\nu}{c_s^2}+0.5$ and $\nu% can be obtained from the Reynolds number $Re=\frac{u_lidL}{\nu}$ where $u_lid$ is the lid velocity in lattice units and $$L is the height of the cavity in lattice units (so $L=NY$).

### Boundry conditions
The problem requested a lid driven cavity so the boundry condition for 3 of the four walls can be chosen arbitartly. For a simple description our approch was to describe all the collision with the borse as elastic and assume perfect reflection at borders. In order to account to the driven top boundry condition we used Dirchlet boundry condition $f_{opp(i)}=f_i-2w_i\rho\frac{\mathbf{e_i} \cdot \mathbf{u}_{lid}}{c_s^2}$ with $\rho$ local density and $opp(i)$ the opposite direction of i (so for example if $i$ is right: (1,0) then $opp(i)$ will be left: (-1,0).

## Code structure
### Initialization
In this process we perfrom the initialization of the quantity in particular we start from a full null velocity, a uniform and equal density (that in lattice units it's 1) and a distribution function based only on the weights: $f_i(\mathbf{x},t=0)=w_i$. Here the equilibrium distribution function can be calculated with the formula from the Taylor expansion above or, since we are in a static initial case, as a copi of $f$.

### Collision
THe collision term is a simple result of the BGK approch explained above so we redefine the probability distribution function as a result of the operation $f_i^{after-collision}(\mathbf{x},t)=f_i(\mathbf{x},t)+\frac{f_i^{eq}(\mathbf{x},t)-f_i(\mathbf{x},t)}{\tau}$

### Streaming and boundry conditions
The streaming term is simply described by a variation in the space coordinate in time: $f_i(\mathbf{x}+\mathbf{e_i}\delta_t,t+delta_t)=f_i(\mathbf{x},t)$. ABout the boundry condition the idea is to invert the direction of the particles that are flowing outside so we consider the particles at the borders that are going outside and we simply reflect their direction. For instance if we take a particle that has $\mathbf{x}=(NX-1,y)$ with direction right we impose that after this step it will have diretion left and will be again in the cell $\mathbf{x}=(NX-1,y)$. 

### Calculation of macroscopic quantities
From the calculation of the probability distribution function we can have all the macroscopic quantities such as density $\rho$ and velocity $\mathbf{u}$. In order to calculate them we use $\rho=\sum_i f_i$ and $\mathbf{u}=\frac{\sum_i f_i\mathbf{e_i}}{\rho}$

### Printing result and videomaking
We have implemented the visualization part using the C-native library OpenCV in order to have fast performances. <br />
We create 2 heatmaps in which the velocity magnitude and the density are shown at each time step. In particular we made sure to work with normalized matrices than converted into 8-bit images. Each frame is than saved in a subfolder. Regarding the OpenCV::Mat function we noticed that it uses a row-major indexing, so, in order to have more easily readable images we need to flip the images along the x axis. <br />
When the simulation is completed, a last line in the code creates the animation of it, assembling all saved frames. <br /> <br />
We implemented also some codes for the static and dynamic visualization on python, whose read the .csv output files. However we noticed that the compilation time gets way more long, so we've preferred to include in the overall default code only the C++ visualization 


## Key feature
### Lid velocity considerations
In order to avoid numerical instability due to the fact that the velocity is, at the first step, different from zero at the top and the probability distribution function instead describes a static situation we decided to use at first a zero velocity on all the fluid. Then we linearly increase the lid velocity till the desired value, in this way we plan to resolve possible instability in the first steps of the code
### Parallelization
We decide to exploit the parallelizion with openmp. We exploited strong and weak scalability test on the latest version and get the following results:<br /><br />
STRONG SCALABILITY:
<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="code/LBM_classic/video/Strong_fede.png" alt="Strong scalability_F" width="300">
  <img src="code/LBM_classic/video/Strong_Scalability_P.png" alt="Strong scalability_P" width="300">
</div>
<br /><br /> 
WEAK SCALABILITY:
<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="code/LBM_classic/video/Weak_fede.png" alt="Weak scalability_F" width="300">
  <img src="code/LBM_classic/video/Weak_Scalability_P.png" alt="Weak scalability_P" width="300">
</div>
<br /><br /> 
GRID SIZE IMPACT:
<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="code/LBM_classic/video/Grid_fede.png" alt="Grid_size_impact_F" width="300">
  <img src="code/LBM_classic/video/Grid_size_impact_P.png" alt="Grid_size_impact_P" width="300">
</div>
<br /><br /> 

