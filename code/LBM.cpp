#include <iostream>
#include <array>
#include <vector>

//First version straight code
int main(){
    const unsigned int NSTEPS=10;
    const unsigned int NX = 8;       // Number of nodes in x
    const unsigned int NY = NX;       // Number of nodes in y
    const unsigned int ndirections=9; //We are using D2Q9, if we pass to D3Q27 we have to change
    const double u_lid=1; //velocity for BC, imposed
    const double Re=100; //imposed
    const double L=1; //imposed
    const double nu= (u_lid*L)/Re; // calculated from formula of Re
    const double tau=3.0*nu+0.5;  //for BGK model
    const double rho0=1.0; //imposed at BC

    //D2Q9 lattice directions
    const std::array<std::pair<int, int>,9> direction = {
        std::make_pair(0, 0),   // Rest
        std::make_pair(1, 0),   // Right
        std::make_pair(0, 1),   // Up
        std::make_pair(-1, 0),  // Left
        std::make_pair(0, -1),  // Down
        std::make_pair(1, 1),   // Top-right diagonal
        std::make_pair(-1, 1),  // Top-left diagonal
        std::make_pair(-1, -1), // Bottom-left diagonal
        std::make_pair(1, -1)   // Bottom-right diagonal
    };
    //D2Q9 lattice weigths
    const std::array<double,9> weight = {
        4.0 / 9.0, // Rest
        1.0 / 9.0, // Right
        1.0 / 9.0, // Up
        1.0 / 9.0, // Left
        1.0 / 9.0, // Down
        1.0 / 36.0, // Top-right diagonal
        1.0 / 36.0, // Top-left diagonal
        1.0 / 36.0, // Bottom-left diagonal
        1.0 / 36.0 // Bottom-right diagonal
    };

    //Equilibrium
    //at first define the velocity at equilibrium
    //We must have u=0 everywhere exept for the lid that is dirven
    std::array<std::pair<int, int>,2> u;
    for(unsigned int x=0;x<NX;++x){
        for(unsigned int y=0;y<NY;++y){
            u[x,y]=std::make_pair(0, 0);
        }
        u[x,NY]=std::make_pair(u_lid, 0);
    }
    //Now we define the equilibrium distribution function in every point for each direction
    std::array<double,3> f_eq;
    double cx;//all of this can be dumped if defined below it's just to sketch and got result
    double cy;
    double ux;
    double uy;
    double c_scalar;
    for(unsigned int x=0;x<NX;++x){
        for(unsigned int y=0;y<NY;++y){
            ux = u[x,y].first;
            uy = u[x,y].second;
            for (unsigned int i=0;i<ndirections;++i){
                cx = direction[i].first;
                cy = direction[i].second;
                c_scalar=cx*ux+cy*uy;
                //Here we compute the BGK formula to calculate the distribution function
                //f=w*rho*(1+3*(c·u)+9/2*((c·u)^2)-3/2*(u·u)
                //w are the weight, c are the directions, rho is the density, u is the veclocity
                f_eq[x,y,i]=weight[i]*rho0*(1+3*c_scalar+9/2*c_scalar*c_scalar-3/2*(ux*ux+uy*uy));
            }
        }
    }
    //Since we are using equilibrium values we can initialize rho (and u) using
    //rho=rho0 everywhere (u=0 everywhere exept for the lid BC where u=(u_lid,0))
    //We show below the way that we need to implement in the code in order to get the value at each time
    //Now that we have equilibrium distribution function we can compute density overall
    std::array<double,2> rho;
    for (unsigned int x=0;x<NX;++x){
        for (unsigned int y=0;y<NY;++y){
            rho[x,y]=0;//initialize
            for (unsigned int i=0;i<ndirections;i++){
                //we use rho=integral(f) over the directions that discretized becomes
                rho[x,y]+=f_eq[x,y,i];
            }
        }
    }
    //Although we arleady have u we show a way to compute it for successive code
    
    for (unsigned int x=0;x<NX;++x){
        for (unsigned int y=0;y<NX;++y){
            u[x,y]=std::make_pair(0, 0);//initialize
            for (unsigned int i=0;i<ndirections;i++){
                u[x,y].first+=f_eq[x,y,i]*direction[i].first;
                u[x,y].second+=f_eq[x,y,i]*direction[i].second;
            }
            u[x,y].first=u[x,y].first/rho[x,y];
            u[x,y].second=u[x,y].second/rho[x,y];
        }
    }

    for(unsigned int t=0;t<NSTEPS;++t){
        //move the particle and apply conditions
    }

    return 0;
}  
