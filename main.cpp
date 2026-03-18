#include <iostream>
#include <fstream>
#include <cstdlib> // Required for srand() and rand()
#include <ctime>   // Required for time()
#include <omp.h>

#include "kernel.hpp"
#include "ACA.hpp"
#include "HODLR2D.hpp"


typedef Eigen::VectorXd Vec;
typedef Eigen::MatrixXd Mat;

int main(int argc,char* argv[]){

	std::ofstream file("output.txt");  // create fil


    if (!file) {
        std::cout << "Error opening file\n";
        return 1;
    }

    int sqrtRootN   =	atoi(argv[1]);// how many particle in one axis direction
	double L    	=	atoi(argv[2]); //domain size
	int TOL_POW     = atoi(argv[3]); // tolerance for compressions in powers of 10
	int nParticlesInLeafAlong1D	=	atoi(argv[4]); // assuming the particles are located at tensor product chebyNodes/uniform

    int nLevelsUniform	=	ceil(log(double(sqrtRootN)/nParticlesInLeafAlong1D)/log(2));//number of levels with the assumption that particles distributed uniformly
    
    std::vector<pts2D> particles_X, particles_Y; //dummy variables - to create the locations of the particle's charges

    userkernel* mykernel		=	new userkernel(sqrtRootN, L, nLevelsUniform); // This object will hold our charge locations, kernel information,hence we can create our matrix

    unsigned N = sqrtRootN*sqrtRootN;
    unsigned nLevels  = nLevelsUniform;

    double* locations       =       new double [N*2];   // This stroes the (x,y) for each location 
/**********************************Storing the x and y co-ordinates in single array*******************************************/
    unsigned count_Location =       0;
	for (unsigned j=0; j<N; ++j) {
			for (unsigned k=0; k<2; ++k) {
				if(k == 0) {
					locations[count_Location]       =       mykernel->gridPoints[j].x;//2*(int(rand())%2)-1;//double(rand())/double(RAND_MAX);
					++count_Location;
				}
				else {
					locations[count_Location]       =       mykernel->gridPoints[j].y;//2*(int(rand())%2)-1;//double(rand())/double(RAND_MAX);
					++count_Location;
				}
			}
	}
	particles_X = mykernel->gridPoints;
/******************************************this part working fine **************************************************************/    
  Vec  Algo_x;
  double err;
  double start,end;

/**************************************************  ONES AT RANDOM PLACES  ****************************************************/
	// For testing the code, x is considered to be vector of ones and zeros and the corresponding b is calculated in the following code snippet

		Vec true_x=Vec::Zero(N);
		int n = N/500; //randomly choosing n different indices where b is set to 1, b at the rest of the indices is set to 0
		srand(time(NULL));
		std::set<int> s;
		while(s.size() < n) {
			int index	=	rand()%N;
			s.insert(index);
		}
		std::set<int>::iterator it;
		for (it = s.begin(); it != s.end(); it++) {
			true_x(*it) = 1.0;
		}
		Vec true_Ax = Vec::Zero(N);//This part is to get b value to apply the algorithm to find approx x
		for (it = s.begin(); it != s.end(); it++) {
			true_Ax = true_Ax + mykernel->getCol(N,*it);
		}
		//
   // start	=	omp_get_wtime();
	HODLR2D<userkernel> *HODLR = new HODLR2D<userkernel>(mykernel, N, nLevelsUniform, TOL_POW, locations);
	
    std::cout<<"nLevels:"<<nLevels<<std::endl;
	std::cout<<"these two nodes redirection climbs " << HODLR->A->climber(3,31,53,0)<<" Levels up"<<std::endl;
	//end		=	omp_get_wtime();
	//double timeAssemble =	(end-start);
    //std::cout<<"nLevels:"<<nLevels<<std::endl;
	//////////////////////// ELIMINATE PHASE ////////////////////////////////////////
	// start	=	omp_get_wtime();ss
	// HODLR->factorize();
	// end		=	omp_get_wtime();
	// double timeFactorize =	(end-start);
   
	// //start	=	omp_get_wtime();
	// HODLR->backSubstitute1(true_Ax);
	// //end		=	omp_get_wtime();
	// //double timeFactorize +=	(end-start);

	// start	=	omp_get_wtime();
	// HODLR->backSubstitute2();
    // end		=	omp_get_wtime();
	// timeFactorize +=	(end-start);

	// start	=	omp_get_wtime();
	// HODLR->backSubstitute3();
    // end		=	omp_get_wtime();
    // timeFactorize +=	(end-start); 

	// start	=	omp_get_wtime();
	// HODLR->getSolution(Algo_x);
    // end		=	omp_get_wtime();
	// double timeSolve =	(end-start);

	// err = (true_x - Algo_x).norm()/true_x.norm();

	// std::cout << std::endl << "Max rank of compressible sub-matrices: " << HODLR->A->getMaxRank() << std::endl;  // (including those of compressible fill-ins)

	// std::cout << std::endl << "Assemble time: " << timeAssemble << std::endl;

	// std::cout << std::endl << "Factorization time: " << timeFactorize << std::endl;

	// std::cout << std::endl << "Solve time: " << timeSolve << std::endl;

	// std::cout << std::endl << "Relative forward error in the solution: " << err << std::endl; // relative forward error in 2 norm sense
    

	delete HODLR;

	delete locations;
	delete mykernel;

	file.close();
}