#ifndef _Hodlr2DTree_HPP__
#define _Hodlr2DTree_HPP__


#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

class Box {
public:
	int boxNumber;  
	int parentNumber;
	int childrenNumbers[4];
	int vs_neighbor[4];
    int es_neighbor[4];
	int innerNumbers[4];
	int outerNumbers[8];
	bool Eliminated;

	Box () {//prototype of a box in the tree
		boxNumber		=	-1;
		parentNumber	=	-1;
		for (int l=0; l<4; ++l) {
			childrenNumbers[l]	=	-1;
		}
		for (int l=0; l<4; ++l) {
			vs_neighbor[l]	=	-1;
		}
        for (int l=0; l<4; ++l) {
			es_neighbor[l]	=	-1;
		}
		for (int l=0; l<4; ++l) {
			innerNumbers[l]		=	-1;
		}
		for (int l=0; l<8; ++l) {
			outerNumbers[l]		=	-1;
		}
		Eliminated = false;
	}

	int NumParticles;
	int NumMultipoles; // = NumLocals
	int NumLocals; // = NumLocals
	pts2D center;
	Vec particles;
	Vec multipoles;
	Vec locals;
	Vec particle_rhs;
	Vec multipole_rhs; //intially value will be 0
	Vec local_rhs; //intially value will be 0

	std::map<int, Mat> M2L;
	std::map<int, Mat> P2P;
	std::map<int, Mat> M2P;
	std::map<int, Mat> P2L;
    std::map<int, Mat> L2P_f;
	Mat L2M_f; //fill-ins that occur in U.transpose() equation due to elimination of x
	std::map<int, Mat> P2M_f; //fill-ins that occur in U.transpose() equation due to elimination of x
	std::map<int, Mat> M2M_f;
	Mat L2P;					//	Transfer from locals of parent to locals of children.
	Mat P2M;					//	Transfer from multipoles of 4 children to multipoles of parent.
	//	The following will be stored only at the leaf nodes
	Eigen::ColPivHouseholderQR<Mat> U_qr;
	Eigen::ColPivHouseholderQR<Mat> V_qr;
    std::vector<pts2D> particle_loc;
    std::vector<int> chargeLocations;

	std::vector<int> incoming_chargePoints;//equivalent points {y_{k}^{B,i}}
	std::vector<int> incoming_checkPoints;//check points {x_{k}^{B,i}}
	std::vector<int> outgoing_chargePoints;
	std::vector<int> outgoing_checkPoints;
	Mat Ac;
	Mat Ar;
};

template <typename kerneltype>

class HODLR2DTree {
	public:
	kerneltype* K;
	int nLevels;		                      	//	Number of levels in the tree.
	int N;				                     	//	Number of particles.
	double L;				                    //	Semi-length of the simulation box.
	double smallestBoxSize;	                    //	This is L/2.0^(nLevels).

	std::vector<int> nBoxesPerLevel;			//	Number of boxes at each level in the tree.
	std::vector<double> boxRadius;				//	Box radius at each level in the tree assuming the box at the root is [-1,1]^2
	std::vector<std::vector<Box> > tree;	    //	The tree storing all the information.

	int nParticlesInLeafAlong1D;
    int nParticlesInLeaf;
 
	std::vector<double> Nodes1D;
	std::vector<pts2D> Nodes;
    std::vector<pts2D> gridPoints;              //all particles in domain
	int TOL_POW;
	double RRQR_threshold,LU_threshold;
	Vec b_error_check;
	double* locations;
	std::vector<std::pair<int, int> > P2L_M2P;
	std::vector<std::pair<int, int> > P2P;

	HODLR2DTree(kerneltype* K, int N, int nLevels, int TOL_POW, double* locations) {
		this->K					=	K;
		this->nLevels		=	nLevels;
		this->L					=	1.0;
		this->locations = locations;
		this->TOL_POW = TOL_POW;
		this->RRQR_threshold = pow(10,-1.0*TOL_POW);
		this->LU_threshold = pow(10,-1.0*TOL_POW);
        nBoxesPerLevel.push_back(1);
		boxRadius.push_back(L);
		for (int k=1; k<=nLevels; ++k) {
			nBoxesPerLevel.push_back(4*nBoxesPerLevel[k-1]);
			boxRadius.push_back(0.5*boxRadius[k-1]);
		}
		this->smallestBoxSize	=	boxRadius[nLevels];
		K->a					=	smallestBoxSize;
		this->N				=	N; 
	}

	void createTree(){

		Box root;
		root.boxNumber=0;
		std::vector<Box> rootLevel;
		rootLevel.push_back(root);
		tree.push_back(rootLevel);
		
		for (int j=1; j<=nLevels; ++j) {//Let's create the skeleton of the tree first. 
			std::vector<Box> level;
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				Box box;
				box.boxNumber		=	k;
				box.parentNumber	=	k/4;
				for (int l=0; l<4; ++l) {
					box.childrenNumbers[l]	=	4*k+l;
				}
				level.push_back(box);
			}
			tree.push_back(level);
		}

	}
 
	void assign_Child0_Interaction(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k;
		int nN;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  e2  |  v2  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  **  |  e1  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].es_neighbor[1]	=	nC+1;
			tree[nL][nC].vs_neighbor[2]	=	nC+2;
			tree[nL][nC].es_neighbor[2]	=	nC+3;
		}
        //	Assign children of parent's zeroth neighbor
		{
			/****************************/
			/*	 ______			  		*/
			/*	|	   |	  			*/
			/*	|  **  |				*/
			/*	|______|______			*/
			/*	|	   |	  |			*/
			/*	|  e0  |  v1  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  o0  |  o1  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].es_neighbor[0];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[0]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[1]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].vs_neighbor[1]	    =	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].es_neighbor[0]	    =	tree[j][nN].childrenNumbers[3];
			}
		}
       //	Assign children of parent's first neighbor
		{
			/************************************/
			/*				   _____________	*/
			/*				  |	     |	    |	*/
			/*				  |  I1  |  O3  |	*/
			/*	 ______		  |______|______|	*/
			/*	|	   |	  |	     |	    |	*/
			/*	|  **  |	  |  I0  |  O2  |	*/
			/*	|______|	  |______|______|	*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].es_neighbor[1];
			if (nN!=-1) {
				tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[2]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[3]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[3];
			}

		}
	   //	Assign children of parent's second neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  O5  |  O4  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  I3  |  I2  |		*/
			/*	|______|______|		*/
			/*						*/
			/*						*/
			/*	 ______				*/
			/*  |	   |			*/
			/*	|  **  |			*/
			/*	|______|			*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].es_neighbor[2];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[4]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[5]	=	tree[j][nN].childrenNumbers[3];
			}
		}
       //	Assign children of parent's seventh neighbor
		{
			/****************************/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  O6  |  v3  |			*/
			/*	|______|______|______	*/
			/*  |	   |	  |		 |	*/
			/*	|  O7  |  e3  |	 **  |	*/
			/*	|______|______|______|	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].es_neighbor[3];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[7]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].es_neighbor[3]	    =	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].vs_neighbor[3] 	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[6]	=	tree[j][nN].childrenNumbers[3];
			}
		}
	}
	
	void assign_Child1_Interaction(int j, int k){
        int nL	=	j+1;
		int nC	=	4*k+1;
		int nN;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  v3  |  e2  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  e3  |  **  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].es_neighbor[3]	=	nC-1;
			tree[nL][nC].es_neighbor[2]	=	nC+1;
			tree[nL][nC].vs_neighbor[3]	=	nC+2;
		}
 
		//	Assign children of parent's zeroth neighbor
		{
			/****************************/
			/*	 		______		  	*/
			/*		   |	  |			*/
			/*	       |  **  |			*/
			/*	 ______|______|			*/
			/*	|	   |	  |			*/
			/*	|  v0  |  e0  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  O0  |  O1  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].es_neighbor[0];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[0]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[2]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].es_neighbor[0] 	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].vs_neighbor[0]	    =	tree[j][nN].childrenNumbers[3];
			}
		}
	    //	Assign children of parent's first neighbor
		{
			/****************************/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  v2  |	 O3	 |  */
			/*	 ______|______|______|	*/
			/*	|	   |	  |	     |	*/
			/*	|  **  |  e1  |  O2  |  */
			/*	|______|______|______| 	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].es_neighbor[1];
			if (nN != -1){
				tree[nL][nC].es_neighbor[1]	    =	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[2]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[3]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].vs_neighbor[2] 	=	tree[j][nN].childrenNumbers[3];
			}
		}
	   //	Assign children of parent's second neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  O5  |  O4  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  I1  |   I0 |		*/
			/*	|______|______|		*/
			/*						*/
			/*						*/
			/*		    ______		*/
			/* 		   |	  |		*/
			/*		   |  **  |		*/
			/*		   |______|		*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].es_neighbor[2];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[4]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[5]	=	tree[j][nN].childrenNumbers[3];
			}
		}

	   //	Assign children of parent's seventh neighbor
		{
			/************************************/
			/*									*/
			/*	 _____________					*/
			/*	|	   |	  |					*/
			/*	|  O6  |  I2  |					*/
			/*	|______|______|		  ______	*/
			/*  |	   |	  |		 |		|	*/
			/*	|  O7  |  I3  |	 	 |	**  |	*/
			/*	|______|______|		 |______|	*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].es_neighbor[3];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[7]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[6]	=	tree[j][nN].childrenNumbers[3];
			}
		}
	}

	void assign_Child2_Interaction(int j, int k){
        int nL	=	j+1;
		int nC	=	4*k+2;
		int nN;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  e3  |  **  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  v0  |  e0  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].vs_neighbor[0]	=	nC-2;
			tree[nL][nC].es_neighbor[0]	=	nC-1;
			tree[nL][nC].es_neighbor[3]	=	nC+1;
		}

       //	Assign children of parent's zeroth neighbor
		{
			/****************************/
			/*	 		______		  	*/
			/*		   |	  |			*/
			/*	       |  **  |			*/
			/*	 	   |______|			*/
			/*							*/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I0  |  I1  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  O0  |  O1  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].es_neighbor[0];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[0]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[1]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[3];
			}
		}
       //	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 ____________________	*/
			/*	|	   |	  |	     |	*/
			/*	|  **  |  e1  |	 O3	 |  */
			/*	|______|______|______|	*/
			/*		   |	  |	     |	*/
			/*		   |  v1  |  O2  |  */
			/*		   |______|______| 	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].es_neighbor[1];
			if (nN != -1) {
				tree[nL][nC].vs_neighbor[1]	    =	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[2]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[3]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].es_neighbor[1]	=	tree[j][nN].childrenNumbers[3];
			}
		}

      //	Assign children of parent's second neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  O5  |  O4  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  v3  |  e2  |		*/
			/*	|______|______|		*/
			/*		   |	  |		*/
			/*		   |  **  |		*/
			/*		   |______|		*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].es_neighbor[2];
			if (nN != -1) {
				tree[nL][nC].vs_neighbor[3]  	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].es_neighbor[2]  	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[4]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[5]	=	tree[j][nN].childrenNumbers[3];
			}
		}
	//	Assign children of parent's seventh neighbor
		{
			/************************************/
			/*									*/
			/*	 _____________		  ______	*/
			/*	|	   |	  |		 |	    |	*/
			/*	|  O6  |  I2  |		 |	**	|	*/
			/*	|______|______|		 |______|	*/
			/*  |	   |	  |		 			*/
			/*	|  O7  |  I3  |	 	 			*/
			/*	|______|______|		 			*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].es_neighbor[3];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[7]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[6]	=	tree[j][nN].childrenNumbers[3];
			}
		}
	}
   
	void assign_Child3_Interaction(int j,int k){
		int nL	=	j+1;
		int nC	=	4*k+3;
		int nN;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  **  |  e1  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  e0  |  v1  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].es_neighbor[0]	=	nC-3;
			tree[nL][nC].vs_neighbor[1]	=	nC-2;
			tree[nL][nC].es_neighbor[1]	=	nC-1;
		}
	   //	Assign children of parent's zeroth neighbor
		{
			/****************************/
			/*	 ______		  			*/
			/*	|	   |				*/
			/*	|  **  |				*/
			/*	|______|				*/
			/*							*/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I0  |  I1  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  O0  |  O1  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].es_neighbor[0];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[0]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[1]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[3];
			}
		}
	   //	Assign children of parent's first neighbor
		{
			/************************************/
			/*	 ______		   _____________	*/
			/*	|	   |	  |	     |		|	*/
			/*	|  **  |      |	 I3	 |  O3	|	*/
			/*	|______|	  |______|______|	*/
			/*		   		  |	  	 |	    |	*/
			/*		   		  |  I2  |  O2  |  	*/
			/*		   		  |______|______| 	*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].es_neighbor[1];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[2]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[3]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[3];
			}
		}
	   //	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  O5  |  O4  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  e2  |  v2  |		*/
			/*	|______|______|		*/
			/*	|	   |			*/
			/*	|  **  |			*/
			/*	|______|			*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].es_neighbor[2];
			if (nN != -1) {
				tree[nL][nC].es_neighbor[2]	    =	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].vs_neighbor[2]	    =	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[4]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[5]	=	tree[j][nN].childrenNumbers[3];
			}
		}
	  //	Assign children of parent's seventh neighbor
		{
			/****************************/
			/*							*/
			/*	 ____________________	*/
			/*	|	   |	  |		 |	*/
			/*	|  O6  |  e3  |	 **	 |	*/
			/*	|______|______|______|	*/
			/*  |	   |	  |		 	*/
			/*	|  O7  |  v0  |	 	 	*/
			/*	|______|______|		 	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].es_neighbor[3];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[7]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].vs_neighbor[1]	    =	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].es_neighbor[3]	    =	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[6]	=	tree[j][nN].childrenNumbers[3];
			}
		}
	}
	//Lets assign the child interactions:
  	void assign_Box_Interactions(int j, int k) {
		assign_Child0_Interaction(j,k);
		assign_Child1_Interaction(j,k);
		assign_Child2_Interaction(j,k);
		assign_Child3_Interaction(j,k);
	}

	//	Assigns the interactions for the children all boxes at a given level
	void assign_Level_Interactions(int j) {
		#pragma omp parallel for
		for (int k=0; k<nBoxesPerLevel[j]; ++k) {
			assign_Box_Interactions(j,k);
		}
	}

	//	Assigns the interactions for the children all boxes in the tree
	void assign_Tree_Interactions() {
		for (int j=0; j<nLevels; ++j) {
			assign_Level_Interactions(j);
		}
	}
  
    void assign_Center_Location() {
		int J;
		tree[0][0].center.x	=	0.0;
		tree[0][0].center.y	=	0.0;
		for (int j=0; j<nLevels; ++j) {
			J	=	j+1;
			double shift	=	0.5*boxRadius[j];
			#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				tree[J][4*k].center.x		=	tree[j][k].center.x-shift;
				tree[J][4*k+1].center.x	=	tree[j][k].center.x+shift;
				tree[J][4*k+2].center.x	=	tree[j][k].center.x+shift;
				tree[J][4*k+3].center.x	=	tree[j][k].center.x-shift;

				tree[J][4*k].center.y		=	tree[j][k].center.y-shift;
				tree[J][4*k+1].center.y	=	tree[j][k].center.y-shift;
				tree[J][4*k+2].center.y	=	tree[j][k].center.y+shift;
				tree[J][4*k+3].center.y	=	tree[j][k].center.y+shift;
			}
		}
	}

	void assignChargeLocations() {
		for (size_t i = 0; i < N*2; i+=2) {
			pts2D temp;
			temp.x = locations[i];
			temp.y = locations[i+1];
			gridPoints.push_back(temp);
		}
		K->particles_X = gridPoints;//object of base class FMM_Matrix
		K->particles_Y = gridPoints;

		for (size_t i = 0; i < N; i++) {
			tree[0][0].chargeLocations.push_back(i);//(0,1,...,N) This is what charge locations
		}
		for (size_t j = 0; j < nLevels; j++) { //assign particles to its children
			for (size_t k = 0; k < nBoxesPerLevel[j]; k++) {
				int J = j+1;
				int Kp = 4*k;
				for (size_t i = 0; i < tree[j][k].chargeLocations.size(); i++) {
					int index = tree[j][k].chargeLocations[i];
					if (gridPoints[index].x <= tree[j][k].center.x) { //children 0,3
						if (gridPoints[index].y <= tree[j][k].center.y) { //child 0
							tree[J][Kp].chargeLocations.push_back(index);
						}
						else { //child 3
							tree[J][Kp+3].chargeLocations.push_back(index);
						}
					}
					else { //children 1,2
						if (gridPoints[index].y <= tree[j][k].center.y) { //child 1
							tree[J][Kp+1].chargeLocations.push_back(index);
						}
						else { //child 2
							tree[J][Kp+2].chargeLocations.push_back(index);
						}
					}
				}
			}
		}
	}
   
	void clearNonLeafChargeLocations() {
		for (int j = nLevels-1; j >= 0; j--) {
			for (size_t k = 0; k < nBoxesPerLevel[j]; k++) {
				tree[j][k].chargeLocations.clear();
			}
		}
	} 

    void get_L2P_P2M_box(int j, int k) {//wants to maintain the size of incoming check points and outgoing check points
		if (tree[j][k].incoming_checkPoints.size() < tree[j][k].outgoing_checkPoints.size()) {//if outgoing check points are large we are just keep the first few outgoing checkpoints 
			 //which is the size of incoming check points
			tree[j][k].outgoing_checkPoints.erase(tree[j][k].outgoing_checkPoints.begin()+tree[j][k].incoming_checkPoints.size(), tree[j][k].outgoing_checkPoints.end());
			tree[j][k].outgoing_chargePoints.erase(tree[j][k].outgoing_chargePoints.begin()+tree[j][k].incoming_checkPoints.size(), tree[j][k].outgoing_chargePoints.end());
		}
		else {
			tree[j][k].incoming_checkPoints.erase(tree[j][k].incoming_checkPoints.begin()+tree[j][k].outgoing_checkPoints.size(), tree[j][k].incoming_checkPoints.end());
			tree[j][k].incoming_chargePoints.erase(tree[j][k].incoming_chargePoints.begin()+tree[j][k].outgoing_checkPoints.size(), tree[j][k].incoming_chargePoints.end());
		}
		Mat temp1 = tree[j][k].Ac.block(0,0,tree[j][k].Ac.rows(), tree[j][k].incoming_checkPoints.size());
		tree[j][k].L2P = Mat(tree[j][k].Ac.rows(),tree[j][k].incoming_checkPoints.size());
		if (tree[j][k].incoming_checkPoints.size() > 0) {
			Mat D1 = K->getMatrix(tree[j][k].incoming_checkPoints, tree[j][k].incoming_chargePoints);
			Eigen::PartialPivLU<Mat> D1_lu = Eigen::PartialPivLU<Mat>(D1.adjoint());
			tree[j][k].L2P = D1_lu.solve(temp1.adjoint()).adjoint();
		}
		Mat temp2 = tree[j][k].Ar.block(0,0,tree[j][k].outgoing_checkPoints.size(),tree[j][k].Ar.cols());
		tree[j][k].P2M = Mat(tree[j][k].outgoing_checkPoints.size(),tree[j][k].Ar.cols());
		if (tree[j][k].outgoing_checkPoints.size() > 0) {
			Mat D2 = K->getMatrix(tree[j][k].outgoing_checkPoints, tree[j][k].outgoing_chargePoints);
			Eigen::PartialPivLU<Mat> D2_lu = Eigen::PartialPivLU<Mat>(D2);
			tree[j][k].P2M = D2_lu.solve(temp2);
		}
	}

	void get_L2P_P2M_level(int j) {
		#pragma omp parallel for
		for (int k=0; k<nBoxesPerLevel[j]; ++k) {
			get_L2P_P2M_box(j, k);
		}
	}
	
	void assign_particle_rhs_NonLeafLevel(int j) {
    	#pragma omp parallel for
    	for (size_t k = 0; k < nBoxesPerLevel[j]; k++) {
			int particle_rhs_length = 0;
			for (size_t c = 0; c < 4; c++) {
				// particle_rhs_length += tree[j+1][4*k+c].NumLocals;
        	particle_rhs_length += tree[j+1][4*k+c].multipole_rhs.size();
			}
			tree[j][k].particle_rhs = Vec(particle_rhs_length);
			tree[j][k].particle_rhs << tree[j+1][4*k+0].multipole_rhs, tree[j+1][4*k+1].multipole_rhs, tree[j+1][4*k+2].multipole_rhs, tree[j+1][4*k+3].multipole_rhs;
		}
	}

	void reorder(Vec &potential) {
		Vec potentialTemp = potential;
		int start = 0;
		for (size_t k = 0; k < nBoxesPerLevel[nLevels]; k++) {
			for (size_t i = 0; i < tree[nLevels][k].chargeLocations.size(); i++) {
				int index = tree[nLevels][k].chargeLocations[i];
				potential(index) = potentialTemp(start);
				start++;
			}
		}
	}
	/*********************Lets get the charges for the non leaf level through the pivots ***************************/
	void getNodes() {
		for (int j=nLevels; j>=1; j--) {
			getNodes_outgoing_level(j);
			getNodes_incoming_level(j);
			get_L2P_P2M_level(j);
		}
	}

	void getNodes_outgoing_level(int j) { //LFR; box interactions
    #pragma omp parallel for
		for (int k=0; k<nBoxesPerLevel[j]; ++k) {
			getNodes_outgoing_box(j, k);
		}
	}

	void getNodes_incoming_level(int j) { //LFR; box interactions
    #pragma omp parallel for
    for (int k=0; k<nBoxesPerLevel[j]; ++k) {
			getNodes_incoming_box(j, k);
		}
	}

	void getParticlesFromChildren_outgoing_row(int j, int k, std::vector<int>& searchNodes) {
		if (j==nLevels) {
			searchNodes.insert(searchNodes.end(), tree[j][k].chargeLocations.begin(), tree[j][k].chargeLocations.end());
		}
		else {
			int J = j+1;
			for (int c = 0; c < 4; c++) {
				searchNodes.insert(searchNodes.end(), tree[J][4*k+c].incoming_checkPoints.begin(), tree[J][4*k+c].incoming_checkPoints.end());
			}
		}
	}

	void getParticlesFromChildren_outgoing_col(int j, int k, std::vector<int>& searchNodes) {
		if (j==nLevels) {
			searchNodes.insert(searchNodes.end(), tree[j][k].chargeLocations.begin(), tree[j][k].chargeLocations.end());
		}
		else {
			int J = j+1;
			for (int c = 0; c < 4; c++) {
				searchNodes.insert(searchNodes.end(), tree[J][4*k+c].outgoing_chargePoints.begin(), tree[J][4*k+c].outgoing_chargePoints.end());
			}
		}
	}
    
	void getParticlesFromChildren_incoming_row(int j, int k, std::vector<int>& searchNodes) {
		if (j==nLevels) {
			searchNodes.insert(searchNodes.end(), tree[j][k].chargeLocations.begin(), tree[j][k].chargeLocations.end());
		}
		else {
			int J = j+1;
			for (int c = 0; c < 4; c++) {
				searchNodes.insert(searchNodes.end(), tree[J][4*k+c].incoming_checkPoints.begin(), tree[J][4*k+c].incoming_checkPoints.end());
			}
		}
	}

	void getParticlesFromChildren_incoming_col(int j, int k, std::vector<int>& searchNodes) {
		if (j==nLevels) {
			searchNodes.insert(searchNodes.end(), tree[j][k].chargeLocations.begin(), tree[j][k].chargeLocations.end());
		}
		else {
			int J = j+1;
			for (int c = 0; c < 4; c++) {
				searchNodes.insert(searchNodes.end(), tree[J][4*k+c].outgoing_chargePoints.begin(), tree[J][4*k+c].outgoing_chargePoints.end());
			}
		}
	}

	void getNodes_outgoing_box(int j, int k) {
		int n_rows, n_cols, ComputedRank;
        std::vector<int> boxA_Nodes;
		getParticlesFromChildren_outgoing_col(j, k, boxA_Nodes);
		std::vector<int> IL_Nodes;//indices
		for(int vn=0; vn<4; ++vn) {
			if(tree[j][k].vs_neighbor[vn] != -1) {
				int kIL = tree[j][k].vs_neighbor[vn];
				std::vector<int> chargeLocations;
				getParticlesFromChildren_outgoing_row(j, kIL, chargeLocations);
				IL_Nodes.insert(IL_Nodes.end(), chargeLocations.begin(), chargeLocations.end());
			}
		}
		for(int in=0; in<4; ++in) {
			if(tree[j][k].innerNumbers[in] != -1) {
				int kIL = tree[j][k].innerNumbers[in];
				std::vector<int> chargeLocations;
				getParticlesFromChildren_outgoing_row(j, kIL, chargeLocations);
				IL_Nodes.insert(IL_Nodes.end(), chargeLocations.begin(), chargeLocations.end());
			}
		}
		for(int on=0; on<8; ++on) {
			if(tree[j][k].outerNumbers[on] != -1) {
				int kIL = tree[j][k].outerNumbers[on];
				std::vector<int> chargeLocations;
				getParticlesFromChildren_outgoing_row(j, kIL, chargeLocations);
				IL_Nodes.insert(IL_Nodes.end(), chargeLocations.begin(), chargeLocations.end());
			}
		}
		n_rows = IL_Nodes.size();
		n_cols = boxA_Nodes.size();
		std::vector<int> row_indices, col_indices;
		row_indices = IL_Nodes;
		col_indices = boxA_Nodes;//object of base class G_LowRank
		std::vector<int> row_bases, col_bases;
		Mat Ac;
		LowRank* LR		=	new LowRank(K, pow(10,-TOL_POW), row_indices, col_indices);
		LR->rookPiv(row_bases, col_bases, ComputedRank, Ac, tree[j][k].Ar);
		// LR->ACA_only_nodes(row_bases, col_bases, ComputedRank, Ac, tree[j][k].Ar);

		int minN = n_rows;
		if (n_rows > n_cols) {
			minN = n_cols;
		}
		if(ComputedRank > 0) {
			for (int r = 0; r < row_bases.size(); r++) {
				tree[j][k].outgoing_checkPoints.push_back(IL_Nodes[row_bases[r]]);
			}
			for (int c = 0; c < col_bases.size(); c++) {
				tree[j][k].outgoing_chargePoints.push_back(boxA_Nodes[col_bases[c]]);
			}
		}
	}

	void getNodes_incoming_box(int j, int k) {
		int n_rows, n_cols, ComputedRank;
        std::vector<int> boxA_Nodes;
		getParticlesFromChildren_incoming_row(j, k, boxA_Nodes);
		std::vector<int> IL_Nodes;//indices

		for(int vn=0; vn<4; ++vn) {
			if(tree[j][k].vs_neighbor[vn] != -1) {
				int kIL = tree[j][k].vs_neighbor[vn];
				std::vector<int> chargeLocations;
				getParticlesFromChildren_incoming_col(j, kIL, chargeLocations);
				IL_Nodes.insert(IL_Nodes.end(), chargeLocations.begin(), chargeLocations.end());
			}
		}
		for(int in=0; in<4; ++in) {
			if(tree[j][k].innerNumbers[in] != -1) {
				int kIL = tree[j][k].innerNumbers[in];
				std::vector<int> chargeLocations;
				getParticlesFromChildren_incoming_col(j, kIL, chargeLocations);
				IL_Nodes.insert(IL_Nodes.end(), chargeLocations.begin(), chargeLocations.end());
			}
		}
		for(int on=0; on<8; ++on) {
			if(tree[j][k].outerNumbers[on] != -1) {
				int kIL = tree[j][k].outerNumbers[on];
				std::vector<int> chargeLocations;
				getParticlesFromChildren_incoming_col(j, kIL, chargeLocations);
				IL_Nodes.insert(IL_Nodes.end(), chargeLocations.begin(), chargeLocations.end());
			}
		}
		n_rows = boxA_Nodes.size();
		n_cols = IL_Nodes.size();
		std::vector<int> row_indices, col_indices;
		row_indices = boxA_Nodes;//object of base class G_LowRank
		col_indices = IL_Nodes;
		std::vector<int> row_bases, col_bases;
		Mat Ar;
		LowRank* LR		=	new LowRank(K, pow(10,-TOL_POW), row_indices, col_indices);
		LR->rookPiv(row_bases, col_bases, ComputedRank, tree[j][k].Ac, Ar);
		// LR->ACA_only_nodes(row_bases, col_bases, ComputedRank, tree[j][k].Ac, Ar);

		int minN = n_rows;
		if (n_rows > n_cols) {
			minN = n_cols;
		}
		if(ComputedRank > 0) {
			for (int r = 0; r < row_bases.size(); r++) {
				tree[j][k].incoming_checkPoints.push_back(boxA_Nodes[row_bases[r]]);
			}
			for (int c = 0; c < col_bases.size(); c++) {
				tree[j][k].incoming_chargePoints.push_back(IL_Nodes[col_bases[c]]);
			}
		}
	}

    int getMaxRank() {
		int max = 0;
		for (int j = nLevels; j >= 1; j--) {
			for (size_t k = 0; k < nBoxesPerLevel[j]; k++) {
				if (max < tree[j][k].multipoles.size()) {
					max = tree[j][k].multipoles.size();
				}
			}
		}
		return max;
	}
	
	int climber(int j,int k,int b,int count){//to count how many levels a P2P interaction climbing
		int pk=tree[j][k].parentNumber;
		int pb=tree[j][b].parentNumber;
		j-=1;
		if(is_vertex_sharing(j,pk,pb)){
			count+=1;
			return climber(j,pk,pb,count);
			
		}
        return count;
	}
	
	void assemble_M2L() {//To get the M2L operator 
		// #pragma omp parallel for
		for (size_t j = 1; j <= nLevels; j++) {
			#pragma omp parallel for
			for (size_t k = 0; k < nBoxesPerLevel[j]; k++) {
				// #pragma omp parallel for
				for(int vn=0; vn<4; ++vn) {
					if(tree[j][k].vs_neighbor[vn] != -1) {
						int kIL = tree[j][k].vs_neighbor[vn];
						tree[j][k].M2L[kIL] = K->getMatrix(tree[j][k].incoming_checkPoints, tree[j][kIL].outgoing_chargePoints);
					}
				}
				for(int in=0; in<4; ++in) {
					if(tree[j][k].innerNumbers[in] != -1) {
						int kIL = tree[j][k].innerNumbers[in];
						tree[j][k].M2L[kIL] = K->getMatrix(tree[j][k].incoming_checkPoints, tree[j][kIL].outgoing_chargePoints);
					}
				}
				// #pragma omp parallel for
				for(int on=0; on<8; ++on) {
					if(tree[j][k].outerNumbers[on] != -1) {
						int kIL = tree[j][k].outerNumbers[on];
						tree[j][k].M2L[kIL] = K->getMatrix(tree[j][k].incoming_checkPoints, tree[j][kIL].outgoing_chargePoints);
					}
				}
			}
		}
	}
   
	/**************************************Initialisation*************************************************/
	void initialise_P2P_Leaf_Level() {
		#pragma omp parallel for
		for (int k=0; k<nBoxesPerLevel[nLevels]; ++k) {
			// #pragma omp parallel for
			for(int en=0; en<4; ++en) {
				int nn = tree[nLevels][k].es_neighbor[en];//P2P_{k,nn}
				if(nn != -1) {
					tree[nLevels][k].P2P[nn] = K->getMatrix(tree[nLevels][k].chargeLocations, tree[nLevels][nn].chargeLocations);
				}
			}
			tree[nLevels][k].P2P[k] = K->getMatrix(tree[nLevels][k].chargeLocations, tree[nLevels][k].chargeLocations);//self interaction
		}
	}

	void initialise_rhs() {
		// #pragma omp parallel for
		for (size_t j = 1; j <= nLevels; j++) {
			#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				tree[j][k].local_rhs = Vec::Zero(tree[j][k].NumMultipoles);
				tree[j][k].multipole_rhs = Vec::Zero(tree[j][k].NumLocals);
			}
		}
	}

	void initialise_phase() {
		// #pragma omp parallel for
		for (size_t j = 1; j <= nLevels; j++) {
			#pragma omp parallel for
			for (size_t k = 0; k < nBoxesPerLevel[j]; k++) {
				tree[j][k].NumMultipoles = tree[j][k].P2M.rows();
				tree[j][k].NumLocals = tree[j][k].L2P.cols();
         
			}
		}
		initialise_rhs();//initialise local_rhs and multipole_rhs
	}
	
    /****************************************************************************************************/
	bool is_well_separated(int j, int k, int l) {
		if(k==l) { //self
			return false;
		}
		for (size_t i = 0; i < 4; i++) {
			int nn = tree[j][k].es_neighbor[i];
			if(l==nn) {
				return false;
			}
		}
		return true;
	}
	bool is_vertex_sharing(int j, int n,int m){
		if(m == n){
			return false;
		}
		for(size_t i=0;i<4;i++){
			int vn=tree[j][n].vs_neighbor[i];
			if(m==vn){
				return true;
			}
		}
		return false;
	}
    /*************************************Fill-ins ******************************************************/
    void filings_in_equation_P2M_due_to_x(int j, int k, Eigen::PartialPivLU<Mat>& P2P_self_QR) {         
		filing_in_equation_P2M_due_to_L2P(j,k,P2P_self_QR);//U fill-in/update; in P2P(j,n) equation
		filing_in_equation_P2M_due_to_P2P_neighbors(j,k,P2P_self_QR);//P2P_neighbor fill-in/update; in P2P(j,n) equation
    }
    void filing_in_equation_P2M_due_to_L2P(int j, int k, Eigen::PartialPivLU<Mat>& P2P_self_QR) {
		tree[j][k].L2M_f = -tree[j][k].P2M * P2P_self_QR.solve(tree[j][k].L2P);
	}
    void filing_in_equation_P2M_due_to_P2P_neighbors(int j, int k, Eigen::PartialPivLU<Mat>& P2P_self_QR) {
		for (size_t n = 0; n < 4; n++) {//neighbors of k
			int nn = tree[j][k].es_neighbor[n];//P2P(j,nn)
			if (nn != -1) {
				if (!tree[j][nn].Eliminated) {
					tree[j][k].P2M_f[nn] = -tree[j][k].P2M * P2P_self_QR.solve(tree[j][k].P2P[nn]);
				}
				else {
					tree[j][k].M2M_f[nn] = -tree[j][k].P2M * P2P_self_QR.solve(tree[j][k].M2P[nn]);
				}
			}
		}
	}

    void filings_in_equation_P2P_due_to_x(int j, int k, int nn, Eigen::PartialPivLU<Mat>& P2P_self_QR) {
		filing_due_to_L2P(j,k,nn,P2P_self_QR);//U fill-in/update; in P2P(j,n) equation
		filing_due_to_P2P_neighbors(j,k,nn,P2P_self_QR);//P2P_neighbor fill-in/update; in P2P(j,n) equation
	}
	void filing_due_to_L2P(int j, int k, int nn, Eigen::PartialPivLU<Mat>& P2P_self_QR) {//in P2P(j,nn) equation; nn:eq number
		if (!tree[j][nn].Eliminated) {
			tree[j][nn].L2P_f[k] = -tree[j][nn].P2P[k] * P2P_self_QR.solve(tree[j][k].L2P); //temporary; this gets eliminated when z is eliminated
		}
		else {
			tree[j][nn].L2P_f[k] = -tree[j][nn].P2L[k] * P2P_self_QR.solve(tree[j][k].L2P);
		}// 0-K21 K11^{-1} U1
	}
	void filing_due_to_P2P_neighbors(int j, int k, int nn, Eigen::PartialPivLU<Mat>& P2P_self_QR) {//in P2P(j,nn) equation
		// Eq: P2P_n
		// workout for fill-ins due to all neighbors of (j,k)
		for (size_t p = 0; p < 4; p++) { //loop over all neighbors of (j,k)
			int pp = tree[j][k].es_neighbor[p];
			if (!tree[j][nn].Eliminated) {
				if (pp != -1) {
					if (!tree[j][pp].Eliminated) {
						if (tree[j][nn].P2P[pp].size() == 0) {
							tree[j][nn].P2P[pp] = -tree[j][nn].P2P[k] * P2P_self_QR.solve(tree[j][k].P2P[pp]);
						}
						else {
							tree[j][nn].P2P[pp] = tree[j][nn].P2P[pp] - tree[j][nn].P2P[k] * P2P_self_QR.solve(tree[j][k].P2P[pp]);
						}
						if(is_well_separated(j,nn,pp)) {
							if (nn < pp) { // using symmetry
								if(tree[j][nn].P2P[pp].norm() > 1e-10) {
									std::pair<int, int> g(nn,pp);
									P2P.push_back(g);
								}
							}
						}
					}
					else {
						if (tree[j][nn].M2P[pp].size() == 0) {
							tree[j][nn].M2P[pp] = -tree[j][nn].P2P[k] * P2P_self_QR.solve(tree[j][k].M2P[pp]);
						}
						else {
							tree[j][nn].M2P[pp] = tree[j][nn].M2P[pp] - tree[j][nn].P2P[k] * P2P_self_QR.solve(tree[j][k].M2P[pp]);
						}
					}
				}
			}
			else {
				if (pp != -1) {
					if (!tree[j][pp].Eliminated) {
						if (tree[j][nn].P2L[pp].size() == 0) {
							tree[j][nn].P2L[pp] = -tree[j][nn].P2L[k] * P2P_self_QR.solve(tree[j][k].P2P[pp]);
						}
						else {
							tree[j][nn].P2L[pp] = tree[j][nn].P2L[pp] - tree[j][nn].P2L[k] * P2P_self_QR.solve(tree[j][k].P2P[pp]);
						}
						if(is_well_separated(j,nn,pp)) {
							if (tree[j][nn].P2L[pp].norm() > 1e-10) {
								std::pair<int, int> g(nn,pp);
								P2L_M2P.push_back(g);
							}
						}
					}
					else {
						tree[j][nn].M2L[pp] = tree[j][nn].M2L[pp] - tree[j][nn].P2L[k] * P2P_self_QR.solve(tree[j][k].M2P[pp]);
					}
				}
			}
		}
	}

    void filings_in_equation_P2M_due_to_x(int j, int k) {
	  filing_in_equation_P2M_due_to_L2P(j,k);//U fill-in/update; in P2P(j,n) equation
	  filing_in_equation_P2M_due_to_P2P_neighbors(j,k);//P2P_neighbor fill-in/update; in P2P(j,n) equation
    }
	void filing_in_equation_P2M_due_to_L2P(int j, int k) {
	     tree[j][k].L2M_f = Mat(tree[j][k].P2M.rows(),tree[j][k].L2P.cols());//-tree[j][k].P2M * P2P_self_QR.solve(tree[j][k].L2P);
    }
    void filing_in_equation_P2M_due_to_P2P_neighbors(int j, int k) {
	   for (size_t n = 0; n < 4; n++) {//neighbors of k
		    int nn = tree[j][k].es_neighbor[n];//P2P(j,nn)
		    if (nn != -1) {
			 if (!tree[j][nn].Eliminated) {
				tree[j][k].P2M_f[nn] = Mat(tree[j][k].P2M.rows(),tree[j][k].P2P[nn].cols());//-tree[j][k].P2M * P2P_self_QR.solve(tree[j][k].P2P[nn]);
			  }
			 else {
				tree[j][k].M2M_f[nn] = Mat(tree[j][k].P2M.rows(),tree[j][k].M2P[nn].cols());//-tree[j][k].P2M * P2P_self_QR.solve(tree[j][k].M2P[nn]);
			 }
		    }
	    }
     }
	
	void filings_in_equation_P2P_due_to_x(int j, int k, int nn) {
	    filing_due_to_L2P(j,k,nn);//U fill-in/update; in P2P(j,n) equation
	    filing_due_to_P2P_neighbors(j,k,nn);//P2P_neighbor fill-in/update; in P2P(j,n) equation
    }
	void filing_due_to_L2P(int j, int k, int nn) {//in P2P(j,nn) equation; nn:eq number
	if (!tree[j][nn].Eliminated) {
		tree[j][nn].L2P_f[k] = Mat(tree[j][nn].P2P[k].rows(),tree[j][k].L2P.cols());//-tree[j][nn].P2P[k] * P2P_self_QR.solve(tree[j][k].L2P); //temporary; this gets eliminated when z is eliminated
	}
	else {
		tree[j][nn].L2P_f[k] = Mat(tree[j][nn].P2L[k].rows(),tree[j][k].L2P.cols());//-tree[j][nn].P2L[k] * P2P_self_QR.solve(tree[j][k].L2P);
	}// 0-K21 K11^{-1} U1
    }
    void filing_due_to_P2P_neighbors(int j, int k, int nn) {//in P2P(j,nn) equation
	// Eq: P2P_n
	// workout for fill-ins due to all neighbors of (j,k)
	for (size_t p = 0; p < 4; p++) { //loop over all neighbors of (j,k)
		int pp = tree[j][k].es_neighbor[p];
		if (!tree[j][nn].Eliminated) {
			if (pp != -1) {
				if (!tree[j][pp].Eliminated) {
					if (tree[j][nn].P2P[pp].size() == 0) {
						tree[j][nn].P2P[pp] = Mat::Zero(tree[j][nn].P2P[k].rows(),tree[j][k].P2P[pp].cols());//-tree[j][nn].P2P[k] * P2P_self_QR.solve(tree[j][k].P2P[pp]);
					}
					else {
						tree[j][nn].P2P[pp] = tree[j][nn].P2P[pp];// - tree[j][nn].P2P[k] * P2P_self_QR.solve(tree[j][k].P2P[pp]);
					}
					if(is_well_separated(j,nn,pp)) {
						if (nn < pp) { // using symmetry
							if(tree[j][nn].P2P[pp].norm() > 1e-10) {
								std::pair<int, int> g(nn,pp);
								P2P.push_back(g);
							}
						}
					}
				}
				else {
					if (tree[j][nn].M2P[pp].size() == 0) {
						tree[j][nn].M2P[pp] = Mat(tree[j][nn].P2P[k].rows(),tree[j][k].M2P[pp].cols());//-tree[j][nn].P2P[k] * P2P_self_QR.solve(tree[j][k].M2P[pp]);
					}
					else {
						tree[j][nn].M2P[pp] = tree[j][nn].M2P[pp];// - tree[j][nn].P2P[k] * P2P_self_QR.solve(tree[j][k].M2P[pp]);
					}
				}
			}
		}
		else {
			if (pp != -1) {
				if (!tree[j][pp].Eliminated) {
					if (tree[j][nn].P2L[pp].size() == 0) {
						tree[j][nn].P2L[pp] = Mat(tree[j][nn].P2L[k].rows(),tree[j][k].P2P[pp].cols());//-tree[j][nn].P2L[k] * P2P_self_QR.solve(tree[j][k].P2P[pp]);
					}
					else {
						tree[j][nn].P2L[pp] = tree[j][nn].P2L[pp];// - tree[j][nn].P2L[k] * P2P_self_QR.solve(tree[j][k].P2P[pp]);
					}
					if(is_well_separated(j,nn,pp)) {
						if (tree[j][nn].P2L[pp].norm() > 1e-10) {
							std::pair<int, int> g(nn,pp);
							P2L_M2P.push_back(g);
						}
					}
				}
				else {
					tree[j][nn].M2L[pp] = tree[j][nn].M2L[pp];// - tree[j][nn].P2L[k] * P2P_self_QR.solve(tree[j][k].M2P[pp]);
				}
			}
		}
	}
}

    void filings_in_equation_M2L_due_to_z(int j, int k) {
		filings_in_equation_M2L_due_to_y(j,k);
		filings_in_equation_M2L_due_to_neighbors(j,k);
	}
	void filings_in_equation_M2L_due_to_y(int j, int k) {
		tree[j][k].M2L[k] = Mat(0,tree[j][k].NumMultipoles);//L2M_f_QR.solve(-Mat::Identity(tree[j][k].NumMultipoles, tree[j][k].NumMultipoles));
	}
	void filings_in_equation_M2L_due_to_neighbors(int j, int k) {
		for (size_t n = 0; n < 4; n++) {
			int nn = tree[j][k].es_neighbor[n];//P2P(j,nn)
			if(nn != -1) {
				if (!tree[j][nn].Eliminated) {
					tree[j][k].P2L[nn] = Mat(0,tree[j][k].P2M_f[nn].cols());//L2M_f_QR.solve(tree[j][k].P2M_f[nn]);
					if(is_well_separated(j,k,nn)) {
						if (tree[j][k].P2L[nn].norm() > 1e-10) {
							std::pair<int, int> g(k,nn);
							P2L_M2P.push_back(g);
							// compress_P2L(j,k,nn);
						}
					}
				}
				else {
					tree[j][k].M2L[nn] = Mat(0,tree[j][k].M2M_f[nn].cols());//L2M_f_QR.solve(tree[j][k].M2M_f[nn]);
				}
			}
		}
	}

    void filings_in_equation_M2L_due_to_z(int j, int k, Eigen::PartialPivLU<Mat>& L2M_f_QR) {
		filings_in_equation_M2L_due_to_y(j,k,L2M_f_QR);
		filings_in_equation_M2L_due_to_neighbors(j,k,L2M_f_QR);
	}
    void filings_in_equation_M2L_due_to_y(int j, int k, Eigen::PartialPivLU<Mat>& L2M_f_QR) {
		tree[j][k].M2L[k] = L2M_f_QR.solve(-Mat::Identity(tree[j][k].NumMultipoles, tree[j][k].NumMultipoles));
	}
	void filings_in_equation_M2L_due_to_neighbors(int j, int k, Eigen::PartialPivLU<Mat>& L2M_f_QR) {
		for (size_t n = 0; n < 4; n++) {
			int nn = tree[j][k].es_neighbor[n];//P2P(j,nn)
			if(nn != -1) {
				if (!tree[j][nn].Eliminated) {
					tree[j][k].P2L[nn] = L2M_f_QR.solve(tree[j][k].P2M_f[nn]);
					if(is_well_separated(j,k,nn)) {
						if (tree[j][k].P2L[nn].norm() > 1e-10) {
							std::pair<int, int> g(k,nn);
							P2L_M2P.push_back(g);
							// compress_P2L(j,k,nn);
						}
					}
				}
				else {
					tree[j][k].M2L[nn] = L2M_f_QR.solve(tree[j][k].M2M_f[nn]);
				}
			}
		}
	}

    void filings_in_equation_P2P_due_to_z(int j, int k, int nn, Eigen::PartialPivLU<Mat>& L2M_f_QR) {
		filings_in_equation_P2P_due_to_y(j,k,nn,L2M_f_QR);
		filings_in_equation_P2P_due_to_neighbors(j,k,nn,L2M_f_QR);
	}
    void filings_in_equation_P2P_due_to_neighbors(int j, int k, int nn, Eigen::PartialPivLU<Mat>& L2M_f_QR) {
		// nn: equation number
		// fill-ins between neighbors of k
		for (size_t p = 0; p < 4; p++) {
			int pp = tree[j][k].es_neighbor[p];//P2P(j,nn)
			if (!tree[j][nn].Eliminated) {
				if(pp != -1) {
					if (!tree[j][pp].Eliminated) { //case 4
						tree[j][nn].P2P[pp] = tree[j][nn].P2P[pp] - tree[j][nn].L2P_f[k] * L2M_f_QR.solve(tree[j][k].P2M_f[pp]);
						if(is_well_separated(j,nn,pp)) {
							if (nn < pp) { // using symmetry
								if(tree[j][nn].P2P[pp].norm() > 1e-10) {
									// compress_P2P(j,nn,pp);
									std::pair<int, int> g(nn,pp);
									P2P.push_back(g);
								}
							}
							// tree[j][nn].P2P[pp] = Mat(0,0);
						}
					}
					else { //case 2/3
						tree[j][nn].M2P[pp] = tree[j][nn].M2P[pp] - tree[j][nn].L2P_f[k] * L2M_f_QR.solve(tree[j][k].M2M_f[pp]);
						if(is_well_separated(j,nn,pp)) {
							if (tree[j][nn].M2P[pp].norm() > 1e-10) {
								// compress_M2P(j,nn,pp);
							}
						}
					}
				}
			}
			else {
				if(pp != -1) {
					if (!tree[j][pp].Eliminated) { //case 2/3
						tree[j][nn].P2L[pp] = tree[j][nn].P2L[pp] - tree[j][nn].L2P_f[k] * L2M_f_QR.solve(tree[j][k].P2M_f[pp]);
						if(is_well_separated(j,nn,pp)) {
							if (tree[j][nn].P2L[pp].norm() > 1e-10) {
								// compress_P2L(j,nn,pp);
								std::pair<int, int> g(nn,pp);
								P2L_M2P.push_back(g);
							}
							// tree[j][nn].P2L[pp] = Mat(0,0);
						}
					}
					else { // case 1
						tree[j][nn].M2L[pp] = tree[j][nn].M2L[pp] - tree[j][nn].L2P_f[k] * L2M_f_QR.solve(tree[j][k].M2M_f[pp]);
					}
				}
			}
		}
	}
	void filings_in_equation_P2P_due_to_y(int j, int k, int nn, Eigen::PartialPivLU<Mat>& L2M_f_QR) {
		// fill-ins between k and its neighbors
		if (!tree[j][nn].Eliminated) { //case 2/3
			if(tree[j][nn].M2P[k].size() == 0) {
				tree[j][nn].M2P[k] = - tree[j][nn].L2P_f[k] * L2M_f_QR.solve(-Mat::Identity(tree[j][k].NumMultipoles, tree[j][k].NumMultipoles));
			}
			else {
				tree[j][nn].M2P[k] = tree[j][nn].M2P[k] - tree[j][nn].L2P_f[k] * L2M_f_QR.solve(-Mat::Identity(tree[j][k].NumMultipoles, tree[j][k].NumMultipoles));
			}
			if(is_well_separated(j,nn,k)) {
				if (tree[j][nn].M2P[k].norm() > 1e-10) {
					// compress_M2P(j,nn,k);
				}
				// tree[j][nn].M2P[k] = Mat(0,0);
			}
		}
		else { //case 1
			if (tree[j][nn].M2L[k].size() == 0) {
				tree[j][nn].M2L[k] = - tree[j][nn].L2P_f[k] * L2M_f_QR.solve(-Mat::Identity(tree[j][k].NumMultipoles, tree[j][k].NumMultipoles));
			}
			else {
				tree[j][nn].M2L[k] = tree[j][nn].M2L[k] - tree[j][nn].L2P_f[k] * L2M_f_QR.solve(-Mat::Identity(tree[j][k].NumMultipoles, tree[j][k].NumMultipoles));
			}
		}
	}

	void filings_in_equation_P2P_due_to_z(int j, int k, int nn) {
		filings_in_equation_P2P_due_to_y(j,k,nn);
		filings_in_equation_P2P_due_to_neighbors(j,k,nn);
	}
    void filings_in_equation_P2P_due_to_neighbors(int j, int k, int nn) {
		// nn: equation number
		// fill-ins between neighbors of k
		for (size_t p = 0; p < 4; p++) {
			int pp = tree[j][k].es_neighbor[p];//P2P(j,nn)
			if (!tree[j][nn].Eliminated) {
				if(pp != -1) {
					if (!tree[j][pp].Eliminated) { //case 4
						tree[j][nn].P2P[pp] = tree[j][nn].P2P[pp];// - tree[j][nn].L2P_f * L2M_f_QR.solve(tree[j][k].P2M_f[pp]);
						if(is_well_separated(j,nn,pp)) {
							if (nn < pp) { // using symmetry
								if(tree[j][nn].P2P[pp].norm() > 1e-10) {
									// compress_P2P(j,nn,pp);
									std::pair<int, int> g(nn,pp);
									P2P.push_back(g);
								}
							}
							// tree[j][nn].P2P[pp] = Mat(0,0);
						}
					}
					else { //case 2/3
						tree[j][nn].M2P[pp] = tree[j][nn].M2P[pp];// - tree[j][nn].L2P_f * L2M_f_QR.solve(tree[j][k].M2M_f[pp]);
						if(is_well_separated(j,nn,pp)) {
							if (tree[j][nn].M2P[pp].norm() > 1e-10) {
								// compress_M2P(j,nn,pp);
							}
							// tree[j][nn].M2P[pp] = Mat(0,0);
						}
					}
				}
			}
			else {
				if(pp != -1) {
					if (!tree[j][pp].Eliminated) { //case 2/3
						tree[j][nn].P2L[pp] = tree[j][nn].P2L[pp];// - tree[j][nn].L2P_f * L2M_f_QR.solve(tree[j][k].P2M_f[pp]);
						if(is_well_separated(j,nn,pp)) {
							if (tree[j][nn].P2L[pp].norm() > 1e-10) {
								// compress_P2L(j,nn,pp);
								std::pair<int, int> g(nn,pp);
								P2L_M2P.push_back(g);
							}
							// tree[j][nn].P2L[pp] = Mat(0,0);
						}
					}
					else { // case 1
						tree[j][nn].M2L[pp] = tree[j][nn].M2L[pp];// - tree[j][nn].L2P_f * L2M_f_QR.solve(tree[j][k].M2M_f[pp]);
					}
				}
			}
		}
	}
	void filings_in_equation_P2P_due_to_y(int j, int k, int nn) {
		// fill-ins between k and its neighbors
		if (!tree[j][nn].Eliminated) { //case 2/3
			if(tree[j][nn].M2P[k].size() == 0) {
				tree[j][nn].M2P[k] = Mat(tree[j][nn].L2P_f[k].rows(),tree[j][k].NumMultipoles);//- tree[j][nn].L2P_f * L2M_f_QR.solve(-Mat::Identity(tree[j][k].NumMultipoles, tree[j][k].NumMultipoles));
			}
			else {
				tree[j][nn].M2P[k] = tree[j][nn].M2P[k];// - tree[j][nn].L2P_f * L2M_f_QR.solve(-Mat::Identity(tree[j][k].NumMultipoles, tree[j][k].NumMultipoles));
			}
			if(is_well_separated(j,nn,k)) {
				if (tree[j][nn].M2P[k].norm() > 1e-10) {
				}
			}
		}
		else { //case 1
			if (tree[j][nn].M2L[k].size() == 0) {
				tree[j][nn].M2L[k] = Mat(tree[j][nn].L2P_f[k].rows(), tree[j][k].NumMultipoles);//- tree[j][nn].L2P_f * L2M_f_QR.solve(-Mat::Identity(tree[j][k].NumMultipoles, tree[j][k].NumMultipoles));
			}
			else {
				tree[j][nn].M2L[k] = tree[j][nn].M2L[k];// - tree[j][nn].L2P_f * L2M_f_QR.solve(-Mat::Identity(tree[j][k].NumMultipoles, tree[j][k].NumMultipoles));
			}
		}
	}

	void rhs_filings_in_equation_P2M_due_to_x(int j, int k, Eigen::PartialPivLU<Mat>& P2P_self_QR) {
 	    tree[j][k].local_rhs = - tree[j][k].P2M * P2P_self_QR.solve(tree[j][k].particle_rhs);
	}
	void rhs_filings_in_equation_P2P_due_to_x(int j, int k, int nn, Eigen::PartialPivLU<Mat>& P2P_self_QR) {
		if (!tree[j][nn].Eliminated) {
			tree[j][nn].particle_rhs = tree[j][nn].particle_rhs - tree[j][nn].P2P[k] * P2P_self_QR.solve(tree[j][k].particle_rhs);
		}
		else {
			tree[j][nn].multipole_rhs = tree[j][nn].multipole_rhs - tree[j][nn].P2L[k] * P2P_self_QR.solve(tree[j][k].particle_rhs);
		}
	}

	void rhs_filings_in_equation_P2M_due_to_x(int j, int k){
		tree[j][k].local_rhs = Vec(tree[j][k].P2M.rows());
	}
	void rhs_filings_in_equation_P2P_due_to_x(int j, int k,int nn){
		if (!tree[j][nn].Eliminated) {
			tree[j][nn].particle_rhs = tree[j][nn].particle_rhs;// - tree[j][nn].P2P[k] * P2P_self_QR.solve(tree[j][k].particle_rhs);
		}
		else {
			tree[j][nn].multipole_rhs = tree[j][nn].multipole_rhs;// - tree[j][nn].P2L[k] * P2P_self_QR.solve(tree[j][k].particle_rhs);
		}
	}
	
 	void rhs_filings_in_equation_M2L_due_to_z(int j, int k, Eigen::PartialPivLU<Mat>& L2M_f_QR) {
		tree[j][k].multipole_rhs = L2M_f_QR.solve(tree[j][k].local_rhs);
	}
	void rhs_filings_in_equation_M2L_due_to_z(int j, int k) {
		tree[j][k].multipole_rhs = Vec(0);
	}
	
	void rhs_filings_in_equation_P2P_due_to_z(int j, int k, int nn, Eigen::PartialPivLU<Mat>& L2M_f_QR) {
		if (!tree[j][nn].Eliminated) {
			tree[j][nn].particle_rhs = tree[j][nn].particle_rhs - tree[j][nn].L2P_f[k] * L2M_f_QR.solve(tree[j][k].local_rhs);
		}
		else {
			tree[j][nn].multipole_rhs = tree[j][nn].multipole_rhs - tree[j][nn].L2P_f[k] * L2M_f_QR.solve(tree[j][k].local_rhs);
		}
	}
	void rhs_filings_in_equation_P2P_due_to_z(int j, int k, int nn) {
		if (!tree[j][nn].Eliminated) {
			tree[j][nn].particle_rhs = tree[j][nn].particle_rhs;// - tree[j][nn].L2P_f * L2M_f_QR.solve(tree[j][k].local_rhs);
		}
		else {
			tree[j][nn].multipole_rhs = tree[j][nn].multipole_rhs;// - tree[j][nn].L2P_f * L2M_f_QR.solve(tree[j][k].local_rhs);
		}
	}

	/***************************************Compression **************************************************/
    void compress_P2P_Two_Ways(int j, int k, int b) {
			Mat new_L2P_k; 
			Mat new_P2M_transpose_b;
			Mat new_L2P_k_modified;
			Mat new_P2M_transpose_b_modified;
			Mat M2L_k_b;
        
			compress_P2P(j, k, b, new_L2P_k, new_P2M_transpose_b, M2L_k_b);
           
			Mat new_L2P_b;
			Mat new_P2M_transpose_k;
			Mat new_L2P_b_modified;
			Mat new_P2M_transpose_k_modified;
			Mat M2L_b_k;
			compress_P2P(j, b, k, new_L2P_b, new_P2M_transpose_k, M2L_b_k);
			if (new_L2P_k.cols() < new_P2M_transpose_k.cols()) {
				new_P2M_transpose_k_modified = new_P2M_transpose_k.block(0,0,new_P2M_transpose_k.rows(),new_L2P_k.cols());
				new_L2P_k_modified = new_L2P_k;
			}
			else {
				new_L2P_k_modified = new_L2P_k.block(0,0,new_L2P_k.rows(),new_P2M_transpose_k.cols());
				new_P2M_transpose_k_modified = new_P2M_transpose_k;
			}

			if (new_L2P_b.cols() < new_P2M_transpose_b.cols()) {
				new_P2M_transpose_b_modified = new_P2M_transpose_b.block(0,0,new_P2M_transpose_b.rows(),new_L2P_b.cols());
				new_L2P_b_modified = new_L2P_b;
			}
			else {
				new_L2P_b_modified = new_L2P_b.block(0,0,new_L2P_b.rows(),new_P2M_transpose_b.cols());
				new_P2M_transpose_b_modified = new_P2M_transpose_b;
			}
			tree[j][k].M2L[b] = M2L_k_b.block(0,0,new_L2P_k_modified.cols(),new_P2M_transpose_b_modified.cols());
			    std::cout<< "no issues"<<std::endl;
			update_other_M2Ls_at_k(j, k, b, new_L2P_k_modified);
			update_other_M2Ls_at_b(j, k, b, new_P2M_transpose_b_modified);
			if(j>2) {
				update_parent_L2P(j, k, new_L2P_k_modified);
				update_parent_P2M(j, b, new_P2M_transpose_b_modified);
			}

			tree[j][k].L2P = new_L2P_k_modified;
			tree[j][b].P2M = new_P2M_transpose_b_modified.adjoint();

			tree[j][k].NumLocals = tree[j][k].L2P.cols();
			tree[j][b].NumMultipoles = tree[j][b].P2M.rows();

			tree[j][b].M2L[k] = M2L_b_k.block(0,0,new_L2P_b_modified.cols(),new_P2M_transpose_k_modified.cols());
			update_other_M2Ls_at_k(j, b, k, new_L2P_b_modified);
			update_other_M2Ls_at_b(j, b, k, new_P2M_transpose_k_modified);

			if(j>2) {
				update_parent_L2P(j, b, new_L2P_b_modified);
				update_parent_P2M(j, k, new_P2M_transpose_k_modified);
			}

			tree[j][b].L2P = new_L2P_b_modified;
			tree[j][k].P2M = new_P2M_transpose_k_modified.adjoint();

			tree[j][b].NumLocals = tree[j][b].L2P.cols();
			tree[j][k].NumMultipoles = tree[j][k].P2M.rows();
		}

	void compress_M2P_P2L(int j, int k, int b) {
			Mat new_L2P_k;
			Mat new_L2P_k_modified;
			Mat M2L_k_b;
			compress_M2P(j, k, b, new_L2P_k, M2L_k_b);
			Mat new_P2M_transpose_k;
			Mat new_P2M_transpose_k_modified;
			Mat M2L_b_k;
			compress_P2L(j, b, k, new_P2M_transpose_k, M2L_b_k);

			if (new_L2P_k.cols() < new_P2M_transpose_k.cols()) {
				new_P2M_transpose_k_modified = new_P2M_transpose_k.block(0,0,new_P2M_transpose_k.rows(),new_L2P_k.cols());
				new_L2P_k_modified = new_L2P_k;
			}
			else {
				new_L2P_k_modified = new_L2P_k.block(0,0,new_L2P_k.rows(),new_P2M_transpose_k.cols());
				new_P2M_transpose_k_modified = new_P2M_transpose_k;
			}

			tree[j][k].M2L[b] = M2L_k_b.block(0,0,new_L2P_k_modified.cols(),M2L_k_b.cols());
			update_other_M2Ls_at_k(j, k, b, new_L2P_k_modified);
			if(j>2) {
				update_parent_L2P(j, k, new_L2P_k_modified);
			}
			tree[j][k].L2P = new_L2P_k_modified;
			tree[j][k].NumLocals = tree[j][k].L2P.cols();

			///////////////////////////////////////////////////////
			tree[j][b].M2L[k] = M2L_b_k.block(0,0,M2L_b_k.rows(),new_P2M_transpose_k_modified.cols());
			update_other_M2Ls_at_b(j, b, k, new_P2M_transpose_k_modified);
			if(j>2) {
				update_parent_P2M(j, k, new_P2M_transpose_k_modified);
			}
			tree[j][k].P2M = new_P2M_transpose_k_modified.adjoint();
			tree[j][k].NumMultipoles = tree[j][k].P2M.rows();
		}

	void compress_P2P(int j, int k, int b, Mat &new_L2P, Mat &new_P2M_transpose, Mat &M2L) {
			Mat A_col(tree[j][k].L2P.rows(), tree[j][k].L2P.cols()+tree[j][k].P2P[b].cols());
			A_col << tree[j][k].L2P, tree[j][k].P2P[b];
			Eigen::ColPivHouseholderQR<Mat> qr_A_col(A_col.rows(), A_col.cols());
			qr_A_col.setThreshold(RRQR_threshold);
			qr_A_col.compute(A_col);
			Mat new_L2P_old = qr_A_col.householderQ() ; //new tree[j][k].L2P
			Mat R_col = qr_A_col.matrixR().triangularView<Eigen::Upper>();
			new_L2P = new_L2P_old.block(0,0,new_L2P_old.rows(),qr_A_col.rank());
			R_col = new_L2P.adjoint()*A_col;

			Mat Rk_prime = new_L2P.adjoint()*tree[j][k].P2P[b];
			Mat Rk = new_L2P.adjoint()*tree[j][k].L2P;

			Mat A_row(tree[j][b].P2M.cols(), tree[j][b].P2M.rows()+Rk_prime.rows());
			A_row << tree[j][b].P2M.adjoint(), Rk_prime.adjoint();
			Eigen::ColPivHouseholderQR<Mat> qr_A_row(A_row.rows(), A_row.cols());
			qr_A_row.setThreshold(RRQR_threshold);
			qr_A_row.compute(A_row);
			Mat new_P2M_transpose_old = qr_A_row.householderQ(); //new tree[j][n].L2P
			Mat R_row = qr_A_row.matrixQR().triangularView<Eigen::Upper>();
			new_P2M_transpose = new_P2M_transpose_old.block(0,0,new_P2M_transpose_old.rows(),qr_A_row.rank());
			R_row = new_P2M_transpose.adjoint()*A_row;

			Mat Rn = new_P2M_transpose.adjoint()*tree[j][b].P2M.adjoint();
			Mat Rn_prime = new_P2M_transpose.adjoint()*Rk_prime.adjoint();
			M2L = Rk*tree[j][k].M2L[b]*Rn.adjoint() + Rn_prime.adjoint();
		}

	void compress_M2P(int j, int k, int b, Mat &new_L2P, Mat &M2L) {
			Mat A_col(tree[j][k].L2P.rows(), tree[j][k].L2P.cols()+tree[j][k].M2P[b].cols());

			A_col << tree[j][k].L2P, tree[j][k].M2P[b];
			Eigen::ColPivHouseholderQR<Mat> qr_A_col(A_col.rows(), A_col.cols());
			qr_A_col.setThreshold(RRQR_threshold);
			qr_A_col.compute(A_col);
			Mat new_L2P_old = qr_A_col.householderQ() ; //new tree[j][k].L2P
			Mat R_col = qr_A_col.matrixR().triangularView<Eigen::Upper>();
			new_L2P = new_L2P_old.block(0,0,new_L2P_old.rows(),qr_A_col.rank());
			R_col = new_L2P.adjoint()*A_col;

			Mat Rk_prime = new_L2P.adjoint()*tree[j][k].M2P[b];
			Mat Rk = new_L2P.adjoint()*tree[j][k].L2P;
			M2L = Rk*tree[j][k].M2L[b] + Rk_prime;
		}

	void compress_P2L(int j, int k, int b, Mat &new_P2M_transpose, Mat &M2L) {
			Mat A_row(tree[j][b].P2M.cols(), tree[j][b].P2M.rows()+tree[j][k].P2L[b].rows());
			A_row << tree[j][b].P2M.adjoint(), tree[j][k].P2L[b].adjoint();
			Eigen::ColPivHouseholderQR<Mat> qr_A_row(A_row.rows(), A_row.cols());
			qr_A_row.setThreshold(RRQR_threshold);
			qr_A_row.compute(A_row);
			Mat new_P2M_transpose_old = qr_A_row.householderQ(); //new tree[j][b].L2P
			Mat R_row = qr_A_row.matrixQR().triangularView<Eigen::Upper>();
			new_P2M_transpose = new_P2M_transpose_old.block(0,0,new_P2M_transpose_old.rows(),qr_A_row.rank());
			R_row = new_P2M_transpose.adjoint()*A_row;

			Mat Rn = new_P2M_transpose.adjoint()*tree[j][b].P2M.adjoint();
			Mat Rn_prime = new_P2M_transpose.adjoint()*tree[j][k].P2L[b].adjoint();
			M2L = tree[j][k].M2L[b]*Rn.adjoint() + Rn_prime.adjoint();
		}
   
	/**************************************Updates*******************************************************/
	void update_other_M2Ls_at_k(int j, int k, int b, Mat &new_L2P) {
			#pragma omp parallel for
			for(int vn=0; vn<4; ++vn) {
				if(tree[j][k].vs_neighbor[vn] != -1) {
					int kIL = tree[j][k].vs_neighbor[vn];
					if(kIL == b) { //worked out in update_M2L_and_basis method
						continue;
					}
					std::cout<<"noissues"<<std::endl;
					tree[j][k].M2L[kIL] = new_L2P.adjoint() * tree[j][k].L2P * tree[j][k].M2L[kIL];//this product is the issue.
					
				}
			}
			for(int in=0; in<4; ++in) {
				if(tree[j][k].innerNumbers[in] != -1) {
					int kIL = tree[j][k].innerNumbers[in];
					if(kIL == b) { //worked out in update_M2L_and_basis method
						continue;
					}
					tree[j][k].M2L[kIL] = new_L2P.adjoint() * tree[j][k].L2P * tree[j][k].M2L[kIL];
				}
			}
			#pragma omp parallel for
			for(int on=0; on<24; ++on) {
				if(tree[j][k].outerNumbers[on] != -1) {
					int kIL = tree[j][k].outerNumbers[on];
					if(kIL == b) { //worked out in update_M2L_and_basis method
						continue;
					}
					tree[j][k].M2L[kIL] = new_L2P.adjoint() * tree[j][k].L2P * tree[j][k].M2L[kIL];
				}
			}
			//k's Neighbors
			#pragma omp parallel for
			for(int n=0; n<4; ++n) {
				int kN = tree[j][k].es_neighbor[n];
				if(kN != -1 && tree[j][kN].Eliminated && tree[j][k].Eliminated) {
					tree[j][k].M2L[kN] = new_L2P.adjoint() * tree[j][k].L2P * tree[j][k].M2L[kN];
				}
			}
			//k's self
			if(tree[j][k].Eliminated) {
				tree[j][k].M2L[k] = new_L2P.adjoint() * tree[j][k].L2P * tree[j][k].M2L[k];
			}
		}

	void update_other_M2Ls_at_b(int j, int k, int b, Mat &new_P2M_transpose) {
			#pragma omp parallel for
			for(int vn=0; vn<4; ++vn) {
				if(tree[j][b].vs_neighbor[vn] != -1) {
					int kIL = tree[j][b].vs_neighbor[vn];
					if(kIL == k) { //worked out in update_M2L_and_basis method
						continue;
					}
					tree[j][kIL].M2L[b] = tree[j][kIL].M2L[b] * tree[j][b].P2M * new_P2M_transpose;
				}
			}
			for(int in=0; in<4; ++in) {
				if(tree[j][b].innerNumbers[in] != -1) {
					int kIL = tree[j][b].innerNumbers[in];
					if(kIL == k) { //worked out in update_M2L_and_basis method
						continue;
					}
					tree[j][kIL].M2L[b] = tree[j][kIL].M2L[b] * tree[j][b].P2M * new_P2M_transpose;
				}
			}
			#pragma omp parallel for
			for(int on=0; on<24; ++on) {
				if(tree[j][b].outerNumbers[on] != -1) {
					int kIL = tree[j][b].outerNumbers[on];
					if(kIL == b) { //worked out in update_M2L_and_basis method
						continue;
					}
					tree[j][kIL].M2L[b] = tree[j][kIL].M2L[b] * tree[j][b].P2M * new_P2M_transpose;
				}
			}
			//k's Neighbors
			#pragma omp parallel for
			for(int n=0; n<4; ++n) {
				int kN = tree[j][b].es_neighbor[n];
				if(kN != -1 && tree[j][kN].Eliminated && tree[j][b].Eliminated) {
					tree[j][kN].M2L[b] = tree[j][kN].M2L[b] * tree[j][b].P2M * new_P2M_transpose;
				}
			}
			//k's self
			if(tree[j][b].Eliminated) {
				tree[j][b].M2L[b] = tree[j][b].M2L[b] * tree[j][b].P2M * new_P2M_transpose;
			}
		}

	void update_parent_L2P(int j, int k, Mat &new_L2P) {
			int kP = k/4;//k's parent
			int c_k = k%4; // k is c_k^{th} child of k_parent

			Mat U[4];
			int row_index = 0;
			for (size_t c = 0; c < 4; c++) {
				U[c] = tree[j-1][kP].L2P.block(row_index,0,tree[j][kP*4+c].NumLocals,tree[j-1][kP].L2P.cols());
				row_index += tree[j][kP*4+c].NumLocals;
			}
			U[c_k] = new_L2P.adjoint() * tree[j][k].L2P * U[c_k];
			tree[j-1][kP].L2P = Mat(U[0].rows()+U[1].rows()+U[2].rows()+U[3].rows(), U[0].cols());
			tree[j-1][kP].L2P << U[0], U[1], U[2], U[3];
		}

	void update_parent_P2M(int j, int b, Mat &new_P2M_transpose) {
			int bP = b/4;//k's parent
			int c_b = b%4; // k is c_b^{th} child of k_parent

			Mat V[4];
			int col_index = 0;
			for (size_t c = 0; c < 4; c++) {
				V[c] = tree[j-1][bP].P2M.block(0,col_index,tree[j-1][bP].P2M.rows(),tree[j][bP*4+c].NumMultipoles);
				col_index += tree[j][bP*4+c].NumMultipoles;
			}
			V[c_b] = V[c_b] * tree[j][b].P2M * new_P2M_transpose;
			tree[j-1][bP].P2M = Mat(V[0].rows(), V[0].cols()+V[1].cols()+V[2].cols()+V[3].cols());
			tree[j-1][bP].P2M << V[0], V[1], V[2], V[3];
		}


    /**************************************Elimination **************************************************/
	void initialise_P2P_NonLeafLevel(int j) {
		#pragma omp parallel for
		for (int k=0; k<nBoxesPerLevel[j]; ++k) {
			// #pragma omp parallel for
			for(int n=0; n<4; ++n) {
				int nn = tree[j][k].es_neighbor[n];//P2P_{k,nn}
				if(nn != -1) {
					int n_rows = 0;
					int n_cols = 0;
					for(int a=0; a<4; ++a) {
						n_rows += tree[j+1][4*k+a].NumLocals;//tree[j+1][4*k+a].M2L[4*nn].rows();
					}
					for(int b=0; b<4; ++b) {
						n_cols += tree[j+1][4*nn+b].NumMultipoles;//tree[j+1][4*k].M2L[4*nn+b].cols();
					}
					tree[j][k].P2P[nn] = Mat::Zero(n_rows, n_cols);
					int row_index = 0;
					int col_index = 0;
					for(int a=0; a<4; ++a) {
						for(int b=0; b<4; ++b) {
							tree[j][k].P2P[nn].block(row_index,col_index,tree[j+1][4*k+a].NumLocals,tree[j+1][4*nn+b].NumMultipoles) = tree[j+1][4*k+a].M2L[4*nn+b];
							col_index += tree[j+1][4*nn+b].NumMultipoles;//tree[j+1][4*k+a].M2L[4*nn+b].cols();
						}
						row_index += tree[j+1][4*k+a].NumLocals;//tree[j+1][4*k+a].M2L[4*nn].rows();
						col_index = 0;
					}
				}
			}
			// P2P_self
			{
				int nn = k;
				int n_rows = 0;
				int n_cols = 0;
				for(int a=0; a<4; ++a) {
					n_rows += tree[j+1][4*k+a].NumLocals;//tree[j+1][4*k+a].M2L[4*nn].rows();
				}
				for(int b=0; b<4; ++b) {
					n_cols += tree[j+1][4*nn+b].NumMultipoles;//tree[j+1][4*k].M2L[4*nn+b].cols();
				}
				tree[j][k].P2P[nn] = Mat::Zero(n_rows, n_cols);
				int row_index = 0;
				int col_index = 0;
				for(int a=0; a<4; ++a) {
					for(int b=0; b<4; ++b) {
						tree[j][k].P2P[nn].block(row_index,col_index,tree[j+1][4*k+a].NumLocals,tree[j+1][4*nn+b].NumMultipoles) = tree[j+1][4*k+a].M2L[4*nn+b];
						col_index += tree[j+1][4*nn+b].NumMultipoles;//tree[j+1][4*k+a].M2L[4*nn+b].cols();
					}
					row_index += tree[j+1][4*k+a].NumLocals;//tree[j+1][4*k+a].M2L[4*nn].rows();
					col_index = 0;
				}
			}
		}
	}
	
	void eliminate_cluster(int j, int k) {
	    eliminate_x(j, k);
		eliminate_z(j, k);
	}

	void eliminate_x(int j, int k) {
		if (tree[j][k].P2P[k].size() != 0) {
			Eigen::PartialPivLU<Mat> P2P_self_QR = tree[j][k].P2P[k].lu();
			filings_in_equation_P2M_due_to_x(j,k,P2P_self_QR);//equation z
			std::cout<< tree[j][k].L2M_f<<std::endl;
			for (size_t n = 0; n < 4; n++) { //loop over P2P(j,N(k))
				int nn = tree[j][k].es_neighbor[n];//P2P(j,nn)
				if(nn != -1) {
					filings_in_equation_P2P_due_to_x(j,k,nn,P2P_self_QR);
						
				}
			}
		}
		else {//In this portion we will form zero matrix of appropriate size just to make the program run with out issue.
			filings_in_equation_P2M_due_to_x(j,k);//equation z
			for (size_t n = 0; n < 4; n++) { //loop over P2P(j,N(k))
				int nn = tree[j][k].es_neighbor[n];//P2P(j,nn)
				if(nn != -1) {
					filings_in_equation_P2P_due_to_x(j,k,nn);
				}
			}
		}
	}
	void eliminate_z(int j, int k) {
		if (tree[j][k].L2M_f.size() > 0) {
			Eigen::PartialPivLU<Mat> L2M_f_QR=tree[j][k].L2M_f.lu(); 
			filings_in_equation_M2L_due_to_z(j,k,L2M_f_QR);
			// in equation P2P(j,N(k))
			for (size_t n = 0; n < 4; n++) { //loop over P2P(j,N(k))
				int nn = tree[j][k].es_neighbor[n];//P2P(j,nn)
				if(nn != -1) {
					filings_in_equation_P2P_due_to_z(j,k,nn,L2M_f_QR);
				}
			}
		}
		else {
			
			filings_in_equation_M2L_due_to_z(j,k);
			// in equation P2P(j,N(k))
			for (size_t n = 0; n < 4; n++) { //loop over P2P(j,N(k))
				int nn = tree[j][k].es_neighbor[n];//P2P(j,nn)
				if(nn != -1) {
					filings_in_equation_P2P_due_to_z(j,k,nn);
				}
			}
		}
	}

	void eliminate_phase_efficient() {
		for(int j=nLevels; j>=1; --j) {
			if(j != nLevels) {
				initialise_P2P_NonLeafLevel(j);
			}
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				for (size_t i = 0; i < P2P.size(); i++) {
					int nn = P2P[i].first;
					int pp = P2P[i].second;
					if (k == nn || k == pp) {
						P2P.erase(P2P.begin()+i);
						i--;
						if (tree[j][nn].P2P[pp].size() != 0) {
							compress_P2P_Two_Ways(j,nn,pp);
							tree[j][nn].P2P[pp] = Mat(0,0);
							tree[j][pp].P2P[nn] = Mat(0,0);
						}
					}
				}
				for (size_t i = 0; i < P2L_M2P.size(); i++) {
					int nn = P2L_M2P[i].first;
					int pp = P2L_M2P[i].second;
					if (k == nn || k == pp) {
						P2L_M2P.erase(P2L_M2P.begin()+i);
						i--;
						if (tree[j][nn].P2L[pp].size() != 0) {
							compress_M2P_P2L(j, pp, nn);
							tree[j][nn].P2L[pp] = Mat(0,0);
							tree[j][pp].M2P[nn] = Mat(0,0);
						}
					}
				}
				tree[j][k].Eliminated = true;
				eliminate_cluster(j,k);
			}
		}
	}
	
	void rhs_eliminate_cluster(int j, int k) {
		rhs_eliminate_x(j, k);
		rhs_eliminate_z(j, k);
	}
    void rhs_eliminate_x(int j, int k) {
		if (tree[j][k].P2P[k].size() != 0) {
			Eigen::PartialPivLU<Mat> P2P_self_QR = tree[j][k].P2P[k].lu();
			rhs_filings_in_equation_P2M_due_to_x(j,k,P2P_self_QR);//equation z
			for (size_t n = 0; n < 4; n++) { //loop over P2P(j,N(k))
				int nn = tree[j][k].es_neighbor[n];//P2P(j,nn)
				if(nn != -1) {
					rhs_filings_in_equation_P2P_due_to_x(j,k,nn,P2P_self_QR);
				}
			}
		}
		else {
			rhs_filings_in_equation_P2M_due_to_x(j,k);//equation z
			for (size_t n = 0; n < 4; n++) { //loop over P2P(j,N(k))
				int nn = tree[j][k].es_neighbor[n];//P2P(j,nn)
				if(nn != -1) {
					rhs_filings_in_equation_P2P_due_to_x(j,k,nn);
				}
			}
		}
	}
	void rhs_eliminate_z(int j, int k) {
		if (tree[j][k].L2M_f.size() > 0) {
			Eigen::PartialPivLU<Mat> L2M_f_QR = tree[j][k].L2M_f.lu();
			rhs_filings_in_equation_M2L_due_to_z(j,k,L2M_f_QR);
			// in equation P2P(j,N(k))
			for (size_t n = 0; n < 4; n++) { //loop over P2P(j,N(k))
				int nn = tree[j][k].es_neighbor[n];//P2P(j,nn)
				if(nn != -1) {
					rhs_filings_in_equation_P2P_due_to_z(j,k,nn,L2M_f_QR);
				}
			}
		}
		else {
			rhs_filings_in_equation_M2L_due_to_z(j,k);
			for (size_t n = 0; n < 4; n++) { //loop over P2P(j,N(k))
				int nn = tree[j][k].es_neighbor[n];//P2P(j,nn)
				if(nn != -1) {
					rhs_filings_in_equation_P2P_due_to_z(j,k,nn);
				}
			}
		}
	}
	/****************************************  RHS ****************************************************/
	void assign_Leaf_rhs(Vec &charges) {
		for (size_t k = 0; k < nBoxesPerLevel[nLevels]; k++) {
			tree[nLevels][k].particle_rhs	=	Vec::Zero(tree[nLevels][k].chargeLocations.size());
			for (size_t i = 0; i < tree[nLevels][k].chargeLocations.size(); i++) {
				int index = tree[nLevels][k].chargeLocations[i];
				tree[nLevels][k].particle_rhs[i]	=	charges[index];
			}
		}
	}
	void rhs_eliminate_phase_efficient() {
        for(int j=nLevels; j>=1; --j) {
		    for (int k=0; k<nBoxesPerLevel[j]; ++k) {
                tree[j][k].Eliminated = false;
        	}
    	}
		for(int j=nLevels; j>=1; --j) {
        	if(j != nLevels) {
        	assign_particle_rhs_NonLeafLevel(j);
			}
        	for (int k=0; k<nBoxesPerLevel[j]; ++k) {
         		tree[j][k].Eliminated = true;
          		rhs_eliminate_cluster(j,k);
			}
		}
	}
	
	/************************************ Solution phse  *********************************************/
	void solve_particles_at_base_level(int j) {
		int NumMultipoles = 0;
		int NumLocals = 0;
		for (int k=0; k<nBoxesPerLevel[j]; ++k) {
			NumMultipoles += tree[j][k].NumMultipoles;
			NumLocals += tree[j][k].NumLocals;
		}
    	if (NumLocals == 0 || NumMultipoles == 0) {
      		return;
   		}
   		else {
  			Mat A = Mat::Zero(NumLocals, NumMultipoles);
  			Vec b = Vec::Zero(NumLocals);
  			int r_index = 0;
  			for (int r=0; r<nBoxesPerLevel[j]; ++r) {
  				int s_index = 0;
  				for (int s=0; s<nBoxesPerLevel[j]; ++s) {
  					A.block(r_index, s_index, tree[j][r].NumLocals, tree[j][s].NumMultipoles) = tree[j][r].M2L[s];
  					s_index += tree[j][s].NumMultipoles;
  				}
  				b.segment(r_index, tree[j][r].NumLocals) = tree[j][r].multipole_rhs;
  				r_index += tree[j][r].NumLocals;
  			}
  			Eigen::PartialPivLU<Mat> A_qr = A.lu();
  			Vec multipoles = A_qr.solve(b);
  			int index = 0;
  			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
  				tree[j][k].multipoles = multipoles.segment(index, tree[j][k].NumMultipoles);
  				index += tree[j][k].NumMultipoles;
  			}
   	 	}
	}

	void back_substitution_phase() {
    //solve for particles at level 0; which are nothing but the multipoles at level 1
    	int j = 1;//base level
    	solve_particles_at_base_level(j);//now multipoles at level 1 are available
		for (size_t j = 1; j <= nLevels; j++) {
			for (int k=nBoxesPerLevel[j]-1; k>=0; k--) {
				if (tree[j][k].P2P[k].size() != 0) {
					Eigen::PartialPivLU<Mat> P2P_qr = tree[j][k].P2P[k].lu();
					Vec rhs1;
					rhs1 = tree[j][k].particle_rhs;
					for (size_t n = 0; n < 4; n++) {
						int nn = tree[j][k].es_neighbor[n];
						if (nn != -1) {
							if (!tree[j][nn].Eliminated) {
								rhs1 = rhs1 - tree[j][k].P2P[nn]*tree[j][nn].particles;
							}
							else {
								rhs1 = rhs1 - tree[j][k].M2P[nn]*tree[j][nn].multipoles;
							}
						}
					}
					Vec rhs2 = tree[j][k].multipoles - tree[j][k].P2M * (P2P_qr .solve(rhs1));
					Mat Atemp = -tree[j][k].P2M * (P2P_qr .solve(tree[j][k].L2P));
					if (Atemp.size() != 0) {
						Eigen::PartialPivLU<Mat> Atemp_qr = Atemp.lu();
						tree[j][k].locals = Atemp_qr .solve(rhs2);
					}
					else {
						tree[j][k].locals = Vec(0);
					}
					tree[j][k].particles = P2P_qr .solve(rhs1-tree[j][k].L2P * tree[j][k].locals);
				}
				else {
					tree[j][k].particles = Vec(0);
				}
				tree[j][k].Eliminated = false;
				if (nLevels != j) {
					int index = 0;
					for (size_t c = 0; c < 4; c++) { // get y^{k+1} from x^{k}
						tree[j+1][4*k+c].multipoles = tree[j][k].particles.segment(index, tree[j+1][4*k+c].NumMultipoles);
						index += tree[j+1][4*k+c].NumMultipoles;
					}
				}
			}
		}
	}
	
	void getx(Vec &x) {
    	int indexVec[nBoxesPerLevel[nLevels]];
    	for (size_t k = 0; k < nBoxesPerLevel[nLevels]; k++) {
      		if(k==0) {
        		indexVec[k] = 0;
      		}
      		else {
        		indexVec[k] = indexVec[k-1] + tree[nLevels][k-1].chargeLocations.size();
      		}
    	}
    	x = Vec::Zero(gridPoints.size()); //all particles
    // #pragma omp parallel for
   		for (size_t k = 0; k < nBoxesPerLevel[nLevels]; k++) {
      		x.segment(indexVec[k], tree[nLevels][k].particles.size()) = tree[nLevels][k].particles;
   		}
  	}
	
	
	
	void print_tree(){
    
		//for(int j=0;j<=nLevels;j++){
			//std::cout<<"----------------------------------------------"<<std::endl;
			//std::cout<<"Level:"<<j<<std::endl;
			//std::cout<<"----------------------------------------------"<<std::endl;
			for(int k=0;k<nBoxesPerLevel[nLevels];k++){
				//std::cout<<"***"<<std::endl;
				std::cout<<"Box Number:"<<tree[nLevels][k].boxNumber<<std::endl;
				// std::cout<<"Parent Number:"<<tree[j][k].boxNumber<<std::endl;
				// std::cout<<"vertex sharers:";
				//  for(int m=0;m<4;m++){
				// 	std::cout<<tree[j][k].vs_neighbor[m]<<",";
				//  }
				//  std::cout<<std::endl;
				// std::cout<<"edge sharers:";
				//  for(int m=0;m<4;m++){
				// 	std::cout<<tree[j][k].es_neighbor[m]<<",";
				//  }
				//  std::cout<<std::endl;
				// std::cout<<"Inner far-field:";
				//  for(int m=0;m<4;m++){
				// 	std::cout<<tree[j][k].innerNumbers[m]<<",";
				//  }
				//  std::cout<<std::endl;
				// std::cout<<"Outer far-field:";
				//  for(int m=0;m<8;m++){
				// 	std::cout<<tree[j][k].outerNumbers[m]<<",";
				//  }
				//  std::cout<<std::endl;
		        // std::cout<<"Is the node eliminated:"<<tree[j][k].Eliminated<<std::endl;	
				//std::cout<<tree[nLevels][k].P2P[k]<<std::endl;
				// std::cout<< "size:"<<tree[nLevels][k].chargeLocations.size();
				// for(size_t i=0;i<tree[nLevels][k].chargeLocations.size();i++){
				// 	std::cout<< tree[nLevels][k].chargeLocations[i]<<",";
				// }
				std::cout<<std::endl;
				std::cout<<"=============================================="<<std::endl;	 
			//}
		    
		}
	}
};
#endif