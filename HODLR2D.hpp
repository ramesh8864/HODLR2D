#include "HODLR2DTREE.hpp"

template <typename kerneltype>
class HODLR2D {
public:
  HODLR2DTree<kerneltype>* A;
  HODLR2D(kerneltype* K, int N, int nLevels, int TOL_POW, double* locations) {
    A	=	new HODLR2DTree<kerneltype>( K, N,  nLevels,  TOL_POW, locations);
    A->createTree();//done - working fine
  	A->assign_Tree_Interactions();//done - working fine
    A->assign_Center_Location();//done - working fine
  	A->assignChargeLocations();//done
    A->clearNonLeafChargeLocations();//done //Clear the charges at non leaf level because in nested approach we dont need charges at non leaf levels.
    A->getNodes();//done
    A->assemble_M2L();//done
    A->initialise_phase();//done
    A->initialise_P2P_Leaf_Level();//done
    //A->print_tree();//done
  }

  void factorize() {
   	A->eliminate_phase_efficient();
  }
  // Vec solve(Vec &rhs) {
    // A->assign_Leaf_rhs(rhs);
  //   A->rhs_eliminate_phase_efficient();
  //   A->back_substitution_phase();
  //   Vec phi;
  //   A->getx(phi);
  //   return phi;
   //}
  // void backSubstitute(Vec &rhs) {
  //   A->assign_Leaf_rhs(rhs);
  //   A->rhs_eliminate_phase_efficient();
  //   A->back_substitution_phase();
  // }

   void backSubstitute1(Vec &rhs) {
     A->assign_Leaf_rhs(rhs);
   }

  void backSubstitute2() {
    A->rhs_eliminate_phase_efficient();
  }

  void backSubstitute3() {
    A->back_substitution_phase();
  }
  void getSolution(Vec &phi) {
    A->getx(phi);
    A->reorder(phi);
  }
  // double getError(Vec &rhs) {
  //   A->assign_Leaf_rhs(rhs);
  //   return A->error_check();
  // }
  // ~AIFMM() {
  //   delete A;
  // };
};
