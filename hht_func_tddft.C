#include "qchem.h"   //Include the Q-Chem global header
#include"BSetMgr.hh"
#include"BasisSet.hh"
#include "../cdman/moloop.h"
#include "../liblocorb/jutil.h"
#include <armadillo>

#include "../libdftn/xcclass.h"
#include "functionals.h"

void get1einfo(double **S, double ** DIPX, double ** DIPY, double ** DIPZ, double ** DIPQ);
extern "C" void makes2_(double *S2, double *OvLp, double *PsOcA, double *PsOcB, double *PsViA, double *PsViB, int *NAct, int *NOa, int *NOb, int *NVa, int *NVb, double *V);
extern "C" void conjugatedipole(int*,int *, double*,double *,double *, double *);
extern double checksymmetry(int , double *,int, char *);
int analyze_diabat_tia(double * Amps, double * OccMO, double * VirtMO, int NMO, int NVirt,
                    int NBasis, char * label, int ilabel);

void symmetrize(double *A, const int N, double *scr);

void hht_func_tddft() {
  printf("\n");
  printf("#######\n");
  printf("# CIS #\n");
  printf("#######\n");
  printf("\n");
  printf("# By HUNG-HSUAN TEH (teh@sas.upenn.edu)\n");
  printf("# Last update 09/15/2018\n");
  printf("# Only works for RHF, namely closed-shell systems w' all orbitals doubly occupied\n");
  printf("# Only singlet states are considered\n");
  printf("# Assume nlinor=nbasis\n");
  printf("# THREE parameters have to be determined: neig, jmin and jmax\n");
  printf("\n");
  printf("# WARNING: residual_threshold should be small enough; otherwise, some eigenstates might be missed,\n");
  printf("# ex. c2h4, hf, sto-3g, residual_threshold=10^-8\n");
  printf("# WARNING: Even when residual_threshold is small, bad initial guess still gives results w' missing eigenvalues\n");

  //READ INPUT
  int nbasis=bSetMgr.crntShlsStats(STAT_NBASIS);
  int nbasis2=nbasis*nbasis;
  int nocc=rem_read(REM_NALPHA);
  int nvir=nbasis-nocc;
  int ncisbasis=nocc*nvir;
  int nbas6d=bSetMgr.crntShlsStats(STAT_NBAS6D); //for Parray, Jarray and Karray
  int nbas6d2=nbas6d*nbas6d;
  int nb2car=rem_read(REM_NB2CAR); //for Pv and Jv

  //#################
  //# DFT VARIABLES #
  int jobNum=0; //inferred from dftman.C
  double *pD1E=NULL;
  double Ex;
  double Ec;
  double *pFxc_v=QAllocDouble(nb2car); VRload(pFxc_v,nb2car,0.0);
  double *pFxc=QAllocDouble(nbasis2); VRload(pFxc,nbasis2,0.0);
  XCFunctional XCFunc = rem_read(REM_LEVEXC) != -1 ? read_xcfunc(rem_read(REM_LEVEXC), (rem_read(REM_LEVCOR) <= XCFUNC_MAX) ? rem_read(REM_LEVCOR):0) : 0; //stolen from rem_setup.C

  BasisSet BasisDen(bSetMgr.crntCode()); //stolen from anlman/stsdata.C
  BasisSet BasisOrb(bSetMgr.crntCode());
  int bCodeDen=BasisDen.code();
  int bCodeOrb=BasisOrb.code();
  int GrdTyp=rem_read(REM_IGRDTY); //Stolen from functionals.C
  int IPrint = rem_read(REM_SCF_PRINT); //Stolen from scfman/scfman.C
  int iterSCF=0;  
  //#################

  //GET HF MO COEFF
  //[occ_a; vir_a; occ_b; vir_b]
  double *mo_coeff=QAllocDouble(nbasis2);
  VRload(mo_coeff,nbasis2,0.0); //initialize w' 0
  FileMan(FM_READ,FILE_MO_COEFS,1,nbasis2,0,1,mo_coeff); //get hf mo coeff here, only alpha spin
  double *mo_coeff_vir=mo_coeff+nbasis*nocc;

  //GET FOCK MATRIX IN MO BASIS (DIAG), used to calculate H1X later
  //In AO basis first (Q-Chem only gives the Fock matrix in AO basis)
  double *fock_ao=QAllocDouble(nbasis2);
  FileMan(FM_READ,FILE_FOCK_MATRIX,FM_DP,nbasis2,0,FM_BEG,fock_ao);
  //Then in MO basis, only do unitary transformation for alpha spin
  double *fock_mo=QAllocDouble(nbasis2);
  double *scratch1=QAllocDouble(nbasis2);
  AtimsB(scratch1,fock_ao,mo_coeff,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,1);
  AtimsB(fock_mo,mo_coeff,scratch1,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,2);

  //H1*X
  //Calculate invariant part first, Delta_eps=eps_a-eps_i
  arma::vec Delta_eps(ncisbasis);
  int ii,aa; //dummy variables
  for(ii=0;ii<nocc;ii++){
    for(aa=0;aa<nvir;aa++){
      Delta_eps(ii+aa*nocc)=fock_mo[(nbasis+1)*(nocc+aa)]-fock_mo[nbasis*ii+ii]; //Indices: ia
    }
  }

  printf("\n\n\n\n\n\n");
  printf("###################################################\n");
  printf("# Print out the Fock matrix in MO basis\n");
  printf("###################################################\n");
  int xx,yy;
  for(xx=0;xx<nbasis;xx++){
    for(yy=0;yy<nbasis;yy++){
      printf("f_mo=%.2f, ",fock_mo[xx+yy*nbasis]);
    }
    printf("\n");
  }

  printf("###################################################\n");
  printf("# Print out the Fock matrix in AO basis\n");
  printf("###################################################\n");
  for(xx=0;xx<nbasis;xx++){
    for(yy=0;yy<nbasis;yy++){
      printf("f_ao=%.2f, ",fock_ao[xx+yy*nbasis]);
    }
    printf("\n");
  }

  printf("###################################################\n");
  printf("# Also the MO coeff\n");
  printf("###################################################\n");
  for(xx=0;xx<nbasis;xx++){
    for(yy=0;yy<nbasis;yy++){
      printf("C=%.2f, ",mo_coeff[xx+yy*nbasis]);
    }
    printf("\n");
  }
  printf("\n\n\n\n\n\n");

  //#######################
  //# DFT, DENSITY MATRIX #
  double *Den=QAllocDouble(nbasis2);
  FileMan(FM_READ,FILE_DENSITY_MATRIX,FM_DP,nbasis2,0,FM_BEG,Den);
  //#######################

  //#######################################
  //# DAVIDSON ALGORITHM STARTS FROM HERE #

  //PARAMETERS
  int neig=3;
  int jmin=5; //j labels the number of guessing vectors for Davidson
  int jmax=10;
  int curj=jmin;
  int iter_max=pow(10.0,2.0);
  double residual_threshold=pow(10.0,-11.0);

  //INITIAL GUESS X
  arma::mat scratch3_arma(ncisbasis,ncisbasis); scratch3_arma.zeros();
  arma::mat X(ncisbasis,jmax);
  for(aa=0;aa<nvir;aa++){
    for(ii=0;ii<nocc;ii++){
      scratch3_arma(nocc-1-ii+nocc*aa,ii+nocc*aa)=1.0;
    }
  }
  X.cols(0,curj-1)=scratch3_arma.cols(0,curj-1);

  //DECLARING VARIABLES
  //For H1X, AOints, H2X and Davidson
  int jj,kk; //dummy variables, jj runs for iteration, kk runs for diff eigvalues
  //  int xx,yy;
  arma::mat H1X(ncisbasis,jmax); H1X.zeros();

  double *X_qchem=QAllocDouble(ncisbasis*jmax); VRload(X_qchem,ncisbasis*jmax,0.0);
  double *curX_qchem;
  double *Pv=QAllocDouble(nb2car*jmax);
  double *Parray=QAllocDouble(nbas6d2*jmax);
  double *curPv;
  double *curParray;

  double *scratch2=QAllocDouble(nbasis2); VRload(scratch2,nbasis2,0.0);
  double *Jv=QAllocDouble(nb2car*jmax); VRload(Jv,nb2car*jmax,0.0);
  double *Jarray=QAllocDouble(nbas6d2*jmax); VRload(Jarray,nbas6d2*jmax,0.0);
  double *curJv;
  double *curJarray;
  double *Karray=QAllocDouble(nbas6d2*jmax); VRload(Karray,nbas6d2*jmax,0.0);
  double *curKarray;
  double *JX=QAllocDouble(ncisbasis*jmax); VRload(JX,ncisbasis*jmax,0.0);
  double *KX=QAllocDouble(ncisbasis*jmax); VRload(KX,ncisbasis*jmax,0.0);
  double *curJX;
  double *curKX;
  //#################
  //# DFT VARIABLES #
  double *XCarray=QAllocDouble(nbas6d2*jmax); VRload(XCarray,nbas6d2*jmax,0.0);
  double *curXCarray;
  double *XCX=QAllocDouble(ncisbasis*jmax); VRload(XCX,ncisbasis*jmax,0.0);
  double *curXCX;
  //#################
  arma::mat H2X(ncisbasis,jmax); H2X.zeros();
  arma::mat HX(ncisbasis,jmax); HX.zeros();

  arma::mat redH;
  arma::vec lambda;
  arma::mat u;
  arma::vec residual_new;
  arma::vec residual;
  arma::vec M;
  arma::vec onesforM(ncisbasis); onesforM.ones();
  arma::vec tD;
  arma::mat Xf;
  arma::vec lambda_storage(neig); lambda_storage.zeros();
  arma::mat X_storage(ncisbasis,neig); X_storage.zeros();

  //LOOP FOR CALCULATING kk-TH EIGENVALUE
  for(kk=0;kk<neig;kk++){

  //LOOP FOR EACH EIGENVALUE
  //Til convergence or reaching iter_max
  for(jj=0;jj<iter_max;jj++){
    //BUILD UP H^CIS*X IN SINGLE EXCITATION STATE |S_ia>={|Psi_ia>+|Psi_bar(i)bar(a)>}/sqrt(2)
    //<S_ia|H^CIS*X, i:occ, a:vir, _ia=_i^a, H=H1+H2
    //=(eps_a-eps_i)*X_ia+sum_jb{2(jb|ai)-(ji|ab)}X_jb

    printf("\n");
    printf("This is %d-th loop\n",jj);
    fflush(stdout);

    //H1*X
    for(xx=0;xx<curj;xx++){
      H1X.col(xx)=Delta_eps%X.col(xx);
    }

    //H2*X, Parray(AOints>Jv,Karray
    //X_qchem, as the corresponding input of X for qchem, [X1;X2;...]
    for(xx=0;xx<ncisbasis;xx++){
      for(yy=0;yy<curj;yy++){
  	X_qchem[yy*ncisbasis+xx]=X(xx,yy);
      }
    }

    //Calculate Pv and Parray, deal w' one state at a time
    VRload(Parray,nbas6d2*jmax,0.0);
    VRload(Pv,nb2car*jmax,0.0);
    for(xx=0;xx<curj;xx++){
      curX_qchem=X_qchem+xx*ncisbasis; //pick up one state
      //calculate Parray
      curParray=Parray+nbas6d2*xx;
      AtimsB(scratch1,curX_qchem,mo_coeff_vir,nocc,nbasis,nvir,nocc,nocc,nbasis,3);
      AtimsB(curParray,mo_coeff,scratch1,nbasis,nbasis,nocc,nbasis,nbasis,nocc,1);

      //transform to Pv form (half matrix)
      curPv=Pv+nb2car*xx;
      VRcopy(scratch1,curParray,nbasis2);
      symmetrize(scratch1,nbasis,scratch2);
      ScaV2M(scratch1,curPv,true,false);
    }

    //AOints
    rem_write(curj,REM_SET_PE_ROOTS); //have to tell Q-Chem the number of input states
    AOints(Jv,Karray,NULL,NULL,Pv,Parray,NULL,NULL,NULL,31);

    //######################
    //# DFT, EXCHANGE PART #
    for(kk=0;kk<curj;kk++){
      curParray=Parray+nbas6d2*kk;
      symmetrize(curParray,nbasis,scratch2);
    }

    tdXcMtrx(XCFunc,Parray,Den,XCarray,curj,nbas6d,bCodeDen,GrdTyp);
    //######################

    //Unpack Jv and transform Jarray and Karray to be in CIS basis (JX and KX)
    for(xx=0;xx<curj;xx++){
      curJv=Jv+xx*nb2car;
      curJarray=Jarray+xx*nbas6d2;
      ScaV2M(curJarray,curJv,true,true);
      curJX=JX+xx*ncisbasis;
      AtimsB(scratch1,curJarray,mo_coeff_vir,nbasis,nvir,nbasis,nbasis,nbasis,nbasis,1);
      AtimsB(curJX,mo_coeff,scratch1,nocc,nvir,nbasis,nocc,nbasis,nbasis,2);

      curKarray=Karray+xx*nbas6d2;
      curKX=KX+xx*ncisbasis;
      AtimsB(scratch1,curKarray,mo_coeff_vir,nbasis,nvir,nbasis,nbasis,nbasis,nbasis,1);
      AtimsB(curKX,mo_coeff,scratch1,nocc,nvir,nbasis,nocc,nbasis,nbasis,2);

      //##########################
      //# DFT, EXCHANGE FUNC * X #
      curXCarray=XCarray+kk*nbas6d2;
      curXCX=XCX+kk*ncisbasis;
      AtimsB(scratch1,curXCarray,mo_coeff,nbasis,nocc,nbasis,nbasis,nbasis,nbasis,1);
      AtimsB(curXCX,mo_coeff_vir,scratch1,nvir,nocc,nbasis,nvir,nbasis,nbasis,2);
      //##########################
    }

    //Finally, H2X, transform to armadillo type
    for(xx=0;xx<ncisbasis;xx++){
      for(yy=0;yy<curj;yy++){
  	//The ouput K from Q-Chem is actually the -K we know...
  	H2X(xx,yy)=2.0*JX[yy*ncisbasis+xx]+KX[yy*ncisbasis+xx];
      }
    }

    HX=H1X+H2X; //total HX in CIS basis

    //Main Davidson
    redH=X.t()*HX; redH=redH.cols(0,curj-1).rows(0,curj-1);
    eig_sym(lambda,u,redH);
    residual_new=HX.cols(0,curj-1)*u.col(kk)-lambda(kk)*X.cols(0,curj-1)*u.col(kk);
    if(norm(residual_new,2)<=residual_threshold){
      Xf=X.cols(0,curj-1)*u;
      lambda_storage(kk)=lambda(kk);
      X_storage.col(kk)=Xf.col(kk);
      break;
    }
    residual=residual_new;

    if(curj>=jmax){
      X.cols(0,jmin-1)=X.cols(0,curj-1)*u.cols(0,jmin-1);
      X.cols(jmin,jmax-1).zeros(); //all other elements should be zero
      curj=jmin;
    }

    M=Delta_eps-lambda(kk)*onesforM; //this is actually NOT Davidson algorithm, need heavy comment later...
    tD=residual.col(0)/M; //D means Davidson
    //normalization
    for(xx=0;xx<curj;xx++){
      tD=tD-dot(tD,X.col(xx))*X.col(xx);
    }
    tD=tD/norm(tD,2);
    X.col(curj)=tD;
    curj++;
  }

  }
  //#######################################

  //PRINT
  printf("\n");
  printf("#Results of CIS:\n");
  printf("\n");
  printf("Correlation Energy for %d State(s):\n",neig);
  for(kk=0;kk<neig;kk++){
    printf("lambda%d=%.6f, ",kk,lambda_storage(kk));
  }
  printf("\n");
  printf("\n");
  printf("Corresponding Eigen State(s), X:\n");
  for(xx=0;xx<(ncisbasis);xx++){
    for(kk=0;kk<neig;kk++){
      printf("X%d=%.6f, ",kk,X_storage(xx,kk));
    }
    printf("\n");
  }

  return;
}
