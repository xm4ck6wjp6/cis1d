#include "qchem.h"   //Include the Q-Chem global header
#include"BSetMgr.hh"
#include"BasisSet.hh"
#include "../cdman/moloop.h"
#include "../liblocorb/jutil.h"
#include <armadillo>

void get1einfo(double **S, double ** DIPX, double ** DIPY, double ** DIPZ, double ** DIPQ);
extern "C" void makes2_(double *S2, double *OvLp, double *PsOcA, double *PsOcB, double *PsViA, double *PsViB, int *NAct, int *NOa, int *NOb, int *NVa, int *NVb, double *V);
extern "C" void conjugatedipole(int*,int *, double*,double *,double *, double *);
extern double checksymmetry(int , double *,int, char *);
int analyze_diabat_tia(double * Amps, double * OccMO, double * VirtMO, int NMO, int NVirt,
                    int NBasis, char * label, int ilabel);

void symmetrize(double *A, const int N, double *scr);

void hht_func_cis(){
  printf("\n");
  printf("#######\n");
  printf("# CIS #");
  printf("#######\n");
  printf("\n");
  printf("#Last update: 09/24/2018");
  printf("#THREE parameters have to be assigned: neig, jmin and jmax\n");
  printf("#Only works for RHF, namely closed-shell systems w' all orbitals doubly occupied\n");
  printf("#Assume nlinor=nbasis\n");

  //READ INPUT
  int nbasis=bSetMgr.crntShlsStats(STAT_NBASIS);
  int nbasis2=nbasis*nbasis;
  int nocc=rem_read(REM_NALPHA);
  int nvir=nbasis-nocc;
  int ncisbasis=nocc*nvir;
  int nbas6d=bSetMgr.crntShlsStats(STAT_NBAS6D); //for Parray, Jarray and Karray
  int nbas6d2=nbas6d*nbas6d;
  int nb2car=rem_read(REM_NB2CAR); //for Pv and Jv

  //GET HF MO COEFF
  //[occ_a; vir_a; occ_b; vir_b]
  double *mo_coeff=QAllocDouble(nbasis2);
  VRload(mo_coeff,nbasis2,0.0); //initialize w' 0
  FileMan(FM_READ,FILE_MO_COEFS,1,nbasis2,0,1,mo_coeff); //get hf mo coeff here, only alpha spin is loaded
  double *mo_coeff_vir=mo_coeff+nbasis*nocc;

  //GET FOCK MATRIX IN MO BASIS (DIAG), used to calculate H1X later
  //In AO basis first (Q-Chem only gives the Fock matrix in AO basis)
  double *fock_ao=QAllocDouble(nbasis2);
  FileMan(FM_READ,FILE_FOCK_MATRIX,FM_DP,2*nbasis2,0,FM_BEG,fock_ao);
  //Then in MO basis, only do unitary transformation for alpha spin
  double *fock_mo=QAllocDouble(nbasis2); //should be diag, checked
  double *scratch1=QAllocDouble(nbasis2);
  AtimsB(scratch1,fock_ao,mo_coeff,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,1);
  AtimsB(fock_mo,mo_coeff,scratch1,nbasis,nbasis,nbasis,nbasis,nbasis,nbasis,2);

  //H1*X
  //Calculate invariant part first, Delta_eps=eps_a-eps_i (I make the index running with ia)
  arma::vec Delta_eps(ncisbasis);
  int jj, kk, ll; //dummy variables, jj runs for # iterations
  for(kk=0;kk<nocc;kk++){
    for(ll=0;ll<nvir;ll++){
      Delta_eps(ll*nocc+kk)=fock_mo[(nbasis+1)*(nocc+ll)]-fock_mo[nbasis*kk+kk];
    }
  }

  //#######################################
  //# DAVIDSON ALGORITHM STARTS FROM HERE #
  //#######################################
  //PARAMETERS
  int jmin=5; //j labels the number of guessing vectors for Davidson
  int jmax=10;
  int curj=jmin;
  int iter_max=pow(10.0,2.0);
  double residual_threshold=pow(10.0,-8.0);

  //INITIAL GUESS X
  arma::mat X(ncisbasis,jmax); X.zeros();
  arma::mat scratch3_arma(ncisbasis,ncisbasis); scratch3_arma.zeros();
  for(kk=0;kk<nocc;kk++){
    for(ll=0;ll<nvir;ll++){
      scratch3_arma(nocc-1-kk+nocc*ll,nvir*kk+ll)=1.0;
    }
  }
  printf("We need to check the form of scratch3_arma:\n");
  scratch3_arma.print();
  X.cols(0,curj-1)=scratch3_arma(0,curj-1);

  //DECLARING VARIABLES
  //For H1X, AOints, H2X and Davidson
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

  //RUNNING THE LOOP
  //Til convergence or reaching iter_max
  for(jj=0;jj<iter_max;jj++){
    //BUILD UP H^CIS*X IN SINGLE EXCITATION STATE |S_ia>={|Psi_ia>+|Psi_bar(i)bar(a)>}/sqrt(2)
    //<S_ia|H^CIS*X, i:occ, a:vir, _ia=_i^a, H=H1+H2
    //=(eps_a-eps_i)*X_ia+sum_jb{2(jb|ai)-(ji|ab)}X_jb

    //H1*X
    for(kk=0;kk<curj;kk++){
      H1X.col(kk)=Delta_eps%X.col(kk);
    }

    //H2*X, Parray(AOints>Jv,Karray
    //X_qchem, as the corresponding input of X for qchem, [X1;X2;...]
    for(kk=0;kk<ncisbasis;kk++){
      for(ll=0;ll<curj;ll++){
	X_qchem[kk+ll*ncisbasis]=X(kk,ll);
      }
    }

    //Calculate Pv and Parray, deal w' one state at a time
    VRload(Parray,nbas6d2*jmax,0.0);
    VRload(Pv,nb2car*jmax,0.0);
    for(kk=0;kk<curj;kk++){
      curX_qchem=X_qchem+kk*ncisbasis; //pick up one state
      //calculate Parray
      curParray=Parray+nbas6d2*kk;
      AtimsB(scratch1,curX_qchem,mo_coeff_vir,nocc,nbasis,nvir,nocc,nocc,nbasis,3); //X_ia
      AtimsB(curParray,mo_coeff_opt,scratch1,nbasis,nbasis,nocc,nbasis,nbasis,nocc,1);
      //transform to Pv form (half matrix)
      curPv=Pv+nb2car*kk;
      VRcopy(scratch1,curParray,nbasis2);
      symmetrize(scratch1,nbasis,scratch2);
      ScaV2M(scratch1,curPv,true,false);
    }

    //AOints
    rem_write(curj,REM_SET_PE_ROOTS); //have to tell Q-Chem the number of input states
    AOints(Jv,Karray,NULL,NULL,Pv,Parray,NULL,NULL,NULL,31);

    //Unpack Jv and transform Jarray and Karray to be in CIS basis (JX and KX)
    for(kk=0;kk<curj;kk++){
      curJv=Jv+kk*nb2car;
      curJarray=Jarray+kk*nbas6d2;
      ScaV2M(curJarray,curJv,true,true);
      curJX=JX+kk*ncisbasis;
      AtimsB(scratch1,curJarray,mo_coeff,nbasis,nocc,nbasis,nbasis,nbasis,nbasis,1);
      AtimsB(curJX,mo_coeff_vir,scratch1,nvir,nocc,nbasis,nvir,nbasis,nbasis,2);
      curKarray=Karray+kk*nbas6d2;
      curKX=KX+kk*ncisbasis;
      AtimsB(scratch1,curKarray,mo_coeff,nbasis,nocc,nbasis,nbasis,nbasis,nbasis,1);
      AtimsB(curKX,mo_coeff_vir,scratch1,nvir,nocc,nbasis,nvir,nbasis,nbasis,2);
    }

    //Finally, H2X, transform to armadillo type
    for(kk=0;kk<ncisbasis;kk++){
      for(ll=0;ll<curj;ll++){
	//The ouput K from Q-Chem is actually the -K we know...
	H2X(kk,ll)=2.0*JX[kk+ll*ncisbasis]+KX[kk+ll*ncisbasis];
      }
    }

    HX=H1X+H2X; //total HX in CIS basis
    redH=X.t()*HX; redH=redH.cols(0,curj-1).rows(0,curj-1);
    eig_sym(lambda,u,redH);
    residual_new=HX.cols(0,curj-1)*u.col(0)-lambda(0)*X.cols(0,curj-1)*u.col(0);
    if(norm(residual_new,2)<=residual_threshold){
      printf("The lowest eigen value is %.6f\n",lambda(0));
      break;
    }
    residual=residual_new;
    if(curj>=jmax){
      X.cols(0,jmin-1)=X.cols(0,curj-1)*u.cols(0,jmin-1);
      X.cols(jmin,jmax-1).zeros(); //all other elements should be zero
      curj=jmin;
    }
    M=Delta_eps-lambda(0)*onesforM; //this is actually NOT Davidson algorithm, need heavy comment later...
    tD=residual.col(0)/M; //D means Davidson
    //normalization
    for(kk=0;kk<curj;kk++){
      tD=tD-dot(tD,X.col(kk))*X.col(kk);
    }
    tD=tD/norm(tD,2);
    X.col(curj)=tD;
    curj++;
  }

  return;
}