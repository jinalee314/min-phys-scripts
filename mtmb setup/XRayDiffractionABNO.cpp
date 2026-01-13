/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2016 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "Colvar.h"
#include "ActionRegister.h"
#include "tools/NeighborList.h"
#include "tools/Communicator.h"
#include "tools/Tools.h"
//#include "compute_xrd_table.h"
#include "core/ActionWithVirtualAtom.h"
#include "core/PlumedMain.h"
#include "core/Atoms.h"
#include <cmath>
#include <time.h>
#include <string>

using namespace std;

#include <iostream>

namespace PLMD{
namespace colvar{

//+PLUMEDOC COLVAR Structure Funtion S(Q)
/*
*/
//+ENDPLUMEDOC

class XRayDiffractioneABNO : public Colvar {
  bool pbc;
  bool serial;
  double q;
  double maxr;
  double   lambda;
  unsigned qhist;

   unsigned int NumAtom;
   unsigned int NumAtom_A;
   unsigned int NumAtom_B;
   std::vector<AtomNumber> atoms_a;
   std::vector<AtomNumber> atoms_b;
   std::vector<AtomNumber> atoms;

  std::vector<double> theta;
  std::vector<double> structureF;
  std::vector<double> finalsf;
  std::vector<double> valueAA;
  std::vector<double> valueAB;
  std::vector<double> valueBB;
  std::vector<double> fij;
  std::vector<Value*> valueSF;


public:
  static void registerKeywords( Keywords& keys );
  explicit XRayDiffractioneABNO(const ActionOptions&);
  virtual void calculate();
};



PLUMED_REGISTER_ACTION(XRayDiffractioneABNO,"XRAYDIFFRACTIONABNO")

void XRayDiffractioneABNO::registerKeywords( Keywords& keys ){
  Colvar::registerKeywords(keys);
  keys.addFlag("SERIAL",false,"Perform the calculation in serial - for debug purpose");
  keys.add("atoms","A_ATOMS","First list of atoms");
  keys.add("atoms","B_ATOMS","Second list of atoms");
  keys.add("compulsory","LAMBDA","0.15406","Wavelength of the incident wave");
  keys.add("compulsory","MAXR","1","Maximum distance for the radial distribution function ");
  keys.add("compulsory","THETA","20","Diffraction angles");
  keys.add("compulsory","QHIST","1","Number of bins of Q ");
  keys.addOutputComponent("SF","default","the calculated structure function");
  ActionWithValue::useCustomisableComponents(keys); //The components in the action will depend on the user. 
}

XRayDiffractioneABNO::XRayDiffractioneABNO(const ActionOptions&ao):
PLUMED_COLVAR_INIT(ao),
pbc(true),
serial(false)
{
  parseFlag("SERIAL",serial);
  parseAtomList("A_ATOMS",atoms_a);
  parseAtomList("B_ATOMS",atoms_b);
  log.printf("%d and %d atoms of A and B are taking into calculation\n",atoms_a.size(),atoms_b.size());
     NumAtom_A=atoms_a.size();
     NumAtom_B=atoms_b.size();
     NumAtom=atoms_a.size()+atoms_b.size();
//  if(2*atoms_a.size()==atoms_b.size()){
//   NumAtom=3*atoms_a.size();
//  }else error("Number of A and B atoms should be matched\n");

  log.printf("Number of atoms= %d\n",NumAtom);
  atoms.resize(NumAtom);
  for(unsigned i=0;i<unsigned(NumAtom_A);i++){
    atoms[i]=atoms_a[i];
  }
  for(unsigned i= 0;i<unsigned(NumAtom_B);i++){
    atoms[i+unsigned(NumAtom_A)]=atoms_b[i];
  }
  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;
  parse("QHIST",qhist);
  log.printf("Calculation of structure function of Q from 0 to %u. \n", qhist );

  structureF.resize(qhist);
  finalsf.resize(qhist);
  valueSF.resize(qhist);
  valueAA.resize(qhist);
  valueAB.resize(qhist);
  valueBB.resize(qhist);

  int pos_count=1;
  std::ostringstream oss;
  for (unsigned int k=0;k<qhist;k++)
  {
   pos_count++;
   oss.str("");
   oss<<"SF["<<k<<"]"; 
   addComponentWithDerivatives(oss.str());  componentIsNotPeriodic(oss.str());  valueSF[k]=getPntrToComponent(oss.str());
  }

 std::cout<<"A_ATOMS=";
  for(unsigned i=0;i<unsigned(NumAtom_A);i++){
    std::cout<<","<<atoms[i].serial();
  }
  std::cout<<std::endl;
  std::cout<<"B_ATOMS=";
  for(unsigned i=unsigned(NumAtom_A);i<NumAtom;i++){
    std::cout<<","<<atoms[i].serial();
  }
  std::cout<<std::endl;

  log.printf("%d and %d atoms of A and B are taking into calculation\n",NumAtom_A,NumAtom_B);

  requestAtoms(atoms);

 std::cout<<"A_ATOMS=";
  for(unsigned i=0;i<unsigned(NumAtom_A);i++){
    std::cout<<","<<atoms[i].serial();
  }
  std::cout<<std::endl;
  std::cout<<"B_ATOMS=";
  for(unsigned i=unsigned(NumAtom_A);i<NumAtom;i++){
    std::cout<<","<<atoms[i].serial();
  }
  std::cout<<std::endl;

  parse("LAMBDA",lambda);
  parse("MAXR",maxr);
  parseVector("THETA",theta);
  checkRead();
}

// calculator
void XRayDiffractioneABNO::calculate()
{

    double d2;
    Vector distance;
    double distanceModulo;
    Vector distance_versor;

  Matrix<Vector> sfPrimeAA(qhist,getNumberOfAtoms());
  Matrix<Vector> sfPrimeAB(qhist,getNumberOfAtoms());
  Matrix<Vector> sfPrimeBB(qhist,getNumberOfAtoms());
  vector<Tensor> sfVirialAA(qhist);
  vector<Tensor> sfVirialAB(qhist);
  vector<Tensor> sfVirialBB(qhist);

   double fij_A=0;
   double fij_B=0;
  for(unsigned int m=0;m<qhist;++m){
    valueAA[m]=0.0;
    valueBB[m]=0.0;
    valueAB[m]=0.0;
    q=4*pi*std::sin(theta[m]*pi/180)/lambda;
        fij_A= 6.2915*exp(-2.4386*(q/(4*pi))*(q/(4*pi)))+3.0353*exp(-32.3337*(q/(4*pi))*(q/(4*pi)))+1.9891*exp(-0.6785*(q/(4*pi))*(q/(4*pi)))+1.541*exp(-81.6937*(q/(4*pi))*(q/(4*pi)))+1.1407;
        fij_B= 6.2915*exp(-2.4386*(q/(4*pi))*(q/(4*pi)))+3.0353*exp(-32.3337*(q/(4*pi))*(q/(4*pi)))+1.9891*exp(-0.6785*(q/(4*pi))*(q/(4*pi)))+1.541*exp(-81.6937*(q/(4*pi))*(q/(4*pi)))+1.1407;

   unsigned stride=comm.Get_size();
   unsigned rank=comm.Get_rank();
   for(unsigned int i=rank;i<NumAtom_A-1;i+=stride) {
   for(unsigned int j=i+1;j<NumAtom_A;j++) {
    if(pbc){
     distance=pbcDistance(getPosition(i),getPosition(j));
    } else {
     distance=delta(getPosition(i),getPosition(j));
    }
     d2=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
     distanceModulo=std::sqrt(d2);
     if(distanceModulo<maxr){
     distance_versor = distance / distanceModulo;
       valueAA[m]+=2*std::sin(q*distanceModulo)/(q*distanceModulo)*std::sin(pi*distanceModulo/maxr)/(pi*distanceModulo/maxr);
       Vector derivsf;
        derivsf=-2*maxr/(q*pi*distanceModulo*distanceModulo*distanceModulo)*(distanceModulo*(q*std::cos(q*distanceModulo)*std::sin(pi*distanceModulo/maxr)+pi/maxr*std::sin(q*distanceModulo)*std::cos(pi*distanceModulo/maxr))-2*std::sin(q*distanceModulo)*std::sin(pi*distanceModulo/maxr))*distance_versor;

       sfPrimeAA[m][i]+= derivsf;
       sfPrimeAA[m][j]+= -derivsf;
       Tensor vv(derivsf, distance);
       sfVirialAA[m] += vv ;
     }
   }
   }
       comm.Sum(valueAA);
       comm.Sum(sfPrimeAA);
       comm.Sum(sfVirialAA);

   unsigned stride2=comm.Get_size();
   unsigned rank2=comm.Get_rank();

   for(unsigned int l=rank2;l<NumAtom_B-1;l+=stride2) {
   for(unsigned int n=l+1;n<NumAtom_B;n++) {
    unsigned int i=l+NumAtom_A;
    unsigned int j=n+NumAtom_A;
    if(pbc){
     distance=pbcDistance(getPosition(i),getPosition(j));
    } else {
     distance=delta(getPosition(i),getPosition(j));
    }
     d2=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
     distanceModulo=std::sqrt(d2); 
     if(distanceModulo<maxr){
     distance_versor = distance / distanceModulo;
     valueBB[m]+=2*std::sin(q*distanceModulo)/(q*distanceModulo)*std::sin(pi*distanceModulo/maxr)/(pi*distanceModulo/maxr);
       Vector derivsf;
        derivsf=-2*maxr/(q*pi*distanceModulo*distanceModulo*distanceModulo)*(distanceModulo*(q*std::cos(q*distanceModulo)*std::sin(pi*distanceModulo/maxr)+pi/maxr*std::sin(q*distanceModulo)*std::cos(pi*distanceModulo/maxr))-2*std::sin(q*distanceModulo)*std::sin(pi*distanceModulo/maxr))*distance_versor;
       sfPrimeBB[m][i]+= derivsf;
       sfPrimeBB[m][j]+= -derivsf;
       Tensor vv(derivsf, distance);
       sfVirialBB[m] += vv ;
     }
   }
   }
       comm.Sum(valueBB);
       comm.Sum(sfPrimeBB);
       comm.Sum(sfVirialBB);


   unsigned stride3=comm.Get_size();
   unsigned rank3=comm.Get_rank();

   for(unsigned int i=rank3;i<NumAtom_A;i+=stride3) {
   for(unsigned int j=NumAtom_A;j<NumAtom;j++) {
    if(pbc){
     distance=pbcDistance(getPosition(i),getPosition(j));
    } else {
     distance=delta(getPosition(i),getPosition(j));
    }
     d2=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
     distanceModulo=std::sqrt(d2);
     if(distanceModulo<maxr){
     distance_versor = distance / distanceModulo;
     valueAB[m]+=2*std::sin(q*distanceModulo)/(q*distanceModulo)*std::sin(pi*distanceModulo/maxr)/(pi*distanceModulo/maxr);
       Vector derivsf;
        derivsf=-2*maxr/(q*pi*distanceModulo*distanceModulo*distanceModulo)*(distanceModulo*(q*std::cos(q*distanceModulo)*std::sin(pi*distanceModulo/maxr)+pi/maxr*std::sin(q*distanceModulo)*std::cos(pi*distanceModulo/maxr))-2*std::sin(q*distanceModulo)*std::sin(pi*distanceModulo/maxr))*distance_versor;
       sfPrimeAB[m][i]+= derivsf;
       sfPrimeAB[m][j]+= -derivsf;
      derivsf[0]=0;
      derivsf[1]=0;
      derivsf[2]=0;
       Tensor vv(derivsf, distance);
       sfVirialAB[m] += vv ;
     }
   }
   }

       comm.Sum(valueAB);
       comm.Sum(sfPrimeAB);
       comm.Sum(sfVirialAB);

         structureF[m]= (fij_A*fij_A*(NumAtom_A+valueAA[m])+fij_B*fij_B*(NumAtom_B+valueBB[m])+fij_A*fij_B*valueAB[m])/NumAtom;
         valueSF[m]->set(structureF[m]);
         for(unsigned i=0;i<getNumberOfAtoms();++i) setAtomsDerivatives(valueSF[m],i,(fij_A*fij_A*sfPrimeAA[m][i]+fij_B*fij_B*sfPrimeBB[m][i]+fij_A*fij_B*sfPrimeAB[m][i])/NumAtom);
         setBoxDerivatives  (valueSF[m],(fij_A*fij_A*sfVirialAA[m]+fij_B*fij_B*sfVirialBB[m]+fij_A*fij_B*sfVirialAB[m])/NumAtom);

  } // end loop for qhist


}

}
}
