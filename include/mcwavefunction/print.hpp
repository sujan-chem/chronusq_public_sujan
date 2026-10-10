/* 
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *  
 *  Copyright (C) 2014-2026 Li Research Group (University of Washington)
 *  
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *  
 *  Contact the Developers:
 *    E-Mail: xsli@uw.edu
 *  
 */
#pragma once

#include <mcscf.hpp>
#include <util/matout.hpp>
#include <util/print.hpp>
#include <cxxapi/output.hpp>

namespace ChronusQ {

  template <typename MatsT, typename IntsT>
  void MCWaveFunction<MatsT,IntsT>::printMOSpacePartition(std::string label) {
      
    auto & mopart = this->MOPartition;
    auto & ref = this->reference();

    std::string particletype = this->reference().particle.charge < 0 ? "Electron" : "Proton";
  
    std::cout << std::left << std::endl;
    std::string id = label.size() ? " For Quantum Subsystem " + label : "";
    FormattedLine(std::cout,"* MO Space Partition:" + id);
    FormattedLine(std::cout,"  Number of " + particletype + "ic MOs:", mopart.nElecMO);
    if (ref.nC == 4)
      FormattedLine(std::cout,"  Number of Negative MOs:", mopart.nNegMO);
    FormattedLine(std::cout,"  Number of " + particletype + "s in Full Space:",  ref.nO);
    
    std::cout << std::endl;
    FormattedLine(std::cout,"  Number of Inactive Orbitals:",     mopart.nInact);
    FormattedLine(std::cout,"  Number of Frozen Virtual Orbitals:",  mopart.nFVirt);
    
    FormattedLine(std::cout,"  Number of Correlated Orbitals:",      mopart.nCorrO);
    if (mopart.scheme == RAS) {
      FormattedLine(std::cout,"  Number of Orbitals in RAS 1:",      mopart.nActOs[0]);
      FormattedLine(std::cout,"  Number of Orbitals in RAS 2:",      mopart.nActOs[1]);
      FormattedLine(std::cout,"  Number of Orbitals in RAS 3:",      mopart.nActOs[2]);
      FormattedLine(std::cout,"  Maximum Number of Holes in RAS 1:",      mopart.mxHole);
      FormattedLine(std::cout,"  Maximum Number of Electrons in RAS 3:",  mopart.mxElec);
    }

    std::cout << std::endl;
    FormattedLine(std::cout,"  Number of Correlated " + particletype + "s:",     mopart.nCorrE);
    if (ref.nC == 1) {
      FormattedLine(std::cout,"  Number of Correlated Alpha " + particletype + "s:", mopart.nCorrEA);
      FormattedLine(std::cout,"  Number of Correlated Beta " + particletype + "s:",  mopart.nCorrEB);
    }
    
    std::cout << std::endl;
    FormattedLine(std::cout,"  Number of Roots Requested:",this->NStates);
    FormattedLine(std::cout,"  Number of Determinants:",   this->NDet);
    
    std::cout << std::endl;
    std::cout << std::right << std::endl;
    this->mointsTF->printMORangesSummary();
    std::cout << std::left  << std::endl;
  
  }; // MCWaveFunction::printMOSpacePartition
  
  template <typename MatsT, typename IntsT>
  void MCWaveFunction<MatsT,IntsT>::print1RDMs() {

    if( printRDMs==0 ) return;

    std::cout << std::endl << "MCWaveFunction 1RDMs:" << std::endl;
    std::cout << bannerTop << std::endl;

    size_t nCorrO = this->MOPartition.nCorrO;
    size_t nS     = this->NStates;
    size_t nRDMEle= 0; // used for line breaker

    //Print full 1RDM
    if( printRDMs==1 ){
      std::cout << "Printing full 1-RDM" << std::endl;
      for (auto i = 0ul; i < nS; i++) {

        prettyPrintSmart(std::cout, "State " + std::to_string(i),
          this->oneRDM[i].pointer(), nCorrO, nCorrO, nCorrO);

        //if (this->SpinAndAngularAnalysis)
          // this->spinAndAngularAnalysis(i);
      }
    } else if( printRDMs==2 ){
      std::cout.precision(2);
      std::cout << "Printing large (>" << std::fixed << rdmCut << ") diagonal elements of real 1-RDM" << std::endl;

      for (auto i = 0ul; i < nS; i++) {

        std::cout << std::endl << "State " << i+1 << ": ";
        nRDMEle = 0;

        for( auto ipp = 0ul; ipp < nCorrO; ipp++){

          if( std::real(this->oneRDM[i].pointer()[ipp*nCorrO+ipp]) > rdmCut ){
            if( nRDMEle > 6 ){
              std::cout << std::endl << "         ";
              nRDMEle = 0;
            }
            std::cout << std::setw(3) << ipp+1 << "(" << std::real(this->oneRDM[i].pointer()[ipp*nCorrO+ipp]) << ")" << " ";
            nRDMEle++;
          }
        }

        std::cout << std::endl;
        //if (this->spinAndAngularAnalysis)
          // this->spinAndAngularAnalysis(i);
      }
    }

    std::cout << bannerTop << std::endl;

  }; //MCWaveFunction::print1RDMs

  template <typename MatsT, typename IntsT>
  void MCWaveFunction<MatsT,IntsT>::printMOInfo(std::ostream& out, 
                      size_t printMOLevel) {
    
    SingleSlater<MatsT,IntsT> * ss_ptr = &reference(); 

    std::fill_n(ss_ptr->eps1, ss_ptr->nC * ss_ptr->nAlphaOrbital(),double(0.));
    if( ss_ptr->eps2 != nullptr )
      std::fill_n(ss_ptr->eps2, ss_ptr->nC * ss_ptr->nAlphaOrbital(),double(0.));

    if (not printMOLevel)
      printMOLevel = printMOCoeffs;
    ss_ptr->WaveFunction<MatsT,IntsT>::printMOInfo(out, printMOLevel);

  } //MCWaveFunction::printMOInfo

  template <typename MatsT, typename IntsT>
  void MCWaveFunction<MatsT,IntsT>::printMCStates()
  {
    size_t NDet  = this->NDet;
    size_t nS    = this->NStates;
    size_t NPrintC = std::min(NDet, size_t(25));   
    if(this->NDetPrint)
    {
        if(this->NDetPrint == DetPrint::ALLDET)
        {
            NPrintC = NDet;
        }
        else
        {
            NPrintC = this->NDetPrint;
        }
    }
    for (auto i = 0ul; i < nS; i++) { 
       
       // sort coeffients based on the norm 
       auto C = this->CIVecs[i];
       std::vector<size_t> Cindx(NDet);        
       std::iota(Cindx.begin(), Cindx.end(), 0);
       
       std::stable_sort(Cindx.begin(), Cindx.end(), 
         [&] (size_t i , size_t j) {
           return std::norm(C[i]) > std::norm(C[j]);
         }
       );
       
       // only print k Largerst coefficient
       this->printMCState(std::cout, i, this->StateEnergy->at(i), C, Cindx, NPrintC);  
       if(this->printDetConfig)
         this->printMCStateConfigs(std::cout, i, C, Cindx, NPrintC);
    }
  }

  /**
   *  \brief Print the active-space occupation string of the leading
   *  determinants of state i (CAS only).
   *
   *  2C/4C: one character per active spinor, 1 = occupied, 0 = empty.
   *  1C   : one character per active orbital, 2 = doubly occupied,
   *         a = alpha only, b = beta only, 0 = empty.
   *  Characters are ordered by active MO index; the absolute MO index
   *  of the first active orbital is printed in the header.
   */
  template <typename MatsT, typename IntsT>
  void MCWaveFunction<MatsT,IntsT>::printMCStateConfigs(std::ostream & out,
    size_t i, MatsT * C, std::vector<size_t> & sorted_CAddr, size_t N) {

    if( this->MOPartition.scheme != CAS ) {
      out << "  * PRINTCONFIG: occupation strings only implemented for CAS" << std::endl;
      return;
    }

    const size_t nC     = this->reference().nC;
    const size_t nCorrO = this->MOPartition.nCorrO;
    const size_t firstAct = this->MOPartition.nNegMO + this->MOPartition.nFCore
                          + this->MOPartition.nInact + 1;

    if( nC == 1 and not this->detStrBeta ) {
      out << "  * PRINTCONFIG: occupation strings NYI for 1C high-spin CAS" << std::endl;
      return;
    }

    auto addrA   = this->detStr->buildAddressingArray(this->detStr->nElectron(), nCorrO);
    auto deAddrA = this->detStr->buildDeAddressingArray(addrA);
    DetStringManager::int_matrix addrB, deAddrB;
    size_t nStrA = this->detStr->nString();
    if( nC == 1 ) {
      addrB   = this->detStrBeta->buildAddressingArray(this->detStrBeta->nElectron(), nCorrO);
      deAddrB = this->detStrBeta->buildDeAddressingArray(addrB);
      nStrA   = Comb(nCorrO, this->detStr->nElectron());
    }

    // Characterize the active orbitals: Mulliken l-character and <s_z>
    // (C^dagger S C restricted to each AO / spin block). 4C is skipped.
    std::vector<std::string> label(nCorrO, "?");
    std::vector<double> sz(nCorrO, 0.);
    const bool doLabel = (nC != 4);
    if( doLabel ) {
      auto & ref = this->reference();
      auto & basis = ref.basisSet();
      const size_t NB  = basis.nBasis;
      const size_t NBC = nC * NB;
      const IntsT * S  = ref.aoints_->overlap->pointer();
      const char lchar[] = "spdfghi";

      std::vector<size_t> bfL(NB, 0);
      for (size_t iSh = 0; iSh < basis.shells.size(); iSh++)
        for (size_t k = 0; k < basis.shells[iSh].size(); k++)
          bfL[basis.mapSh2Bf[iSh] + k] = basis.shells[iSh].contr[0].l;
      const size_t maxL = *std::max_element(bfL.begin(), bfL.end());

      std::vector<size_t> count(2*(maxL+1), 0);
      for (auto k = 0ul; k < nCorrO; k++) {
        const MatsT * Cmo = ref.mo[0].pointer() + (firstAct - 1 + k) * NBC;
        std::vector<double> popL(maxL+1, 0.);
        double popS[2] = {0., 0.};
        for (size_t sp = 0; sp < nC; sp++)
        for (size_t mu = 0; mu < NB; mu++) {
          MatsT SC = 0.;
          for (size_t nu = 0; nu < NB; nu++) SC += S[mu + nu*NB] * Cmo[sp*NB + nu];
          double p = std::real(std::conj(Cmo[sp*NB + mu]) * SC);
          popL[bfL[mu]] += p;
          popS[sp] += p;
        }
        size_t lMax = std::max_element(popL.begin(), popL.end()) - popL.begin();
        size_t spin = 0;
        std::string sLab = "";
        if( nC == 2 ) {
          sz[k] = 0.5 * (popS[0] - popS[1]);
          spin = popS[0] >= popS[1] ? 0 : 1;
          sLab = spin == 0 ? "a" : "b";
        }
        label[k] = std::string(1, lchar[std::min<size_t>(lMax, 6)])
                 + std::to_string(++count[2*lMax + spin]) + sLab;
      }

      if( i == 0 ) {
        out << std::endl << "  Active " << (nC == 1 ? "orbital" : "spinor")
            << " characters (Mulliken l-population"
            << (nC == 2 ? ", spin a/b from <s_z>" : "") << "):" << std::endl;
        for (auto k = 0ul; k < nCorrO; k++) {
          out << "    " << std::setw(4) << firstAct + k << "  "
              << std::left << std::setw(6) << label[k] << std::right;
          if( nC == 2 )
            out << "  <s_z> = " << std::fixed << std::setprecision(4)
                << std::setw(8) << sz[k];
          out << std::endl;
        }
      }
    }

    out << std::endl << "  Leading configurations of state " << i + 1
        << " (active " << (nC == 1 ? "orbitals " : "spinors ") << firstAct
        << " - " << firstAct + nCorrO - 1 << ", left to right):" << std::endl;
    out << "  " << std::setw(6) << "Det" << std::setw(12) << "|C|^2"
        << std::setw(12) << "Cum." << "   Occupation";
    if( nC == 2 and doLabel )
      out << std::string(std::max<int>(0, int(nCorrO + (nCorrO-1)/10) - 10), ' ')
          << "     M_S   Holes";
    out << std::endl;

    double cum = 0.;
    for (auto p = 0ul; p < N; p++) {
      size_t CAddr = sorted_CAddr[p];
      std::string occ(nCorrO, '0');

      if( nC == 1 ) {
        // CI vector is stored alpha-string fastest: addr = Kb * nStrA + Ka
        size_t Ka = CAddr % nStrA, Kb = CAddr / nStrA;
        for (auto o : this->detStr->address2DetString(Ka, addrA, deAddrA))  occ[o] = 'a';
        for (auto o : this->detStrBeta->address2DetString(Kb, addrB, deAddrB))
          occ[o] = (occ[o] == 'a') ? '2' : 'b';
      } else {
        for (auto o : this->detStr->address2DetString(CAddr, addrA, deAddrA)) occ[o] = '1';
      }

      double w = std::norm(C[CAddr]);
      cum += w;
      out << "  " << std::setw(6) << CAddr
          << std::fixed << std::setprecision(6) << std::setw(12) << w
          << std::setw(12) << cum << "   ";
      for (auto k = 0ul; k < nCorrO; k++) {
        if( k and k % 10 == 0 ) out << ' ';
        out << occ[k];
      }
      if( nC == 2 and doLabel ) {
        // M_S of the determinant and its holes (empty active spinors)
        double ms = 0.;
        for (auto k = 0ul; k < nCorrO; k++) if( occ[k] == '1' ) ms += sz[k];
        out << "  " << std::showpos << std::fixed << std::setprecision(2)
            << std::setw(6) << ms << std::noshowpos << "   ";
        for (auto k = 0ul; k < nCorrO; k++)
          if( occ[k] == '0' ) out << label[k] << " ";
      }
      out << std::endl;
    }

  } // MCWaveFunction::printMCStateConfigs

  template <typename MatsT, typename IntsT>
  void MCWaveFunction<MatsT,IntsT>::printMCState(std::ostream & out, 
                                                 size_t i,
                                                 double energy,
                                                 MatsT * C,
                                                 std::vector<size_t> & sorted_CAddr,
                                                 size_t N,
                                                 const size_t n_item_per_row)
  {
    out << std::fixed << std::right<< std::setprecision(10);
    out.fill(' ');
    
    out << std::endl <<  "State:" << std::setw(4) << i + 1 << "  Energy (Hartree):" 
    << std::setw(16) << energy <<  std::endl;
    
    out << std::fixed << std::right<< std::setprecision(7);
    
    size_t C_length = 10;
    size_t CAddr;
    
    for (auto j = 0ul; j < (N-1)/n_item_per_row + 1; j++) {
      size_t l = j*n_item_per_row;
      size_t r = std::min((j+1)*n_item_per_row, N);
      
      for (auto p = l; p < r; p++) {
        CAddr = sorted_CAddr[p]; 
        out << "(" << std::setw(5) << CAddr << ") " 
            << std::setw(C_length) << std::real(C[CAddr]); 
        if(std::is_same<MatsT, dcomplex>::value)  
          out << " " << std::setw(C_length) << std::imag(C[CAddr]);
        out << "  ";  
      }
      out << std::endl;
    }
 
  } // MCWaveFunction::printMCWaveFunctionState



}; // namespace ChronusQ
