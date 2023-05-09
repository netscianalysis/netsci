/*--------------------------------------------------------------------*
 *                         CuRva                                      *
 *--------------------------------------------------------------------*
 * This is part of the GPU-accelerated, random variable analysis      *
 * library CuRva.                                                     *
 * Copyright (C) 2022 Andy Stokely                                    *
 *                                                                    *
 * This program is free software: you can redistribute it             *
 * and/or modify it under the terms of the GNU General Public License *
 * as published by the Free Software Foundation, either version 3 of  *
 * the License, or (at your option) any later version.                *
 *                                                                    *
 * This program is distributed in the hope that it will be useful,    *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      *
 * GNU General Public License for more details.                       *
 *                                                                    *
 * You should have received a copy of the GNU General Public License  *
 * along with this program.                                           *
 * If not, see <https://www.gnu.org/licenses/>                        *
 * -------------------------------------------------------------------*/

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-err58-cpp"

#include <map>

#ifndef CURVA_ELEMENT_H
#define CURVA_ELEMENT_H

	/*!
	@class ChemicalProperties
	@brief A class that contains chemical properties of elements.
	*/
class ChemicalProperties {
public:
/*!
	@function{atomicWeight} @type{double}
	@brief Returns the atomic weight of an element.

	@param symbol The symbol of the element.
	@type{const std::string&}
*/
	double atomicWeight (
			const std::string &
			symbol
	) {
		return _atomicWeights[symbol];
	}

private:
	std::map<
			std::string,
			double
	        > _atomicWeights = {
			{"H",  0.001007940},
			{"He", 0.004002602},
			{"Li", 0.006941000},
			{"Be", 0.009012180},
			{"B",  0.010811000},
			{"C",  0.012011000},
			{"N",  0.014006740},
			{"O",  0.015999400},
			{"F",  0.018998403},
			{"Ne", 0.020179700},
			{"Na", 0.022989768},
			{"Mg", 0.024305000},
			{"Al", 0.026981539},
			{"Si", 0.028085500},
			{"P",  0.030973762},
			{"S",  0.032066000},
			{"Cl", 0.035452700},
			{"Ar", 0.039948000},
			{"K",  0.039098300},
			{"Ca", 0.040078000},
			{"Sc", 0.044955910},
			{"Ti", 0.047880000},
			{"Cr", 0.051996100},
			{"Mn", 0.054938050},
			{"Fe", 0.055847000},
			{"Co", 0.058933200},
			{"Ni", 0.058693400},
			{"Cu", 0.063546000},
			{"Zn", 0.065390000},
			{"Ga", 0.069723000},
			{"Ge", 0.072610000},
			{"As", 0.074921590},
			{"Se", 0.078960000},
			{"Br", 0.079904000},
			{"Kr", 0.083800000},
			{"Rb", 0.085467800},
			{"Sr", 0.087620000},
			{"Y",  0.088905850},
			{"Zr", 0.091224000},
			{"Nb", 0.092906380},
			{"Mo", 0.095940000},
			{"Tc", 0.097907200},
			{"Ru", 0.101070000},
			{"Rh", 0.102905500},
			{"Pd", 0.106420000},
			{"Ag", 0.107868200},
			{"Cd", 0.112411000},
			{"In", 0.114818000},
			{"Sn", 0.118710000},
			{"Sb", 0.121760000},
			{"Te", 0.127600000},
			{"I",  0.126904470},
			{"Xe", 0.131290000},
			{"Cs", 0.132905430},
			{"Ba", 0.137327000},
			{"La", 0.138905500},
			{"Ce", 0.140115000},
			{"Pr", 0.140907650},
			{"Nd", 0.144240000},
			{"Pm", 0.144912700},
			{"Sm", 0.150360000},
			{"Eu", 0.151965000},
			{"Gd", 0.157250000},
			{"Tb", 0.158925340},
			{"Dy", 0.162500000},
			{"Ho", 0.164930320},
			{"Er", 0.167260000},
			{"Tm", 0.168934210},
			{"Yb", 0.173040000},
			{"Lu", 0.174967000},
			{"Hf", 0.178490000},
			{"Ta", 0.180947900},
			{"W",  0.183840000},
			{"Re", 0.186207000},
			{"Os", 0.190230000},
			{"Ir", 0.192220000},
			{"Pt", 0.195080000},
			{"Au", 0.196966540},
			{"Hg", 0.200590000},
			{"Tl", 0.204383300},
			{"Pb", 0.207200000},
			{"Bi", 0.208980370},
			{"Po", 0.208982400},
			{"At", 0.209987100},
			{"Rn", 0.222017600},
			{"Fr", 0.223019700},
			{"Ra", 0.226025400},
			{"Ac", 0.227027800},
			{"Th", 0.232038100},
			{"Pa", 0.231035880},
			{"U",  0.238028900},
			{"Np", 0.237048000},
			{"Pu", 0.244064200},
			{"Am", 0.243061400},
			{"Cm", 0.247070300},
			{"Bk", 0.247070300},
			{"Cf", 0.251079600},
			{"Es", 0.252083000},
			{"Fm", 0.257095100},
			{"Md", 0.258100000},
			{"No", 0.259100900},
			{"Lr", 0.262110000},
			{"Rf", 0.261000000},
			{"Db", 0.262000000},
			{"Sg", 0.266000000},
			{"Bh", 0.264000000},
			{"Hs", 0.269000000},
			{"Mt", 0.268000000},
			{"Ds", 0.269000000},
			{"Rg", 0.272000000},
			{"Cn", 0.277000000},
			{"Nh", 0.000000000},
			{"Fl", 0.289000000},
			{"Mc", 0.000000000},
			{"Lv", 0.000000000},
			{"Ts", 0.000000000},
			{"Og", 0.000000000}
	}; 
	//!< @brief The atomic masses of all the elements in atomic mass units (amu).
	};

#endif //CURVA_ELEMENT_H

#pragma clang diagnostic