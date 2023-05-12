//
// Created by astokely on 5/10/23.
//

#ifndef NETCALC_HEDETNIEMI_H
#define NETCALC_HEDETNIEMI_H

#include "cuarray.h"

void hedetniemi(CuArray<float> *X, CuArray<float> *H, CuArray<int>* paths, int platform);

void gpuHedetniemi(CuArray<float> *X, CuArray<float> *H, CuArray<int>* paths);



#endif //NETCALC_HEDETNIEMI_H
