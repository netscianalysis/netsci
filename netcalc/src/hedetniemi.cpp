//
// Created by astokely on 5/10/23.
//
#include <iostream>
#include <vector>
#include "hedetniemi.h"
#include <limits>

void hedetniemi(CuArray<float> *X, CuArray<float> *H, CuArray<int> *paths, int platform) {
    paths->init(X->m(), X->n() * 10);
    for (int i = 0; i < X->m(); i++) {
        for (int j = 0; j < 10 * X->n(); j++) {
            paths->set(-1, i, j);
        }
    }
    H->init(X->m(), X->n());
    gpuHedetniemi(X, H, paths);
}