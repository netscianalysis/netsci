//
#include "hedetniemi.h"
#include <limits>
#include <cmath>

void netcalc::hedetniemiAllShortestPathsCpu(
        CuArray<float> *A,
        CuArray<float> *H,
        CuArray<int> *paths,
        float tolerance,
        int maxPathLength

) {

    H->fromCuArrayDeepCopy(
            A,
            0,
            A->m() - 1,
            A->m(),
            A->n()
    );
    auto Hi = new CuArray<float>;
    Hi->init(
            A->m() * maxPathLength,
            A->n()
    );
    int n = A->n();
    paths->init(
            n * maxPathLength,
            n
    );
    for (int _ = 0; _ < paths->size(); _++) {
        paths->host()[_] = -1;
    }
    for (int p = 0; p < maxPathLength; p++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float cij = std::numeric_limits<float>::infinity();
                for (int k = 0; k < n; k++) {
                    auto AikHkjSum = A->get(
                            k,
                            j
                    ) + H->get(
                            i,
                            k
                    );
                    if (AikHkjSum < cij) {
                        cij = AikHkjSum;
                    }
                }
                Hi->set(
                        H->get(i,
                               j),
                        n * p + i,
                        j
                );
                H->set(cij,
                       i,
                       j);
            }
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float length = H->get(i,
                                  j);
            int kj = j;
            for (int p = 0; p < maxPathLength; p++) {
                for (int k = 0; k < n; k++) {
                    float a = A->get(k,
                                     kj);
                    float h = Hi->get(
                            n * (maxPathLength - 2 - p) + i,
                            k
                    );
                    if (k != kj) {
                        if (a + h - length < tolerance) {
                            paths->set(
                                    k,
                                    i * maxPathLength + p,
                                    j
                            );
                            length = h;
                            kj = k;
                            break;
                        }
                    }
                }
            }
        }
    }
    delete Hi;
}
