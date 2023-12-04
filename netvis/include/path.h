#ifndef NETSCI_NETVIS_H
#define NETSCI_NETVIS_H
#include <string>
#include "cuarray.h"

class Path {
public:
    Path();

    ~Path();

    void init(
            const std::string& fname
    );

    int get(int i) const;

private:
    CuArray<int> *path_;
};

#endif //NETSCI_NETVIS_H
