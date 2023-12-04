#include "netvis.h"

Path::Path() {
    this->path_ = new CuArray<int>();
}

Path::~Path() = default;

void Path::init(
        const std::string& fname
) {
    this->path_->load(fname);
}

int Path::get(int i) const {
    return this->path_->get(0, i);
}
