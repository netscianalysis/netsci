%{
#include <algorithm>
%}

%inline%{
    std::string whitespace(int n) {
        std::string str = "";
        for (int i = 0; i < n; i++) {
            str += " ";
        }
        return str;
    }
%}

%extend CuArray{
    CuArrayRow<T> *__getitem__(int i) {
        return new CuArrayRow<T>($self, i);
    }
    void __setitem__(int i, CuArrayRow<T> *cuArrayRow) {
        for (int j = 0; j < cuArrayRow->n(); j++) {
            $self->set(cuArrayRow->operator[](j), i, j);
        }
    }
    int __len__() {
        return $self->m();
    }
};

%extend CuArrayRow{
        T __getitem__(int i) {
            return $self->operator[](i);
        }
        void __setitem__(int i, T value) {
            $self->operator[](i) = value;
        }
        int __len__() {
            return $self->n();
        }

        std::string __repr__() {
            int isFloatingPoint =
                    std::to_string($self->operator[](0)).find(".")
                    != std::string::npos;
            std::string repr_str = "";
            auto largestStrSize = std::to_string(
                    static_cast<int>(*std::max_element(
                            $self->data(),
                            $self->data() + $self->n()
                    ))).size();
            for (int i = 0; i < $self->n(); i++) {
                if (isFloatingPoint) {
                    std::string tmp =
                            std::to_string($self->operator[](i));
                    int decimalPos = tmp.find(".");
                    std::string s = tmp.substr(0, decimalPos + 5);
                    repr_str += s;
                    repr_str += whitespace(
                            largestStrSize - std::to_string(
                                    static_cast<int>(
                                            $self->operator[](i)
                                    )).size());
                } else {
                    std::string s = std::to_string(
                            $self->operator[](i)
                            );
                    repr_str += s;
                    repr_str += whitespace(
                            largestStrSize - s.size()
                    );
                }
                if (i < $self->n() - 1) {
                    repr_str += " ";
                }
            }
        return repr_str;
        }

};

%extend CuArray{
    std::string __repr__() {
        int isFloatingPoint =
                std::to_string($self->get(0, 0)).find(".")
                != std::string::npos;
        std::string repr_str = "";
        auto largestStrSize = std::to_string(
                static_cast<int>(*std::max_element(
                $self->host(), $self->host() + $self->size()
                ))).size();
        for (int i = 0; i < $self->m(); i++) {
            for (int j = 0; j < $self->n(); j++) {
                if (isFloatingPoint) {
                    std::string tmp = std::to_string($self->get(i, j));
                    int decimalPos = tmp.find(".");
                    std::string s = tmp.substr(0, decimalPos + 5);
                    repr_str += s;
                    repr_str += whitespace(
                            largestStrSize - std::to_string(
                                    static_cast<int>($self->get(i, j)
                                    )).size());
                } else {
                    std::string s = std::to_string($self->get(i, j));
                    repr_str += s;
                    repr_str += whitespace(
                            largestStrSize - s.size()
                            );
                }
                if (j < $self->n() - 1) {
                    repr_str += " ";
                }
            }
            if (i < $self->m() - 1) {
                repr_str += "\n";
            }
        }
    return repr_str;
    }
};
