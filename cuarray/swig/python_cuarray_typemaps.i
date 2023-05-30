
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
};

