pip uninstall netcalc netchem cuarray -y
cd build
make clean
cd ..
rm -rf build
mkdir build
cd build
cmake .. -DCONDA_DIR=$CONDA_PREFIX
cmake --build . -j
make python

