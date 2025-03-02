set -e
make clean
cmake ../
make -j18
./darius