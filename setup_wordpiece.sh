sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v
