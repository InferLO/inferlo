DAI_PATH=/mnt/d/libdai_linux
g++ -o run_libdai.exe run_libdai.cpp -I$DAI_PATH/include -Llibdai/lib -ldai -lgmp -lgmpxx -L$DAI_PATH/lib -std=c++11