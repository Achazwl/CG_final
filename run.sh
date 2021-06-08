export CUDA_VISIBLE_DEVICES=3
nvcc -O3 main.cu -o main -std=c++14 -arch=compute_75 -code=sm_75 -Wno-deprecated-gpu-targets
echo compileend
time ./main 
convert image.ppm image.png
