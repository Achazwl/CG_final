nvcc -O3 main.cu -o main -std=c++17 -arch=compute_50 -code=sm_50
echo compileend
time ./main 
display image.ppm
