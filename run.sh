declare -i server=0
if [ $server -eq 1 ]; then
	export CUDA_VISIBLE_DEVICES=3
	nvcc -O3 main.cu -o main -std=c++14 -arch=compute_75 -code=sm_75 -Wno-deprecated-gpu-targets
	echo compileend
	time ./main 
	convert image.ppm image.png
else
	nvcc -O3 main.cu -o main -std=c++14 -arch=compute_50 -code=sm_50 -Wno-deprecated-gpu-targets
	nohup gwenview image.ppm >/dev/null 2>&1 &
	echo compileend
	time ./main
fi
