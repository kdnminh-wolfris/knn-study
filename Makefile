SOURCE = ./source/
INCLUDES = ./includes/
FLAGS = -std=c++17 --compiler-options='-Ofast -march=native' -lcublas -lineinfo
EXE_FILE = main

run:
	nvcc ${SOURCE}* -I${INCLUDES} ${FLAGS} -o ${EXE_FILE}
	./${EXE_FILE}

# su thao
# Bio@2020
# sudo /usr/local/cuda/bin/ncu -o report_knn_exact -k __GetDistances --set full -f --import-source true /mnt2/minhkhau/projects/knn-study/main