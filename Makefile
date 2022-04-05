SOURCE = ./source/
INCLUDES = ./includes/
FLAGS = -std=c++17 --compiler-options='-Ofast -march=native' -lcublas
EXE_FILE = main

run:
	nvcc ${SOURCE}* -I${INCLUDES} ${FLAGS} -o ${EXE_FILE}
	./${EXE_FILE}