PROJECT_FILES = main.cpp matrix.cu
EXE_FILE = main

run:
	nvcc ${PROJECT_FILES} -o ${EXE_FILE}
	./${EXE_FILE}