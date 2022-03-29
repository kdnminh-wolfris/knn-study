PROJECT_FILES = main.cpp model_io.cpp exact_solver.cu checker.cpp
FLAGS = -std=c++17 --compiler-options='-Ofast'
EXE_FILE = main

run:
	nvcc ${PROJECT_FILES} ${FLAGS} -o ${EXE_FILE}
	./${EXE_FILE}