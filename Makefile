solve:
	g++-8 model.cpp main.cpp -o main.exe -lopenblas -Ofast -march=native && ./main.exe