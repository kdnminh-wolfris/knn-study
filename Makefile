solve:
	g++-8 -I./eigen-3.4.0/eigen-3.4.0/ model.cpp main.cpp -Ofast -march=native -o main.exe && ./main.exe