#include <iostream>
#include "luin_exact.cuh"

using namespace std;

const string datadir = "./data/";
const string dataset = "GSE128223";

int main(int argc, char** argv) {
    KnnSolver solver;
    solver.ReadData(datadir + dataset + "/inp.inp");
    solver.Solve();
}