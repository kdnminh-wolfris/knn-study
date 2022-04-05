#include <iostream>
#include <chrono>
#include <iomanip>
#include "luin_exact.cuh"

using namespace std;

const string datadir = "./data/";
const string dataset = "minitest";

int main(int argc, char** argv) {
    KnnSolver solver;

    cout << "\nReading data..." << endl;
    solver.ReadData(datadir + dataset + "/inp.inp");
    cout << "Done inputing" << endl;

    cout << "\nSolving..." << endl;
    auto start = chrono::high_resolution_clock::now();
    solver.Solve();
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
    cout << "Done solving..." << endl;
    printf("Duration: %ld.%09lds", duration.count() / int(1e9), duration.count() % int(1e9));
    cout << endl;

    cout << "\nWriting results to output files..." << endl;
    solver.WriteResults(datadir + dataset);
    cout << "Done outputing" << endl;

    cout << "\nChecking results to Faiss'..." << endl;
    float sim = solver.SimilarityCheck(datadir + dataset, true);
    cout << fixed << setprecision(5) << "Similarity: " << sim << endl;
}