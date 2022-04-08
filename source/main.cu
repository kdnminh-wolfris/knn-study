#include <iostream>
#include <chrono>
#include <iomanip>
#include "luin_exact.cuh"

using namespace std;

const string datadir = "./data/";
const string dataset = "GSE128223";

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

    cout << "\nChecking results to the given answers..." << endl;
    cout << fixed << setprecision(5);
    float sim = solver.SimilarityCheck(datadir + dataset, true);
    cout << "Similarity: " << sim << endl;
    // float total_diff = solver.TotalDistanceDifferenceCheck(datadir + dataset);
    // cout << "Total difference in distance: " << total_diff << endl;
}