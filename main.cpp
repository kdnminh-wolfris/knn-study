#include "model.h"
#include <chrono>

const string dataset = "GSE128223";
const Algorithm solver = ver3;

int main() {
    cout << "\nDone compiling" << endl;

    cout << "\nInputting data..." << endl;
    KnnModel model;
    model.ReadData(dataset + "/data/inp.inp");
    model.SetAlgorithm(solver);
    cout << "Done inputting" << endl;

    setNbThreads(1);
    
    cout << "\nStart solving..." << endl;
    auto start = chrono::high_resolution_clock::now();
    model.Solve();
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << "Done solving" << endl;
    cout << "Duration: " << duration.count() / 1000000 << '.' << duration.count() % 1000000 << 's' << endl;

    cout << "\nOutputing results..." << endl;
    model.Output(dataset + "/data/out.out");
    cout << "Done outputing" << endl;
}