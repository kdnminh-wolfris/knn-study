#include "model.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cblas.h>
#include <stdio.h>

const string dataset = "GSE157278";

int main() {
    cout << "\nDone compiling" << endl;

    cout << "\nInputing data..." << endl;
    KnnModel model;
    model.ReadData("./data/" + dataset + "/inp.inp");
    cout << "Done inputing" << endl;

    openblas_set_num_threads(1);
    
    cout << "\nStart solving..." << endl;
    auto start = chrono::high_resolution_clock::now();
    model.Solve();
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << "Done solving" << endl;
    printf("Duration: %ld.%06lds", duration.count() / int(1e6), duration.count() % int(1e6));
    cout << endl;

    // cout << "\nOutputing results..." << endl;
    // model.Output("./data/" + dataset + "/out.out");
    // cout << "Done outputing" << endl;
}