#include "Header.h"
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

void comparaMetodele(const string& text, bool saveToFile) {

    double time_sec, time_omp, time_cuda;

    // secvential
    {
        auto start = high_resolution_clock::now();
        generate_qr_secvential(text, saveToFile);
        auto end = high_resolution_clock::now();
        time_sec = duration_cast<milliseconds>(end - start).count() / 1000.0;
    }

    // opemMP
    {
        auto start = high_resolution_clock::now();
        generate_qr_omp(text, saveToFile);
        auto end = high_resolution_clock::now();
        time_omp = duration_cast<milliseconds>(end - start).count() / 1000.0;
    }

    // cuda
    {
        auto start = high_resolution_clock::now();
        generate_qr_cuda(text, saveToFile);
        auto end = high_resolution_clock::now();
        time_cuda = duration_cast<milliseconds>(end - start).count() / 1000.0;
    }
}
