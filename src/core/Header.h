#ifndef QR_GENERATOR_H
#define QR_GENERATOR_H

#include <string>
using namespace std;

#define SCALE 10	// factor de scalare pentru marirea QR-ului
#define NUM_THREADS 4	// nr de thread-uri pentru OpenMP
#define BLOCK_SIZE 32	// dimensiunea blocului CUDA

// functii pentru generarea codului QR
void generate_qr_secvential(const string& text);
void generate_qr_omp(const string& text);
void generate_qr_cuda(const string& text);
void comparaMetodele(const string& text);

#endif
