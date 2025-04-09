#include "Header.h"
#include "qrcodegen.hpp"
#include "stb_image_write.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

using namespace std;
using namespace std::chrono;

// kernel CUDA pentru a converti modulele QR in imagine
__global__ void qr_to_image_kernel(const unsigned char* qr_modules, int qr_size,
    unsigned char* image, int img_size) {
	// calculam pozitia modulului in matricea QR
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	// verificam daca pozitia este in limitele matricei QR
    if (x < qr_size && y < qr_size) {
        bool module = qr_modules[y * qr_size + x];
        int img_x = x * SCALE;
        int img_y = y * SCALE;

		// deseneaza patratul scalat
        for (int i = 0; i < SCALE; i++) {
            for (int j = 0; j < SCALE; j++) {
                int idx = (img_y + i) * img_size + (img_x + j);
                image[idx * 3 + 0] = image[idx * 3 + 1] = image[idx * 3 + 2] = module ? 0 : 255;
            }
        }
    }
}

void generate_qr_cuda(const string& text) {
    try {
        auto start = high_resolution_clock::now();

        // generam QR pe CPU
        const qrcodegen::QrCode qr = qrcodegen::QrCode::encodeText(text.c_str(), qrcodegen::QrCode::Ecc::MEDIUM);
        const int size = qr.getSize();
        const int img_size = size * SCALE;

        // date pt GPU
        vector<unsigned char> qr_modules(size * size);
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                qr_modules[y * size + x] = qr.getModule(x, y) ? 1 : 0;
            }
        }

        // alocare memorie pe GPU
        unsigned char* d_qr, * d_image;
        cudaMalloc(&d_qr, size * size);
        cudaMalloc(&d_image, img_size * img_size * 3);

        // transfer datele de la CPU la GPU
        cudaMemcpy(d_qr, qr_modules.data(), size * size, cudaMemcpyHostToDevice);
        cudaMemset(d_image, 255, img_size * img_size * 3);

        // configurare dim grid si block
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);

        // lansam kernel-ul CUDA
        qr_to_image_kernel << <grid, block >> > (d_qr, size, d_image, img_size);
        cudaDeviceSynchronize();

        // copy inapoi pe CPU
        vector<unsigned char> image(img_size * img_size * 3);
        cudaMemcpy(image.data(), d_image, img_size * img_size * 3, cudaMemcpyDeviceToHost);

        cudaFree(d_qr);
        cudaFree(d_image);

        // salvam imaginea in format PNG
        if (!stbi_write_png("qr_cuda.png", img_size, img_size, 3, image.data(), img_size * 3)) {
            cerr << "Eroare la salvarea imaginii PNG\n";
        }

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << "Timp de executie (CUDA): " << fixed << setprecision(3)
            << duration.count() / 1000.0 << " secunde." << endl;
    }
    catch (const exception& e) {
        cerr << "Eroare la generarea QR CUDA: " << e.what() << "\n";
    }
}