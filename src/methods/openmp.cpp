#include "Header.h"
#include "qrcodegen.hpp"
#include "stb_image_write.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <omp.h> 

using namespace std;
using namespace std::chrono;

void generate_qr_omp(const string& text, bool saveToFile) {
    try {
        auto start = high_resolution_clock::now();

        // generam codul QR folosind libraria qrcodegen
        const qrcodegen::QrCode qr = qrcodegen::QrCode::encodeText(text.c_str(), qrcodegen::QrCode::Ecc::MEDIUM);
        const int size = qr.getSize();
        const int img_size = size * SCALE;
        vector<unsigned char> image(img_size * img_size * 3, 255);

        // setam numarul de threads
        omp_set_num_threads(NUM_THREADS);

        // paralelizam generarea imaginii folosind OpenMP
        #pragma omp parallel for schedule(dynamic)  // pentru a distribui sarcinile intre thread-uri
        for (int y = 0; y < size; y++) {
			vector<unsigned char> local_row(img_size * 3, 255); // initializam o linie locala pentru fiecare thread

            // procesareaza fiecare modul QR
            for (int x = 0; x < size; x++) {
                if (qr.getModule(x, y)) {
                    for (int i = 0; i < SCALE; i++) {
                        const int row_offset = (x * SCALE) * 3;
                        for (int j = 0; j < 3; j++) {
							local_row[row_offset + i * 3 + j] = 0;
                        }
                    }
                }
            }

			// copiem linia locala in imaginea finala
            for (int i = 0; i < SCALE; i++) {
                const int target_row = y * SCALE + i;
                copy(local_row.begin(), local_row.end(),
                    image.begin() + target_row * img_size * 3);
            }
        }

        // salvam imaginea in format PNG
        if (saveToFile) {
            if (!stbi_write_png("qr_omp.png", img_size, img_size, 3, image.data(), img_size * 3)) {
                cerr << "Eroare la salvarea imaginii PNG\n";
            }
        }

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << "Timp de executie (OpenMP): " << fixed << setprecision(3)
            << duration.count() / 1000.0 << " secunde." << endl;
    }
    catch (const exception& e) {
        cerr << "Eroare la generarea QR: " << e.what() << "\n";
    }
}

