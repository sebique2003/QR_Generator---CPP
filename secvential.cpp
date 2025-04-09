#include "header.h"
#include "qrcodegen.hpp"
#include "stb_image_write.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace std::chrono;

void generate_qr_secvential(const string& text) {
    try {
		// start timer
        auto start = high_resolution_clock::now();

		// generam codul QR folosind libraria qrcodegen
        const qrcodegen::QrCode qr = qrcodegen::QrCode::encodeText(text.c_str(), qrcodegen::QrCode::Ecc::MEDIUM);

		const int size = qr.getSize(); // dimensiunea QR
		const int img_size = size * SCALE; // dimensiunea imaginii finale
		vector<unsigned char> image(img_size * img_size * 3, 255); // initializam imaginea cu albzz

		// desenam fiecare modul QR in imagine
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
				if (qr.getModule(x, y)) { // verificam daca modulul este negru
					// deseneaza un patrat SCALE x SCALE pixeli
                    for (int i = 0; i < SCALE; i++) {
						for (int j = 0; j < SCALE; j++) {
                            const int idx = ((y * SCALE + i) * img_size + (x * SCALE + j)) * 3;
							image[idx] = image[idx + 1] = image[idx + 2] = 0;
                        }
                    }
                }
            }
        }

		// salvam imaginea in format PNG
        if (!stbi_write_png("qr_secvential.png", img_size, img_size, 3, image.data(), img_size * 3)) {
            cerr << "Eroare la salvarea imaginii PNG\n";
        }

		// stop timer
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << "Timp de executie (Secvential): " << fixed << setprecision(3)
            << duration.count() / 1000.0 << " secunde." << endl;
    }
    catch (const exception& e) {
        cerr << "Eroare la generarea QR: " << e.what() << "\n";
    }
}