#include "Header.h"
#include <iostream>
#include <string>
#include <fstream>
using namespace std;

// functie pentru citirea unui fisier text
string read_file(const string& filename) {
    ifstream file(filename);
    string txt;

    if (file.is_open()) {
		getline(file, txt, '\0'); // citim tot fisierul
        file.close();
    }
    else {
        cerr << "Nu s-a putut deschide fisierul: " << filename << endl;
    }

    return txt;
}

int main() {
	// label proiect
    cout << "=====================================\n";
    cout << "Universitatea Transilvania din Brasov\n";
    cout << "IESC - Tehnologia Informatiei\n";
    cout << "Student: Iordache Sebastian-Ionut\n";
    cout << "Grupa: 4LF321B\n";
    cout << "Materia: Programare Paralela\n";
    cout << "======================================\n";
    cout << "Proiect QR Code Generator\n";
    cout << "Secvential vs. OpenMP vs. CUDA\n";
    cout << "======================================\n\n";

    string input;

	// meniu de introducere a textului
    int input_choice;
    cout << "Alegeti cum doriti sa introduceti textul pentru QR:\n";
    cout << "1. Introduceti manual\n";
    cout << "2. Cititi din fisier ('largeText.txt')\n";
    cout << "Alegeti optiunea (1-2): ";
    cin >> input_choice;
    cin.ignore();

    if (input_choice == 1) {
        cout << "Introduceti un text/link pentru codul QR: ";
        getline(cin, input);
    }
    else if (input_choice == 2) {
        input = read_file("A:\\Facultate\\PP\\test\\largeText.txt");
        if (input.empty()) {
            cerr << "Fisierul este gol sau nu s-a putut citi. Se va folosi un text implicit.\n";
            input = "Text implicit pentru generarea QR.";
        }
    }
    else {
        cout << "Optiune invalida! Se va folosi un text implicit.\n";
        input = "Text implicit pentru generarea QR.";
    }

	// meniu de alegere a metodei de generare
    int choice;
    cout << "\nSelectati metoda de generare:\n";
    cout << "1. Secvential\n";
    cout << "2. OpenMP\n";
    cout << "3. CUDA\n";
    cout << "4. Compara metodele\n";
    cout << "Alegeti optiunea (1-4): ";
    cin >> choice;

    switch (choice) {
    case 1:
        generate_qr_secvential(input);
        cout << "Fisierul 'qr_secvential.png' a fost salvat cu succes!\n";
        break;
    case 2:
        generate_qr_omp(input);
        cout << "Fisierul 'qr_omp.png' a fost salvat cu succes!\n";
        break;
    case 3:
        generate_qr_cuda(input);
        cout << "Fisierul 'qr_cuda.png' a fost salvat cu succes!\n";
        break;
    case 4:
        cout << "\nDoar se va compara timpul de executie al metodelor (nu se va genera un QR):\n";
        comparaMetodele (input);
        break;
    default:
        cout << "Optiune invalida! S-a folosit automat metoda secventiala.\n";
        generate_qr_secvential(input);
        cout << "Fisierul 'qr_secvential.png' a fost salvat cu succes!\n";
    }

    return 0;
}
