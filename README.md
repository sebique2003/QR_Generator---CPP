# QR Code Generator - Proiect (Secvential vs. OpenMP vs. CUDA)

> 📘 Proiect realizat în cadrul cursului **Programare Paralelă**  
> Universitatea Transilvania din Brașov

## 📌 Descriere Proiect

### Acest proiect generează coduri QR folosind 3 metode diferite:
- **Versiunea secvențială** - cea mai simplă implementare
- **OpenMP** - paralelizare pe CPU
- **CUDA** - paralelizare pe GPU
Scopul este să comparăm performanța acestor abordări și să înțelegem cum paralelizarea poate îmbunătăți timpii de execuție.

## 🔧 Cum funcționează?

### 1. Generarea QR-ului (toate versiunile)
- Folosim librăria **qrcodegen** pentru a crea matricea de module (pătrățele negre/albe)
- Fiecare modul QR este scalat la SCALE x SCALE pixeli în imaginea finală
- Imaginea este salvată ca PNG folosind **stb_image_write**

### 2. Paralelizarea cu OpenMP
- Am împărțit procesarea pe axa Y a imaginii
- Fiecare thread primește un set de linii de procesat:
```cpp
#pragma omp parallel for schedule(static)
for (int y = 0; y < size; y++) {
    // Procesează linia y
}
```

- Fiecare thread lucrează pe un buffer local pentru a evita conflicte
- La final, thread-urile își combină rezultatele într-o secțiune critică

### 3. Paralelizarea cu CUDA

- Fiecare thread GPU este responsabil pentru un singur modul QR
- Kernel-ul transformă fiecare modul în blocuri de pixeli:
```cpp
  __global__ void qr_to_image_kernel(...) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < qr_size && y < qr_size) {
        // Desenează SCALE x SCALE pixeli
    }
}
```

- Folosim blocuri de 32x32 thread-uri (BLOCK_SIZE)
- Grid-ul este dimensionat să acopere întreaga matrice QR

## 🛠 Cum să rulezi?

### 1. Clonează repository-ul
 ```bash
git clone https://github.com/sebique2003/QR_Generator---CPP.git
cd QR_Generator---CPP
```
### 2. Compilare

Asigură-te că ai:
- Un compilator C++ (ex: g++)
- Suport pentru OpenMP (opțional, pentru paralelizare pe CPU)
- CUDA Toolkit instalat (opțional, pentru paralelizare pe GPU)
Apoi compilează:
```bash
make all
```

### 3. Rulare
```bash
./qr_generator
```

### 4. La Rulare

Introdu sursa textului: manual sau din fișier.
Alege metoda de generare:
1. Secvențial
2. OpenMP
3. CUDA

## 🚀 Îmbunătățiri viitoare posibile

- Suport pentru culori în QR
- Optimizări suplimentare în kernel-ul CUDA
- Paralelizare pe mai multe GPU-uri





