# QR Code Generator - Proiect (Secvential vs. OpenMP vs. CUDA)

> 📘 Proiect realizat în cadrul cursului de **Programare Paralelă**  
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
#pragma omp parallel for schedule(dynamic)
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

## 🧪 Test Performanță

### Test realizat pe un sistem cu:
- CPU: Intel Core i5-9300H (4 nuclee, până la 4.1 GHz)
- GPU: NVIDIA GeForce GTX 1650 4GB GDDR5

### Timpi de execuție pentru generarea unui QR (din textul "Lorem Ipsum"):

| Metodă         | Timp de execuție |
| -------------  | ---------------- |
| **Secvențial** | 2.008 secunde    |
| **OpenMP**     | 1.991 secunde    |
| **CUDA**       | 2.107 secunde    |

### Observații:
- **Secvențial**: Timpul de execuție este de aproximativ 2 secunde.
- **OpenMP**: Timpul de execuție este de aproximativ 1.99 secunde, ceea ce arată o ușoară îmbunătățire față de varianta secvențială. Deși nu este o diferență mare, OpenMP permite procesarea paralelă pe CPU.
- **CUDA**: Timpul de execuție este de aproximativ 2.1 secunde, ceea ce este similar cu varianta secvențială. Deși CUDA oferă avantaje mari la dimensiuni mari de date, overhead-ul inițial al transferului datelor între CPU și GPU poate afecta performanța pentru dimensiuni mai mici ale QR-urilor.

### Concluzie:
- **OpenMP** este cel mai eficient pentru dimensiuni mici ale QR-urilor, cu timpi de execuție apropiati de varianta secvențială, dar cu o paralelizare eficientă pe CPU.
- **CUDA** poate deveni mai eficient pe măsură ce dimensiunile cresc, dar pentru QR-uri mici nu oferă o îmbunătățire semnificativă comparativ cu OpenMP.


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

Introdu sursa textului, manual sau din fișier.

Apoi alege metoda de generare:
1. Secvențial
2. OpenMP
3. CUDA

## 🚀 Îmbunătățiri viitoare posibile

- Suport pentru culori în QR
- Optimizări suplimentare în kernel-ul CUDA
- Paralelizare pe mai multe GPU-uri





