# Approximate Nearest Neighbor Search  
### LSH • Hypercube • IVFFlat • IVFPQ

## Project Team
- **Nikos Vourtsanis — sdi2200022**
- **Konstantinos Fagogenis — sdi2100199**

---

## Project Description

The project is the **1st assignment** of the course *Software Development for Algorithmic Problems* and deals with the implementation and comparison of techniques for **Approximate Nearest Neighbor Search (ANN)**.

The following methods are implemented and compared:

| Method | Description |
|--------|-----------|
| **LSH (Locality Sensitive Hashing)** | Gaussian projections, multiple hash tables |
| **Hypercube Hashing** | Projection onto a binary hypercube and exploration of vertices (probes) |
| **IVFFlat (Inverted File + Flat Quantization)** | K-means Clustering + inverted lists + search in nprobe clusters |
| **IVFPQ (Inverted File + Product Quantization)** | IVFFlat + Product Quantization (splitting into M subvectors, 2^nbits centroids) |

The system supports:

- K-Nearest Neighbors Search (KNN)
- Range Search (with radius **R**)

Datasets:

- **MNIST**
- **SIFT**

---

## Performance Metrics

The program computes:

- **Recall@N** — percentage of true nearest neighbors found
- **Approximation Factor (AF)** — approx/true distance ratio
- **QPS (Queries per Second)** — queries per second
- **tApprox / tTrue** — approximate & exact search times

Results are stored in `docs/report/results/`

---

## Compilation

### With `make`
```bash
make

or 

g++ -O3 -std=c++17 -Iinclude \
src/vector_utils.cpp src/lsh.cpp src/hypercube.cpp \
src/ivf_flat.cpp src/ivf_pq.cpp src/bruteForce.cpp src/main.cpp -o search

Execution Examples
MNIST — LSH
./search -d data/train-images.idx3-ubyte -q data/t10k-images.idx3-ubyte -type mnist \
-lsh -k 4 -L 5 -w 4.0 -N 1 -R 2000 -range false

MNIST — Hypercube
./search -d data/train-images.idx3-ubyte -q data/t10k-images.idx3-ubyte -type mnist \
-hypercube -kproj 14 -M 10 -probes 2 -w 4.0 -N 1 -R 2000 -range false

MNIST — IVFFlat
./search -d data/train-images.idx3-ubyte -q data/t10k-images.idx3-ubyte -type mnist \
-ivfflat -kclusters 50 -nprobe 5 -N 1 -R 2000 -range false

MNIST — IVFPQ
./search -d data/train-images.idx3-ubyte -q data/t10k-images.idx3-ubyte -type mnist \
-ivfpq -M 16 -nbits 8 -N 1 -R 2000 -range false

SIFT Dataset (has some errors)
./search -d data/sift_base.fvecs -q data/sift_query.fvecs -type sift \
-lsh -k 4 -L 5 -w 4.0 -N 1 -R 2 -range false


Output Format

Query:
NearestNeighbor-1:
distanceApproximate:
distanceTrue:
Average AF:
Recall@N:
QPS:
tApproximateAverage:
tTrueAverage:

Notes

The plots are generated via Gnuplot (plot_commands.gp)

CSV result files: docs/report/results/*.csv