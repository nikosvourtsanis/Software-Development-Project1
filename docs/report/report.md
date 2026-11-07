# Results Report and Comparative Analysis of Algorithms

## Introduction

In this assignment, the following approximate nearest neighbor search algorithms (Approximate Nearest Neighbors) were implemented and compared on high-dimensional data:

- **LSH (Locality-Sensitive Hashing)**
- **Hypercube**
- **IVFFlat (Inverted File Flat)**
- **IVFPQ (Inverted File with Product Quantization)**

The algorithms were tested on **MNIST**-type data, with the goal of reducing search time compared to exact Euclidean search (brute-force), while maintaining satisfactory accuracy (Recall).

---

## Theoretical Complexity Analysis and Expected Behavior

### LSH (Locality-Sensitive Hashing)

LSH is based on the use of random projections, so that vectors with a small distance are more likely to end up in the same bucket of a hash table.

**Complexity:**
- Training time: `O(n·k·L)`
- Search time: `O(L·k + C)` (C: number of candidates)
- Memory: `O(n·L)`

Very short search time, but the accuracy (Recall) depends on the parameters L, k, w.

 **Results:**  
Recall ≈ **1.0**, tApprox ≈ **50–60 ms**, QPS ≈ **17–19**  
 Very stable accuracy and fast performance.

---

### IVFFlat (Inverted File Flat)

IVFFlat divides the space into *kclusters* using k-means. Each query examines only *nprobe* clusters.

**Complexity:**
- Training: `O(n·k·d)`
- Search: `O(nprobe·(n/k)·d)`
- Memory: `O(n·d)`

 **Results:**  
Recall = **1.0** for all experiments, with tApprox from **2 to 25 ms**.  
The time increases with *nprobe*, but the accuracy remains perfect.

---

### IVFPQ (Inverted File with Product Quantization)

IVFPQ extends IVFFlat by reducing memory using *M subspaces* and *nbits bits* per subcode.

**Complexity:**
- Training: `O(n·k·d + n·M·2^{nbits})`
- Search: `O(nprobe·M)`
- Memory: `O(n·M)`

 **Results:**  
Recall ≈ **0.2–0.6**, AF ≈ **1.3–1.5**  
tApprox ≈ **0.1–1.5 ms**, QPS **500–7000+**  
Extremely fast with a small loss in Recall.

---

### Hypercube

Hypercube converts each point into a bit-vector and organizes the space as a binary cube graph.

 **Results:**  
Recall ≈ **0–0.2**, AF ≈ **1.7–2.4**  
tApprox ≈ **0.05–0.17 ms**, QPS **10,000–20,000**  
 Extremely fast but low accuracy.

---

## Comparative Evaluation

| Algorithm | Search Time | Memory | Recall | QPS | Remark |
|-------------|------------------|--------|---------|------|-------------|
| **LSH** | ~50–60 ms | High | 1.0 | ~18 | Very stable accuracy |
| **Hypercube** | <0.2 ms | Low | 0–0.2 | 10k–20k | Extremely fast |
| **IVFFlat** | 2–25 ms | High | 1.0 | 100–400 | Excellent balance |
| **IVFPQ** | 0.1–1.5 ms | Very low | 0.2–0.6 | 500–7000 | Fastest overall |

---

## Conclusions

- **LSH** proved to be the most **stable**, with high accuracy but longer time.  
- **Hypercube** has excellent speed but low Recall.  
- **IVFFlat** offers **almost perfect accuracy** with **moderate time**.  
- **IVFPQ** is **the most efficient overall**, offering very high QPS and low memory.  

 **Therefore:**
- For *real-time* searches → **LSH**
- For *high accuracy* → **IVFFlat**
- For *very large datasets* → **IVFPQ**