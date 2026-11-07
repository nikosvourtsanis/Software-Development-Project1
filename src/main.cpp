#include <iostream>
#include <cmath>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream> // Required for file output
#include "../include/bruteForce.h"
#include "../include/dataset_io.hpp"
#include "../include/vector_utils.h"
#include "../include/lsh.h"
#include "../include/hypercube.h"
#include "../include/ivf_flat.hpp"
#include "../include/ivf_pq.hpp"

struct Config {
    // common
    std::string input_path;   // -d
    std::string query_path;   // -q
    std::string output_path = "results.txt"; // -o
    std::string type;         // -type mnist|sift
    std::string knn_method ="ivf"; //default best
    int N = 1;                // -N
    double R = 2000.0;        // -R
    bool do_range = false;    // -range true|false
    int seed = 1;             // -seed

    // LSH
    bool use_lsh = false;
    int k = 4;
    int L = 5;
    double w = 4.0;

    // Hypercube
    bool use_hypercube = false;
    int kproj = 14;
    int M = 10;
    int probes = 2;

    // IVFFlat
    bool use_ivfflat = false;
    int kclusters = 50;
    int nprobe = 5;

    // IVFPQ
    bool use_ivfpq = false;
    int M_pq = 16;
    int nbits = 8;

    // KNN GRAPH MODE
    bool build_knn = false;
    int knn_k = 10;
    std::string index_path;

    void finalize_defaults() {
        if (!type.empty()) {
            if (type == "sift" && R == 2000.0) R = 2.0;
        }
    }
};

static bool iequals(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i])) return false;
    return true;
}

static bool to_bool(const std::string& s) {
    if (iequals(s, "true") || s == "1") return true;
    if (iequals(s, "false") || s == "0") return false;
    throw std::runtime_error("Invalid boolean value: " + s + " (expected true|false|1|0)");
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](int more) {
            if (i + more >= argc) throw std::runtime_error("Missing value after " + k);
        };

        if (k == "-d") { need(1); cfg.input_path = argv[++i]; }
        else if (k == "-q") { need(1); cfg.query_path = argv[++i]; }
        else if (k == "-o") { need(1); cfg.output_path = argv[++i]; }
        else if (k == "-type") { need(1); cfg.type = argv[++i]; }
        else if (k == "-N") { need(1); cfg.N = std::stoi(argv[++i]); }
        else if (k == "-R") { need(1); cfg.R = std::stod(argv[++i]); }
        else if (k == "-range") { need(1); cfg.do_range = to_bool(argv[++i]); }
        else if (k == "-seed") { need(1); cfg.seed = std::stoi(argv[++i]); }

        // LSH
        else if (k == "-lsh") { cfg.use_lsh = true; }
        else if (k == "-k") { need(1); cfg.k = std::stoi(argv[++i]); }
        else if (k == "-L") { need(1); cfg.L = std::stoi(argv[++i]); }
        else if (k == "-w") { need(1); cfg.w = std::stod(argv[++i]); }

        // Hypercube
        else if (k == "-hypercube") { cfg.use_hypercube = true; }
        else if (k == "-kproj") { need(1); cfg.kproj = std::stoi(argv[++i]); }
        else if (k == "-M") {
            need(1);
            int val = std::stoi(argv[++i]);
            cfg.M = val;
            cfg.M_pq = val;
        }
        else if (k == "-probes") { need(1); cfg.probes = std::stoi(argv[++i]); }

        // IVFFlat
        else if (k == "-ivfflat") { cfg.use_ivfflat = true; }
        else if (k == "-kclusters") { need(1); cfg.kclusters = std::stoi(argv[++i]); }
        else if (k == "-nprobe") { need(1); cfg.nprobe = std::stoi(argv[++i]); }

        // IVFPQ
        else if (k == "-ivfpq") { cfg.use_ivfpq = true; }
        else if (k == "-nbits") { need(1); cfg.nbits = std::stoi(argv[++i]); }

        // KNN GRAPH
        else if (k == "-build_knn") { cfg.build_knn = true; }
        else if (k == "-K") { need(1); cfg.knn_k = std::stoi(argv[++i]); }
        else if (k == "-i") { need(1); cfg.index_path = argv[++i]; }
        else if (k == "-build_knn_method") {need(1); cfg.knn_method = argv[++i]; }
    }
    cfg.finalize_defaults();
    return cfg;
}

// --- Forward Declarations ---
void run_lsh(const Matrix& base, const Matrix& queries, const Config& cfg);
void run_hypercube(const Matrix& base, const Matrix& queries, const Config& cfg);
void run_ivfflat(const Matrix& base, const Matrix& queries, const Config& cfg);
void run_ivfpq(const Matrix& base, const Matrix& queries, const Config& cfg);
void run_build_knn(const Matrix& base, const Config& cfg);

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);
        
        Matrix base, queries;
        if (iequals(cfg.type, "mnist")) {
            base = load_mnist_images(cfg.input_path, false);
            if (!cfg.build_knn) queries = load_mnist_images(cfg.query_path, false); 
        } else {
            base = load_fvecs(cfg.input_path);
            if (!cfg.build_knn) queries = load_fvecs(cfg.query_path);
        }

        // Dispatch
        if (cfg.use_lsh) run_lsh(base, queries, cfg);
        else if (cfg.use_hypercube) run_hypercube(base, queries, cfg);
        else if (cfg.use_ivfflat) run_ivfflat(base, queries, cfg);
        else if (cfg.use_ivfpq) run_ivfpq(base, queries, cfg);
        else if(cfg.build_knn) run_build_knn(base, cfg);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}

// ---------------- IMPLEMENTATIONS ----------------

void run_lsh(const Matrix& base, const Matrix& queries, const Config& cfg){
    using namespace std::chrono;
    std::ofstream out(cfg.output_path);
    if(!out){ std::cerr << "[ERROR] Could not open output file: " << cfg.output_path << "\n"; return; }
   
    out << "LSH\n";
    std::vector<std::vector<float>> base_vecs(base.n);
    for(int i = 0; i < base.n; ++i)
        base_vecs[i] = std::vector<float>(base.row(i), base.row(i) + base.d);

    lsh::LSH index(base.d, cfg.k, cfg.L, cfg.w, -1, cfg.seed);
    index.buildIndex(base_vecs);

    for(int qi = 0; qi < queries.n; ++qi){
        std::vector<float> q(queries.row(qi), queries.row(qi) + queries.d);
        auto approx = index.searchKNN(q, cfg.N);
        
        out << "Query: " << qi << "\n";
        for (int i = 0; i < (int)approx.size(); ++i) {
            out << "Nearest neighbor-" << (i+1) << ": " << approx[i].first << "\n";
        }
    }
    std::cout << "[LSH] Results saved to " << cfg.output_path << "\n";
}

void run_hypercube(const Matrix& base, const Matrix& queries, const Config& cfg){
    using namespace std::chrono;
    std::ofstream out(cfg.output_path);
    if(!out){ std::cerr << "[ERROR] Could not open output file: " << cfg.output_path << "\n"; return; }
    out << "Hypercube\n";

    std::vector<std::vector<float>> base_vecs(base.n);
    for(int i = 0; i < base.n; ++i)
        base_vecs[i] = std::vector<float>(base.row(i), base.row(i) + base.d);

    cube::Hypercube hc(base.d, cfg.kproj, cfg.w, cfg.M, cfg.probes, cfg.seed);
    hc.buildIndex(base_vecs);

    for(int qi = 0; qi < queries.n; ++qi){
        std::vector<float> q(queries.row(qi), queries.row(qi) + queries.d);
        auto approx = hc.searchKNN(q, cfg.N);

        out << "Query: " << qi << "\n";
        for (int i = 0; i < (int)approx.size(); ++i) {
            out << "Nearest neighbor-" << (i+1) << ": " << approx[i].first << "\n";
        }
    }
    std::cout << "[Hypercube] Results saved to " << cfg.output_path << "\n";
}

// --- FIX: Output to file for Python compatibility ---
void run_ivfflat(const Matrix& base, const Matrix& queries, const Config& cfg) {
    using namespace std::chrono;
    std::ofstream out(cfg.output_path);
    if(!out){ std::cerr << "[ERROR] Could not open output file: " << cfg.output_path << "\n"; return; }
    out << "IVF_Flat\n";

    std::cout << "[IVFFlat] Building index...\n";
    int train_subset = (int)std::sqrt((double)base.n);
    auto ivf = build_ivf_flat(base, cfg.kclusters, cfg.seed, train_subset);

    double sumApprox = 0.0;
    for (int qi = 0; qi < queries.n; ++qi) {
        std::vector<float> q(queries.row(qi), queries.row(qi) + queries.d);

        auto t0 = high_resolution_clock::now();
        auto ans = ivf_flat_query_topN(ivf, base, q.data(), cfg.nprobe, cfg.N);
        auto t1 = high_resolution_clock::now();
        sumApprox += duration_cast<microseconds>(t1 - t0).count() / 1000.0;

        // WRITE TO FILE
        out << "Query: " << qi << "\n";
        for (int i = 0; i < (int)ans.ids.size(); ++i) {
            out << "Nearest neighbor-" << (i+1) << ": " << ans.ids[i] << "\n";
        }
    }
    
    out << "tApproximateAverage: " << (sumApprox / queries.n) << "\n";
    std::cout << "[IVFFlat] Results saved to " << cfg.output_path << "\n";
}

// --- FIX: Output to file for Python compatibility ---
void run_ivfpq(const Matrix& base, const Matrix& queries, const Config& cfg) {
    using namespace std::chrono;
    std::ofstream out(cfg.output_path);
    if(!out){ std::cerr << "[ERROR] Could not open output file: " << cfg.output_path << "\n"; return; }
    out << "IVF_PQ\n";

    std::cout << "[IVFPQ] Building index...\n";
    int train_subset = std::min(base.n, 1000);
    auto ivf = build_ivf_pq(base, cfg.kclusters, cfg.M_pq, cfg.nbits, cfg.seed, train_subset);

    double sumApprox = 0.0;
    for (int qi = 0; qi < queries.n; ++qi) {
        std::vector<float> q(queries.row(qi), queries.row(qi) + queries.d);

        auto t0 = high_resolution_clock::now();
        auto ans = ivf_pq_query_topN(ivf, base, q.data(), cfg.nprobe, cfg.N);
        auto t1 = high_resolution_clock::now();
        sumApprox += duration_cast<microseconds>(t1 - t0).count() / 1000.0;

        // WRITE TO FILE
        out << "Query: " << qi << "\n";
        for (int i = 0; i < (int)ans.ids.size(); ++i) {
            out << "Nearest neighbor-" << (i+1) << ": " << ans.ids[i] << "\n";
        }
    }
    out << "tApproximateAverage: " << (sumApprox / queries.n) << "\n";
    std::cout << "[IVFPQ] Results saved to " << cfg.output_path << "\n";
}

// Keep your original build_knn logic
void run_build_knn(const Matrix& base, const Config& cfg) {
    using namespace std;
    const int n = base.n;
    const int d = base.d;
    const int K = cfg.knn_k;

    string method = cfg.knn_method;  
    for (auto& c : method) c = std::tolower(c);

    string out_path = cfg.index_path;
    if (out_path.empty()) out_path = "knn_graph_" + method + "_K" + std::to_string(K) + ".bin";

    cout << "[build_knn] Building kNN using method = " << method << endl;

    vector<vector<float>> base_vecs(n);
    for (int i = 0; i < n; i++)
        base_vecs[i] = vector<float>(base.row(i), base.row(i) + d);

    vector<int> knn_idx(n * K, -1);

    if (method == "lsh") {
        lsh::LSH index(d, cfg.k, cfg.L, cfg.w, -1, cfg.seed);
        index.buildIndex(base_vecs);
        for (int i = 0; i < n; i++) {
            auto neigh = index.searchKNN(base_vecs[i], K+1);
            int count = 0;
            for (auto& p : neigh) {
                if (p.first == i) continue;
                if (count < K) knn_idx[i*K + count++] = p.first;
            }
        }
    }
    else if (method == "cube") {
        cube::Hypercube hc(d, cfg.kproj, cfg.w, cfg.M, cfg.probes, cfg.seed);
        hc.buildIndex(base_vecs);
        for (int i = 0; i < n; i++) {
            auto neigh = hc.searchKNN(base_vecs[i], K+1);
            int count = 0;
            for (auto& p : neigh) {
                if (p.first == i) continue;
                if (count < K) knn_idx[i*K + count++] = p.first;
            }
        }
    }
    else if (method == "ivf") {
        int train_subset = (int)std::sqrt((double)n);
        auto ivf = build_ivf_flat(base, cfg.kclusters, cfg.seed, train_subset);
        for (int i = 0; i < n; i++) {
            auto ans = ivf_flat_query_topN(ivf, base, base.row(i), cfg.nprobe, K+1);
            int count = 0;
            for (int id : ans.ids) {
                if (id == i) continue;
                if (count < K) knn_idx[i*K + count++] = id;
            }
        }
    }
    else if (method == "pq") {
        int train_subset = (int)std::sqrt((double)n);
        auto ivf = build_ivf_pq(base, cfg.kclusters, cfg.M_pq, cfg.nbits, cfg.seed, train_subset);
        for (int i = 0; i < n; i++) {
            auto ans = ivf_pq_query_topN(ivf, base, base.row(i), cfg.nprobe, K+1);
            int count = 0;
            for (int id : ans.ids) {
                if (id == i) continue;
                if (count < K) knn_idx[i*K + count++] = id;
            }
        }
    }
    brute::save_knn_binary(out_path, knn_idx, n, K);
    std::cout << "[build_knn] Saved kNN graph to " << out_path << endl;
}