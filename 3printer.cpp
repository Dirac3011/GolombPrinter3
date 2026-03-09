#include <iostream>
#include <vector>
#include <chrono>
#include <atomic>
#include <omp.h>
#include <mutex>
#include <fstream>
#include <filesystem>
#include <iomanip>

using namespace std;
namespace fs = std::filesystem;

typedef __uint128_t bitmask;

/**
 * g_min_span: Minimum diameter a(n) for gamma=3 (A392462).
 * These are used for aggressive pruning.
 */
const int g_min_span[] = {
    0,  0,  1,  2,  3,  5,  7, 10, 13, 16, 
    20, 25, 30, 35, 42, 49, 58, 67, 76, 85
};

vector<vector<int>> all_found_sets;
mutex g_data_mutex;
atomic<bool> g_any_found_this_alpha(false);

void save_to_json(int n, int alpha, double elapsed) {
    if (!fs::exists("data")) fs::create_directory("data");
    
    string filename = "data/y3_n" + to_string(n) + "_a" + to_string(alpha) + ".json";
    ofstream file(filename);

    file << "{\n";
    file << "  \"n\": " << n << ",\n";
    file << "  \"alpha\": " << alpha << ",\n";
    file << "  \"gamma\": 3,\n";
    file << "  \"count\": " << all_found_sets.size() << ",\n";
    file << "  \"time_seconds\": " << fixed << setprecision(4) << elapsed << ",\n";
    file << "  \"sets\": [\n";

    for (size_t i = 0; i < all_found_sets.size(); ++i) {
        file << "    [";
        for (size_t j = 0; j < all_found_sets[i].size(); ++j) {
            file << all_found_sets[i][j] << (j == all_found_sets[i].size() - 1 ? "" : ", ");
        }
        file << "]" << (i == all_found_sets.size() - 1 ? "" : ",") << "\n";
    }

    file << "  ]\n";
    file << "}\n";
}

void backtrack(int* current_set, int size, bitmask s1, bitmask s2, bitmask s3, int n_target, int alpha) {
    if (size >= n_target) return;

    int last_val = current_set[size - 1];
    int remaining = n_target - size;

    // Pruning: if we don't have enough room left to fit the remaining elements
    if (last_val + g_min_span[remaining + 1] > alpha) return;

    // Base Case: We are looking for the last element (which MUST be alpha)
    if (size == n_target - 1) {
        for (int i = 0; i < size; ++i) {
            int d = alpha - current_set[i];
            if ((s3 >> d) & 1) return; // Difference already appears 3 times
        }
        
        vector<int> found(n_target);
        for(int i=0; i<size; ++i) found[i] = current_set[i];
        found[n_target-1] = alpha;

        lock_guard<mutex> lock(g_data_mutex);
        all_found_sets.push_back(found);
        g_any_found_this_alpha.store(true, std::memory_order_relaxed);
        return; 
    }

    int max_cand = alpha - g_min_span[remaining];

    for (int cand = last_val + 1; cand <= max_cand; ++cand) {
        bitmask new_bits = 0;
        bool invalid = false;
        
        for (int i = 0; i < size; ++i) {
            int d = cand - current_set[i];
            if ((s3 >> d) & 1) { invalid = true; break; }
            new_bits |= ((bitmask)1 << d);
        }

        if (invalid) continue;

        current_set[size] = cand;
        // Update layer bitmasks: 
        // s1: differences appearing >= 1 time
        // s2: differences appearing >= 2 times
        // s3: differences appearing >= 3 times
        backtrack(current_set, size + 1, 
                  s1 | new_bits, 
                  s2 | (s1 & new_bits), 
                  s3 | (s2 & new_bits), 
                  n_target, alpha);
    }
}

int main() {
    cout << "Batch Gamma=3 Search | N=0 to 17 | Full Search (No Symmetry Breaking)" << endl;
    cout << "----------------------------------------------------------------------" << endl;

    for (int n = 0; n <= 17; ++n) {
        // Handle n < 2 separately to avoid loop logic issues
        if (n <= 2) {
            all_found_sets.clear();
            if (n == 0) all_found_sets.push_back({});
            else if (n == 1) all_found_sets.push_back({0});
            else if (n == 2) all_found_sets.push_back({0, 1});
            
            int alpha = (n == 2) ? 1 : 0;
            save_to_json(n, alpha, 0.0);
            cout << "n=" << n << " | alpha=" << alpha << " | Found: " << all_found_sets.size() << endl;
            continue;
        }

        int alpha = g_min_span[n]; 
        bool solved = false;

        while (!solved) {
            all_found_sets.clear();
            g_any_found_this_alpha.store(false);
            auto start_time = chrono::high_resolution_clock::now();

            #pragma omp parallel
            {
                int t_set[32];
                t_set[0] = 0;
                
                // Symmetry breaking removed: c1 now goes up to the maximum possible value
                int max_c1 = alpha - g_min_span[n - 1];

                #pragma omp for schedule(dynamic, 1)
                for (int c1 = 1; c1 <= max_c1; ++c1) {
                    t_set[1] = c1;
                    backtrack(t_set, 2, ((bitmask)1 << c1), 0, 0, n, alpha);
                }
            }

            double elapsed = chrono::duration<double>(chrono::high_resolution_clock::now() - start_time).count();

            if (g_any_found_this_alpha.load()) {
                save_to_json(n, alpha, elapsed);
                cout << "n=" << n << " | alpha=" << alpha << " | Found: " << all_found_sets.size() << " | Time: " << elapsed << "s" << endl;
                solved = true;
            } else {
                alpha++; 
            }
        }
    }
    return 0;
}