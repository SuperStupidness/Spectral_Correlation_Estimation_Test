// algorithm/s3ca/src/python_bridge.cpp
//
// ctypes-compatible C bridge for cyclostationary::single_S3CA.
// Exposes a single extern "C" function so the Python loader in
// algorithm/s3ca/__init__.py can call it directly via ctypes.CDLL.
//
// Build (MSYS2 UCRT64 shell, from this directory):
//
//   g++ -O3 -shared python_bridge.cpp autossca.cpp computefourier.cc \
//       filters.cc parameters.cc utils.cc fftw.cc \
//       -o s3ca.dll -lfftw3f
//
// Do NOT include run_autossca.cpp — it has its own main() and will
// cause linker errors.

#include "autossca.h"
#include <vector>
#include <map>
using namespace cyclostationary;

extern "C" {
    int compute_s3ca_in_memory(float* i_data, float* q_data, int data_length,
                               int size_of_n, int np_channels, int seeds,
                               unsigned int* out_keys, float* out_values,
                               int max_out_size)
    {
        // Zero-padded buffer: size_of_n + np_channels (algorithm's windowing tail).
        std::vector<complex_t> input_signal(size_of_n + np_channels, complex_t(0, 0));
        int mylength = size_of_n > data_length ? data_length : size_of_n;
        for (int i = 0; i < mylength; i++)
            input_signal[i] = complex_t(i_data[i], q_data[i]);

        auto Sx_S3CA = single_S3CA(input_signal, np_channels, size_of_n, seeds);

        int count = 0;
        for (auto const& [key, val] : Sx_S3CA) {
            if (count >= max_out_size) break;
            out_keys[count] = key;
            out_values[count] = val;
            count++;
        }
        return count;
    }
}
