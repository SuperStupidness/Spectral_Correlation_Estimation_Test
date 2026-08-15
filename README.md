# Spectral Correlation Estimation Algorithm Tests

4 main tests:
- Validation/Accuracy ---> validation_test_example
- Cycle Leakage Test ---> cycle_leakage_test_example
- Speed and Memory ---> speed_memory_test_example
- Performance/Robustness (Cycle Frequency Detection) --> roc_test_example

Extra Functions:
- Cyclic Domain Plot for Extra Validation
- Spectral Coherence Validation

## Recommended Usage Order
1. Validation (including coherence and cyclic domain profile)

2. Speed and Memory test

3. Cycle Leakage Test and Window test

4. Performance/Robustness Test

## Available Algorithm Implemetations

- Strip Spectral Correlation Algorithm (SSCA) [1]
- FFT Accumulation Method (FAM) [1]
- Fast Spectral Correlation (FastSC) [2]
- Spectral Correlation Function using 2D FFT (Scf 2D FFT) [3]
- Sparse Strip Spectral Correlation Algorithm (S3CA) [4]

## Note

The FFT Accumulation Method (FAM) algorithm using wide mainlobe windows like flattop needs L = Np/8 instead of the recommended L = Np/4 to avoid cycle leakage.

FastSC python implementation is from: [RBFastSC](https://github.com/rodrigoel/RBFastSC) by Rodrigo Emanoel de Britto Andrade Barros

The [CSPBlog](https://cyclostationary.blog/) is also very good resources for implementing your own FAM and SSCA algorithms along with all the theory behind it. Here is the guide for [FAM](https://cyclostationary.blog/2018/06/01/csp-estimators-the-fft-accumulation-method/), [SSCA](https://cyclostationary.blog/2016/03/22/csp-estimators-the-strip-spectral-correlation-analyzer/), and why coherence is useful which is in SSCA guide.

For a more step by step guide for SSCA, check out: Eric April, “On the implementation of the strip spectral correlation
algorithm for cyclic spectrum estimation,” Tech. Rep., DEFENCE
RESEARCH ESTABLISHMENT OTTAWA (ONTARIO), 1994.

## General Findings
SSCA with flattop and boxcar window is best for general use. It has a good combination of speed and performance. FAM is best for speed and low memory usage with a performance trade-off. FastSC is good for fast analysis of small alpha range around 0, usually 0.01 fs or lower. SCF 2D FFT has the best performance/robustness but it has high memory usage (O(N^2) where N is the signal length).

## Updates

15/08/2026

- README Update. Add references.

26/03/2026:

- Update README with gallery

## References

1. R. S. Roberts, W. A. Brown and H. H. Loomis, "Computationally efficient algorithms for cyclic spectral analysis," in IEEE Signal Processing Magazine, vol. 8, no. 2, pp. 38-49, April 1991, doi: 10.1109/79.81008.

2. Jérôme Antoni, Ge Xin, Nacer Hamzaoui,
Fast computation of the spectral correlation,
Mechanical Systems and Signal Processing, Volume 92, 2017,
Pages 248-277,
ISSN 0888-3270,
https://doi.org/10.1016/j.ymssp.2017.01.011.
(https://www.sciencedirect.com/science/article/pii/S0888327017300134)

3. T. Shevgunov and E. Efimov, "Two-dimensional FFT Algorithm for Estimating Spectral Correlation Function of Cyclostationary Random Processes," 2019 Signal Processing Symposium (SPSympo), Krakow, Poland, 2019, pp. 216-220, doi: 10.1109/SPS.2019.8881963.

4. C. J. Li, R. Rademacher, D. Boland, C. T. Jin, C. M. Spooner and P. H. W. Leong, "S 3 CA: A Sparse Strip Spectral Correlation Analyzer," in IEEE Signal Processing Letters, vol. 31, pp. 646-650, 2024, doi: 10.1109/LSP.2024.3364062.

## Gallery

*Validation*

![SSCA Validation](/fig/ssca_validation_1024.png)
![SSCA Validation](/fig/ssca_validation_8192.png)
![SSCA Validation](/fig/ssca_validation_65536.png)
![SSCA Validation](/fig/ssca_validation_524288.png)

*Cycle Leakage Test*

![SSCA Cycle Leakage](/fig/ssca_cycle_leakage_test.png)

*Speed and Memory*

![Speed Test](/fig/all_algo_benchmark.png)
![Memory Test](/fig/memory_test.png)

*Performance Test*

![Performance Test](/fig/ssca_roc_test.png)

*Coherence Validation*

![SSCA Coherence Validation](/fig/ssca_coh_validation_1024.png)
![SSCA Coherence Validation](/fig/ssca_coh_validation_8192.png)
![SSCA Coherence Validation](/fig/ssca_coh_validation_65536.png)
![SSCA Coherence Validation](/fig/ssca_coh_validation_262144.png)

*Cyclic Domain Profile Test*

![SSCA CDP](/fig/cdp_test.png)

*Window Test (Extended Cycle Leakage Test)*

![SSCA Window Test](/fig/ssca_window_test.png)

*Extended Validation Test*

![SSCA Extended Validation](/fig/ssca_extended_validation_test.png)

