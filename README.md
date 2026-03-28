# Spectral Correlation Estimation Algorithm Tests

4 main tests:
- Validation/Accuracy ---> validation_test_example
- Cycle Leakage Test ---> cycle_leakage_test_example
- Speed and Memory ---> speed_memory_test_example
- Performance (Cycle Frequency Detection) --> roc_test_example

Extra Functions:
- Cyclic Domain Plot for Extra Validation
- Spectral Coherence Validation

## Recommended Usage Order
1. Validation (including coherence and cyclic domain profile)

2. Speed and Memory test

3. Cycle Leakage Test and Window test

4. Performance Test

## Note

The FFT Accumulation Method (FAM) algorithm using high mainlobe windows like flattop needs L = Np/8 instead of the recommended L = Np/4 to avoid cycle leakage.

FastSC python implementation is from: [[https://github.com](https://github.com/rodrigoel/RBFastSC)](https://github.com/rodrigoel/RBFastSC)

## Updates

26/03/2026:

- Update README with gallery

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


