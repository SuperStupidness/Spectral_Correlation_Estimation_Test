# Spectral Correlation Estimation Algorithm Tests

4 main tests:
- Validation/Accuracy ---> validation_test_example
- Cycle Leakage Test ---> cycle_leakage_test_example
- Speed and Memory ---> speed_memory_test_example
- Performance (Cycle Frequency Detection) --> roc_test_example

Extra Functions:
- Cyclic Domain Plot for Extra Validation
- Spectral Coherence Validation

## Recommended Usage
1. Validation (including coherence and cyclic domain profile)

2. Speed and Memory test

3. Cycle Leakage Test and Window test

4. Performance Test

## Note

The FFT Accumulation Method (FAM) algorithm using high mainlobe windows like flattop needs L = Np/8 instead of the recommended L = Np/4 to avoid cycle leakage.

(image here)

(image here)

## Gallery

(To be done)

## Updates




