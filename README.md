# Data-Mining
Before running the code, please make sure LSH.py and data.txt are under the same directory.

To run LSH on the dataset, type in:
    python LSH.py

To modify k, the number of hash functions, please go to line 99 of LSH.py and modify it.
Sorry for this inconvinence, this is due to the parallel implementation.

During running, you will be asked to input number of concurrent processes. The recommended value
is two times of CPU cores, but if you are not sure please type in:
    4
