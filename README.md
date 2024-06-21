# Using Data-Driven Methods to Analyze the Roles of Different Elements in Liquid Metal Batteries

This repository contains the code and datasets used in the paper titled "Using Data-Driven Methods to Analyze the Roles of Different Elements in Liquid Metal Batteries". Below is an overview of the structure and contents:

### Datasets
- The dataset consisting of 63 data points used in the paper is located in the `datasets` folder.

### Methods
- **Combinations:** The `method` folder contains 8 elements plus different current density combinations, detailed in `combinations.csv`. You can generate this combinations file by running:
`python 1.get_all_combinations.py`

- **Models:** Implementation of the three models discussed in the paper can also be found in the `method` folder. The code and models are provided to reproduce the results mentioned at the end of each code snippet. To replicate these results, follow the sequential execution order provided in the code comments.

### Environment
- Python 3.11.4
- The implementation of Gaussian Process Regression uses [GPyTorch](https://github.com/cornellius-gp/gpytorch). Make sure to install this library before running the code.
