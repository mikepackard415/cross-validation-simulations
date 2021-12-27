# cross-validation-simulations
This repository holds the python code to run simulations that compare model-selection procedures across sample sizes and data-generating processes. The details and results are presented in a paper authored by Ai Deng (PhD) entitled hv-block Cross Validation: Some Theoretical and Finite Sample Properties.

The files included in this repository are:

1) main.py sets parameters and runs all simulations.

2) dgp.py generates synthetic datasets.

3) cv.py, and bic.py contain the cross validation (CV) and bayesian information criterion (BIC) procedures for evaluating models.

4) mse.py calculates mean squared errors using matrix algebra.

5) comp.py compares the performance of each procedure in selecting the correct model.

6) export.py exorts the results to an excel file. If a path for an excel file is not specified in main.py, the 
   code prints the results in the console.

7) CV Simulations.xlsx includes the simulation results for the parameters described in the paper:
   * 2,000 simulations
   * Six sample sizes: 50, 100, 200, 250, 500, 1000
   * Five model-selection procedures: v-block equal cv, v-block unequal cv, hv-block equal cv, hv-block unequal cv, and BIC.
