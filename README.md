# Comparison_non_linear_solvers


Do you want to solve a non-linear optimization problem? Have you doubts about what solver is the best? Then, you are in 
the correct place! =) Continue reading and have a look at this repo.

## Goals

This repository aims to perform a comparison of several of the available non linear solvers. Such a comparison is made in 
terms of the objective value and the computational time.

To do this, some non linear optimization problems of different nature are modelled using Pyomo, and run with different
solvers. The results are saved and analyse below to get conclusions.

It is important to note that the conclusions draw here cannot be extended to all the existing optimization problems.
This repo just serves as a guide for your decisions, but it is not an universal truth. Please take this into account when 
deciding which solver is more suitable in your case.

## Examples

The following table summarizes the optimization problems which have been used for the non linear solvers comparison. This table 
includes the name of the problem, the problem formulation and the folder with the obtained results.

| # | Name  | Model Formulation | Folder results |
| - | ----- | ------------------| -------------- |
| 1 | M2SVM_Optimal_Weights | [M2SVM_Optimal_Weights Model Formulation](./M2SVM_Optimal_Weights/model_pdf/M2SVM_Optimal_Weights.pdf)  |[M2SVM_Optimal_Weights Results](./M2SVM_Optimal_Weights/results_m2svm_optimal_weights) |
| 1 | MINLP_Trigonometric_Functions | [MINLP_Trigonometric_Functions Model Formulation](./MINLP_Trigonometric_Functions/model_pdf/MINLP_Trigonometric_Functions.pdf)  |[M2SVM_Optimal_Weights Results](./M2SVM_Optimal_Weights/results_m2svm_optimal_weights) |

