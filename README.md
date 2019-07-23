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


## How is comparison testing performed?

In this section, we explain the strategy applied to compare the different solvers. Let us consider the following list of
 non linear solvers used in our examples to compare their performance:

* [bonmin](https://projects.coin-or.org/Bonmin)
* [conopt](http://www.conopt.com/)
* [couenne](https://projects.coin-or.org/Couenne)
* [filmint](https://www.swmath.org/software/6197)
* [filter](https://neos-server.org/neos/solvers/nco:filter/AMPL.html)
* [ipopt](https://projects.coin-or.org/Ipopt/wiki)
* [knitro](https://www.artelys.com/solvers/knitro/)
* [loqo](https://neos-server.org/neos/solvers/nco:LOQO/AMPL.html)
* [minos](https://www.aimms.com/english/developers/resources/solvers/minos/)
* [mosek](https://www.mosek.com/)
* [snopt](https://web.stanford.edu/group/SOL/snopt.htm)
 
 We are interested not only in comparing the performance of the solver but also the differences 
between using them with or without calling the [Neos Server](https://neos-server.org/neos/), as well as to compare the 
 efficiency of using them through [AMPL](https://ampl.com/). Hence, three lists of solvers are created from the previous one.

The first list is formed by the solvers whose executable file are included in the [AMPL license](https://ampl.com/try-ampl/request-a-full-trial/#Form).

**List solvers AMPL:**

* [conopt](http://www.conopt.com/)
* [loqo](https://neos-server.org/neos/solvers/nco:LOQO/AMPL.html)
* [minos](https://www.aimms.com/english/developers/resources/solvers/minos/)
* [snopt](https://web.stanford.edu/group/SOL/snopt.htm)

The solvers of the second list are open source and the executable files can be downloaded [here](https://ampl.com/products/solvers/open-source/).
Thus, it is not necessary to use them via the Neos Server:

**List solvers without Neos**

* [bonmin](https://projects.coin-or.org/Bonmin)
* [couenne](https://projects.coin-or.org/Couenne)
* [ipopt](https://projects.coin-or.org/Ipopt/wiki)

Finally, the third list is composed by those solvers that require license for their use. In order to avoid such an issue, 
we run them with the help of the Neos Server:

 **List solver with Neos**
 
* [bonmin](https://projects.coin-or.org/Bonmin)
* [conopt](http://www.conopt.com/)
* [couenne](https://projects.coin-or.org/Couenne)
* [filmint](https://www.swmath.org/software/6197)
* [filter](https://neos-server.org/neos/solvers/nco:filter/AMPL.html)
* [ipopt](https://projects.coin-or.org/Ipopt/wiki)
* [knitro](https://www.artelys.com/solvers/knitro/)
* [loqo](https://neos-server.org/neos/solvers/nco:LOQO/AMPL.html)
* [minos](https://www.aimms.com/english/developers/resources/solvers/minos/)
* [mosek](https://www.mosek.com/)
* [snopt](https://web.stanford.edu/group/SOL/snopt.htm)

 

## Examples

The following table summarizes the optimization problems which have been used for the non linear solvers comparison. This table 
includes the name of the problem, the problem formulation and the folder with the obtained results.

| # | Name  | Model Formulation | Folder results |
| - | ----- | ------------------| -------------- |
| 1 | M2SVM_Optimal_Weights | [M2SVM_Optimal_Weights Model Formulation](./M2SVM_Optimal_Weights/model_pdf/M2SVM_Optimal_Weights.pdf)  |[M2SVM_Optimal_Weights Results](./M2SVM_Optimal_Weights/results_m2svm_optimal_weights) |
| 2 | MINLP_Trigonometric_Functions | [MINLP_Trigonometric_Functions Model Formulation](./MINLP_Trigonometric_Functions/model_pdf/MINLP_Trigonometric_Functions.pdf)  |[MINLP_Trigonometric_Functions Results](./MINLP_Trigonometric_Functions/results_MINLP_Trignometric_Functions) |

More details about the results obtained in each example are provided in next sections.

### M2SVM Optimal Weights

The aim of the optimization problem formulated [here](./M2SVM_Optimal_Weights/model_pdf/M2SVM_Optimal_Weights.pdf) is to
 seek the optimal weights in the Gaussian kernel in order to obtain a good classification. Since the problem is based on
  a toy example, the size of this optimization problem is small. Particularly, it is formed by 15 continuous variables 
  and 36 constraints. 
  
  
  
  
  In spite of the small size of the problem, the results allow us to get conclusions about the different solvers.