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
 
 We are interested not only in comparing the performance of each solver but also the differences 
between using them with or without calling the [Neos Server](https://neos-server.org/neos/), as well as to compare the 
 efficiency of using them through [AMPL](https://ampl.com/). Hence, three lists of solvers are created from the previous one.

The first list is formed by the solvers whose executable file are included in the [AMPL license](https://ampl.com/try-ampl/request-a-full-trial/#Form):

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

 
The next step consists of running the optimization problem for the different solvers enumerated in the previous three lists.
Since the optimization problems we are considering in this repo are (highly) non linear, they may stuck at local optimal.
Hence, a multistart approach is applied in which the same problem is run several times starting from different initial solutions,
randomly chosen from the feasible region. The number of runs at the multistart is defined by the user.

The objective value and the computational time will be used as measure of performance. Particularly, we provide the average,
the maximum and minimum of all the objective values obtained after running the multistart, as well as the average, the 
maximum and minimum value of the computational time. The results of each optimization problem are summarized in a table which,
apart of the performance values previously mentioned, includes extra information such as the name of the problem, if the Neos Server 
or AMPL is used or not, the solver utilized, the number of variables and constraints involved, and also 
the sense of the optimization problem, i.e., if the objective is to minimize or maximize certain objective function. An 
example of the results is given below:

| problem  | neos | ampl | solver | #variables | #constraints | sense | mean obj. val. | max obj. val.| min obj. val. | mean comp. time | max comp. time | min comp. time |
| -------- | ---- | ---- | ------ | ---------- | ------------ | ----- | -------------- | ------------ | ------------- | --------------- | -------------- | -------------- |
| Problem 1| yes | no   | conopt |   100      |     200      | min   |  1.2           |     5.7      | 0.23          |    180          | 300            | 50             |         



## Examples

This section briefly explains the organization of the examples used here. This repo contains one folder per optimization problem.
Each folder contains three new folders and a script called `main_name_of_the_problem.py`. Such a script is the only one that should be executed
when running the experiments. Moreover, the first folder, entitled `model_pdf` contains the document with the model formulation. The second folder, `optimization_problem_utils` is formed
by the scripts which solves the optimization problem, so *be careful when modifying it*. Do not hesitate to contact the person who has written this code.
Finally, the third folder entitled `results_name_of_the_problem` includes the results obtained in all the runs of the multistart, and a
summary of them, called `summary_results.csv`.

The following table provides the optimization problems which have been used for the non linear solvers comparison. This table 
includes the name of the problem, the problem formulation and the folder with the obtained results.

| # | Name  | Model Formulation | Folder results |
| - | ----- | ------------------| -------------- |
| 1 | M2SVM_Optimal_Weights | [M2SVM_Optimal_Weights Model Formulation](./M2SVM_Optimal_Weights/model_pdf/M2SVM_Optimal_Weights.pdf)  |[M2SVM_Optimal_Weights Results](./M2SVM_Optimal_Weights/results_m2svm_optimal_weights) |
| 2 | MINLP_Trigonometric_Functions | [MINLP_Trigonometric_Functions Model Formulation](./MINLP_Trigonometric_Functions/model_pdf/MINLP_Trigonometric_Functions.pdf)  |[MINLP_Trigonometric_Functions Results](./MINLP_Trigonometric_Functions/results_MINLP_Trignometric_Functions) |

More details about the results obtained in each example are given in next sections.

### Example 1: M2SVM Optimal Weights

The aim of the optimization problem formulated [here](./M2SVM_Optimal_Weights/model_pdf/M2SVM_Optimal_Weights.pdf) is to
 seek the optimal weights in the Gaussian kernel in order to obtain a good classification with the Support Vector Machine. Since the problem is based on
  a toy example, the size of this optimization problem is small. Particularly, it is formed by **15 continuous variables**
  and **36 constraints**. 
  
  A multistart approach with **1000 runs** is run for the different solvers as explained in previous section. A summary of the
  results obtained can be seen in the following table. If further information is necessary, we suggest the reader to download the summary
  [here](./M2SVM_Optimal_Weights/results_m2svm_optimal_weights/summary_results.csv) or to have a look at the results of the different runs in [this folder](./M2SVM_Optimal_Weights/results_m2svm_optimal_weights).
  
  | problem  | neos | ampl | solver | #variables | #constraints | sense | mean obj. val. | max obj. val.| min obj. val. | mean comp. time | max comp. time | min comp. time |
| -------- | ---- | ---- | ------ | ---------- | ------------ | ----- | -------------- | ------------ | ------------- | --------------- | -------------- | -------------- |
m2svm_optimal_weights|no|yes|conopt|15|36|min|-6.46E-14|1.45E-15|-3.61E-13|0.15|0.35|0.07
m2svm_optimal_weights|no|yes|loqo|15|36|min|-1.37E+12|9.00E+03|-3.07E+12|0.30|1.45|0.11
m2svm_optimal_weights|no|yes|minos|15|36|min|-2.79E-14|0.00E+00|-8.56E-14|0.11|0.49|0.06
m2svm_optimal_weights|no|yes|snopt|15|36|min|-1.05E-14|4.44E-16|-5.81E-14|0.10|0.20|0.06
m2svm_optimal_weights|no|no|ipopt|15|36|min|-7.50E-08|-7.49E-08|-7.50E-08|0.11|0.30|0.10
m2svm_optimal_weights|no|no|bonmin|15|36|min|-9.97E-08|-9.38E-08|-9.99E-08|0.12|0.34|0.07
m2svm_optimal_weights|no|no|couenne|15|36|min|-2.88E-13|-2.88E-13|-2.88E-13|0.67|14.98|0.18
m2svm_optimal_weights|yes|no|conopt|15|36|min|-6.46E-14|1.45E-15|-3.61E-13|18.05|73.97|2.38
m2svm_optimal_weights|yes|no|ipopt|15|36|min|-7.50E-08|-7.49E-08|-7.50E-08|17.31|38.74|2.38
m2svm_optimal_weights|yes|no|filter|15|36|min|-1.74E-13|0.00E+00|-2.86E-13|18.51|29.40|2.59
m2svm_optimal_weights|yes|no|knitro|15|36|min|1.87E-07|4.85E-06|-9.89E-10|18.11|66.91|2.41
m2svm_optimal_weights|yes|no|loqo|15|36|min|-1.13E+12|1.89E+04|-2.98E+12|17.85|66.01|2.47
m2svm_optimal_weights|yes|no|minos|15|36|min|-2.79E-14|0.00E+00|-8.56E-14|18.15|38.81|2.78
m2svm_optimal_weights|yes|no|mosek|15|36|min|3.33E-09|3.37E-09|3.33E-09|17.41|82.82|2.39
m2svm_optimal_weights|yes|no|snopt|15|36|min|-1.05E-14|4.44E-16|-5.81E-14|18.16|39.06|2.41
m2svm_optimal_weights|yes|no|bonmin|15|36|min|-9.97E-08|-9.38E-08|-9.99E-08|18.04|41.91|2.60
m2svm_optimal_weights|yes|no|couenne|15|36|min|-2.88E-13|-2.88E-13|-2.88E-13|17.72|76.28|2.55
m2svm_optimal_weights|yes|no|filmint|15|36|min|-2.77E-13|-2.77E-13|-2.77E-13|17.21|66.69|2.56
  
  
  In spite of the small size of the problem, the results allow us to get conclusions about the different solvers. 
  
  Regarding the objective values, we observe that in all the cases a mean value close to zero is obtained, except in the `loqo` solver, which
  at first sight seems to be the best one. However, the optimal solution obtained with `loqo` (which can be downloaded [here](./M2SVM_Optimal_Weights/results_m2svm_optimal_weights/results_by_line_2_random_neos_flag_False_ampl_flag_True_solver_loqo.pydata) and [here](./M2SVM_Optimal_Weights/results_m2svm_optimal_weights/results_by_line_2_random_neos_flag_True_ampl_flag_False_solver_loqo.pydata))
  is numerically unstable, since values of order 1E+25 are obtained. Therefore, `loqo` is not a good solver for our problem, but 
  any of the remaining ones can be used.
  
  With respect to the computational time, we see that there is a difference mean of three order of magnitude between using the solvers through the Neos server
  or directly using the executable files. The reason of this issue is that solving an optimization problem via Neos not 
  only includes the computational cost associating to the problem resolution but also the calls to the server which significantly increase the total elapsed time.
  Looking at that solvers that have been used without Neos, we check that `couenne` is the one that spent most time solving the problem, and then is not advisable to use.
  
  Thus, taking into account the previous comments, for this particular optimization problem, we suggest to use any of the following solvers without Neos: `conopt`,
  `minos`, `snopt`, `ipopt` and `bonmin`.
  
  ### Example 2: MINLP Trigonometric Functions
  
  In this example, we solve a Mixed Integer Non Linear Programming (MINLP) problem with trigonometric functions involved.
  Contrary to the Example 1, this problem is of medium size, since it is formed by **300 variables (150 integer and 150 continuous)**
  and **153 constraints**.
  
  The results of the different solvers after solving the multistart with **100 runs** are shown in the following table. For more details
  the reader is referred to [this file](./MINLP_Trigonometric_Functions/results_MINLP_Trignometric_Functions/summary_results.csv)
   and [this folder](./MINLP_Trigonometric_Functions/results_MINLP_Trignometric_Functions):
   
  | problem  | neos | ampl | solver | #variables | #constraints | sense | mean obj. val. | max obj. val.| min obj. val. | mean comp. time | max comp. time | min comp. time |
| -------- | ---- | ---- | ------ | ---------- | ------------ | ----- | -------------- | ------------ | ------------- | --------------- | -------------- | -------------- |
 MINLP_trigonometric_functions|no|yes|conopt|300|153|min|24.63|122.80|-165.20|8.28|24.72|0.99
MINLP_trigonometric_functions|no|yes|loqo|300|153|min|-7.63|4.33|-22.19|3.45|14.33|1.77
MINLP_trigonometric_functions|no|yes|minos|300|153|min|39.47|463.60|-417.00|1.74|3.13|0.73
MINLP_trigonometric_functions|no|yes|snopt|300|153|min|30.89|463.60|-228.50|3.16|9.51|0.54
MINLP_trigonometric_functions|no|no|ipopt|300|153|min|3.27|64.99|-92.47|57.32|198.60|15.01
MINLP_trigonometric_functions|no|no|bonmin|300|153|min|8.08|196.60|-9.60|89.60|327.60|10.15
MINLP_trigonometric_functions|yes|no|conopt|300|153|min|24.43|87.75|-200.60|19.14|25.74|9.51
MINLP_trigonometric_functions|yes|no|ipopt|300|153|min|6.24|64.99|-9.12|22.11|57.23|9.91
MINLP_trigonometric_functions|yes|no|filter|300|153|min|27.43|62.73|-9.65|19.68|38.31|15.17
MINLP_trigonometric_functions|yes|no|knitro|300|153|min|-4.54|62.66|-9.26|19.54|43.99|7.20
MINLP_trigonometric_functions|yes|no|loqo|300|153|min|-7.16|-0.23|-9.50|19.77|39.89|9.05
MINLP_trigonometric_functions|yes|no|minos|300|153|min|39.47|463.60|-417.00|18.92|28.98|8.31
MINLP_trigonometric_functions|yes|no|mosek|300|153|min|0.00|0.00|0.00|18.51|23.19|4.77
MINLP_trigonometric_functions|yes|no|snopt|300|153|min|31.40|299.30|-22.19|19.49|37.79|5.63
MINLP_trigonometric_functions|yes|no|bonmin|300|153|min|7.13|64.26|-9.72|22.33|48.24|6.62