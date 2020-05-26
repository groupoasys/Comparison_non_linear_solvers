 # Comparison_non_linear_solvers


Do you want to solve a non-linear optimization problem? 📈📉 Have you doubts about which solver is the best? 🤔 Then, you are in 
the correct place! 🎉😁 Continue reading and have a look at this repo 👁🧐🤓

<img src="https://pyomo.readthedocs.io/en/latest/_images/PyomoNewBlue3.png" height="150" width="600"/>

## Goals ⚽

This repository aims to perform a comparison of several of the available non linear solvers. Such a comparison is made in 
terms of the objective value and the computational time.

To do this, some non linear optimization problems of different nature are modeled using [Pyomo](http://www.pyomo.org/) and run with different
solvers. The results are saved and analyzed below to get conclusions.

It is important to note that the conclusions drawn here cannot be extended to all the existing optimization problems 🌍🌎🌏
This repo just serves as a guide for your decisions ➡↖⬅ but it is not a universal truth. Please take this into account when 
deciding which solver is more suitable in your case.


## How is comparison testing performed? ✏📝📏

In this section, we explain the strategy applied to compare the different solvers. Let us consider the following list of
 non linear solvers used in our examples to compare their performance:

| Solver  | Open-source  | Run in local  | Run in NEOS  | 
|---|---|---|---|
|   |   |   |   |
|   |   |   |   |
|   |   |   |   |


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

The first list is formed by the solvers whose executable files are included in the [AMPL license](https://ampl.com/try-ampl/request-a-full-trial/#Form):

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

Finally, the third list is composed of those solvers that require a license for their use. In order to avoid such an issue, 
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

 
The next step consists of running the optimization problem for the different solvers enumerated in the previous 3️⃣ lists.
Since the optimization problems we are considering in this repo are (highly) non linear, they may be stuck at local optima ➡⬅
Hence, a multistart approach is applied in which the same problem is run several times starting from different initial solutions,
randomly chosen from the feasible region. The number of runs at the multistart is defined by the user.

The objective value and the computational time will be used as measures of performance. Particularly, we provide the average,
the maximum and minimum of all the objective values obtained after running the multistart, as well as the average, the 
maximum and minimum value of the computational time. The results of each optimization problem are summarized in a table which,
apart of the performance values previously mentioned, includes extra information such as the name of the problem, if the Neos Server 
or AMPL is used or not, the solver utilized, the number of variables and constraints involved, and also 
the sense of the optimization problem, i.e., if the objective is to minimize or maximize certain objective function. An 
example of the results is given below:

| problem  | neos | ampl | solver | #variables | #constraints | sense | mean obj. val. | max obj. val.| min obj. val. | mean comp. time | max comp. time | min comp. time |
| -------- | ---- | ---- | ------ | ---------- | ------------ | ----- | -------------- | ------------ | ------------- | --------------- | -------------- | -------------- |
| Problem 1| yes | no   | conopt |   100      |     200      | min   |  1.2           |     5.7      | 0.23          |    180          | 300            | 50             |         


The whole computational experience is executed on a laptop with 8Gb of RAM memory at 1.80 GHz, running Windows 10.

## Examples 📌📋

This section briefly explains the organization of the examples used here. This repo contains one folder per optimization problem.
Each folder contains three new folders and a script called `main_name_of_the_problem.py`. Such a script is the only one that should be executed
when running the experiments. Moreover, the first folder, entitled `model_pdf` contains the document with the model formulation. The second folder, `optimization_problem_utils` is formed
by the scripts which solve the optimization problem, so *be careful when modifying it*. Do not hesitate to contact the person who has written this code if you want to make changes 📳
Finally, the third folder entitled `results_name_of_the_problem` includes the results obtained in all the runs of the multistart, and a
summary of them, called `summary_results.csv`.

The following table provides the optimization problems which have been used for the non linear solvers' comparison. This table 
includes the name of the problem, the problem formulation and the folder with the obtained results.

| # | Name  | Model Formulation | Folder results |
| - | ----- | ------------------| -------------- |
| 1 | M2SVM_Optimal_Weights | [M2SVM_Optimal_Weights Model Formulation](./M2SVM_Optimal_Weights/model_pdf/M2SVM_Optimal_Weights.pdf)  |[M2SVM_Optimal_Weights Results](./M2SVM_Optimal_Weights/results_m2svm_optimal_weights) |
| 2 | MINLP_Trigonometric_Functions | [MINLP_Trigonometric_Functions Model Formulation](./MINLP_Trigonometric_Functions/model_pdf/MINLP_Trigonometric_Functions.pdf)  |[MINLP_Trigonometric_Functions Results](./MINLP_Trigonometric_Functions/results_MINLP_Trignometric_Functions) |

More details about the results obtained in each example are given in the next sections.

### Example 1️⃣: M2SVM Optimal Weights

The aim of the optimization problem formulated [here](./M2SVM_Optimal_Weights/model_pdf/M2SVM_Optimal_Weights.pdf) is to
 seek the optimal weights in the Gaussian kernel in order to obtain a good classification with the Support Vector Machine. Since the problem is based on
  a toy example, the size of this optimization problem is small. Particularly, it is formed by **15 continuous variables**
  and **36 constraints**. 
  
  A multistart approach with **1000 runs** is run for the different solvers as explained in the previous section. A summary of the
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
  
  Concerning the computational time, we see that there is a difference of three orders of magnitude between using the solvers through the Neos server
  or directly using the executable files in a standard laptop. The reason for this issue is that solving an optimization problem via Neos not 
  only includes the computational cost associating to the problem resolution but also the calls to the server which significantly increase the total elapsed time.
  Looking at those solvers that have been used without Neos, we check that `couenne` is the one that spent most time solving the problem, and then is not advisable to use.
  
  Thus, taking into account the previous comments, for this particular optimization problem, we suggest using any of the following solvers without Neos: `conopt`,
  `minos`, `snopt`, `ipopt` and `bonmin`.
  
  ### Example 2️⃣: MINLP Trigonometric Functions
  
  In this example, we solve a Mixed Integer Non Linear Programming (MINLP) problem with trigonometric functions involved.
  Contrary to what happened in Example 1️⃣, this problem is of medium size, since it is formed by **300 variables (150 integer and 150 continuous)**
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
MINLP_trigonometric_functions|no|no|couenne|300|153|min|-|-|-|max time reached|-|-
MINLP_trigonometric_functions|yes|no|conopt|300|153|min|24.43|87.75|-200.60|19.14|25.74|9.51
MINLP_trigonometric_functions|yes|no|ipopt|300|153|min|6.24|64.99|-9.12|22.11|57.23|9.91
MINLP_trigonometric_functions|yes|no|filter|300|153|min|27.43|62.73|-9.65|19.68|38.31|15.17
MINLP_trigonometric_functions|yes|no|knitro|300|153|min|-4.54|62.66|-9.26|19.54|43.99|7.20
MINLP_trigonometric_functions|yes|no|loqo|300|153|min|-7.16|-0.23|-9.50|19.77|39.89|9.05
MINLP_trigonometric_functions|yes|no|minos|300|153|min|39.47|463.60|-417.00|18.92|28.98|8.31
MINLP_trigonometric_functions|yes|no|mosek|300|153|min|0.00|0.00|0.00|18.51|23.19|4.77
MINLP_trigonometric_functions|yes|no|snopt|300|153|min|31.40|299.30|-22.19|19.49|37.79|5.63
MINLP_trigonometric_functions|yes|no|bonmin|300|153|min|7.13|64.26|-9.72|22.33|48.24|6.62
MINLP_trigonometric_functions|yes|no|couenne|300|153|min|-|-|-|max time reached|-|-
MINLP_trigonometric_functions|yes|no|filmint|300|153|min|-|-|-|Neos error|-|-


It is important to note that not all the solvers tested are designed for handling optimization problems with integer variables. 
Particularly, only `bonmin`, `couenne`, `filmint` and `mosek` are able to treat such variables. However, looking at the 
table of results, we see that `couenne` and `filmint` cannot be used because of different reasons. On the other hand,
`mosek` only solves problems with a conic structure. Unfortunately, our problem does not have such a structure. Therefore, it just 
remains `bonmin` for the resolution. Similar results in terms of the objective values are obtained when comparing the average
obtained with and without Neos. Nevertheless, contrary to what happened in Example 1, the computational time decreases when solving the problem
through Neos server. In other words, when a medium-size non linear optimization problem (with integer variables)is considered, it is better to run the problem
in a server than in a standard laptop, since the computational time spent in the calls to the server is insignificant when comparing it with the problem resolution itself.

When the numerical experiments with the rest of the solvers are performed, the integrity of the variables is simply omitted. 
In other words, those solvers which do not handle problems with integer variables just omit this constraint. With respect to
the average of the objective values using these solvers, we observe that `knitro` is the best choice. Moreover, the results in terms of the computational time
 are acceptable (around 20 seconds). Note that we have ignored the results of `loqo` for the 
very same reason as in Example 1. Regarding the computational times, the best choice is to use `minos` through AMPL. However, the
mean of objective values has a very bad performance.

Hence, to solve the optimization problem of this example with integer variables we suggest to use `bonmin`, and if the integrality constraint
is omitted, then it is better to use `knitro`.


 ## Final Conclusions 🔚😋 

As previously mentioned at the beginning of this repo, the conclusions obtained here are not a universal truth, and therefore, they
have to be used just as a guide. However, we can state that:

* `loqo` is not a good solver due to the unstable results that it provides.
* `couenne` has more difficulties to find a local optimum solution than other solvers, e.g. `bonmin`.
* For small-size problems, all the solvers are equivalent in terms of the objective values. However, there exist a difference of
three orders of magnitude when running locally or via Neos.
* Running a medium-size optimization problem in a local computer is equivalent to run it on the Neos server, in terms of the computational time.
* For the optimization problems tested, the best solvers are `conopt`, `minos`, `snopt`, `ipopt` and `bonmin` in the first
case and `bonmin`, `knitro` and `ipopt` in the second case. 

## How to perform a solver comparison of a new optimization problem? 🤔

This section explains the steps to follow if a comparison of the solvers wants to be performed on a new optimization problem.
First, a folder with the name of the optimization problem should be created. This folder should contain three subfolders and 
a `main` file. The role of each of the folders is explained in the *Examples* section. Then, a script named `main_name_of_the_problem.py`
is built. This script contains the preamble in which the possible variables saved in the environment are deleted and which automatically changes the directory:
```
from __future__ import division

for name in dir():
    if not name.startswith('_'):
        del globals()[name]


import os

directory_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(directory_path, os.path.pardir))


import comparison_utils as cu

os.chdir(directory_path)

```

Then, the input parameters for the function which executes the comparison should be defined for the new problem. An example of the values of these parameters is given below:
```
problem = 'new_problem'
number_of_variables = 500
number_of_constraints = 200
sense_opt_problem = 'max'    
maximum_number_iterations_multistart = 1000
folder_results = 'results_' + problem + '/'
csv_file_name_multistart = 'results_multistart'
csv_file_summary_results = 'summary_results'
```

The next lines of the script are formed the lists of the solvers used for the comparison. They are divided according to their use
through AMPL and Neos. The lists below are the ones used in the previous examples. This choice is flexible and can be modified
according to the user requirements.
```
solvers_list_ampl = ['conopt',
                    'loqo',
                    'minos',
                    'snopt']
solvers_list_neos_flag_false = ['ipopt',
                                'bonmin',
                                'couenne']
solvers_list_neos_flag_true = ['conopt',
                               'ipopt',
                               'filter',
                               'knitro',
                               'loqo',
                               'minos',
                               'mosek',
                               'snopt',
                               'bonmin',
                               'couenne',
                               'filmint']
```

Finally, the optimization problem is solved for each element in the list using the function `run_optimization_problem_given_solver`.
Here you can see an example:
```
for solver in solvers_list_neos_flag_true:
    neos_flag = True
    ampl_flag = False
    print(solver)
    cu.run_optimization_problem_given_solver(solver = solver,
                                             problem = problem,
                                             neos_flag = neos_flag,
                                             number_of_variables = number_of_variables,
                                             number_of_constraints = number_of_constraints,
                                             sense_opt_problem = sense_opt_problem,
                                             maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                                             folder_results = folder_results,
                                             csv_file_name_multistart = csv_file_name_multistart,
                                             ampl_flag = ampl_flag)
```
Apart from the parameters previously mentioned, this function has two extra parameters called `neos_flag` and `ampl_flag`
indicating if the Neos server or the AMPL are used or not. The value of each parameter is given in next lines depending on 
the list used:

**solvers_list_ampl**
* `neos_flag = False`
* `ampl_flag = True`

**solvers_list_neos_flag_false**
* `neos_flag = False`
* `ampl_flag = False`

**solvers_list_neos_flag_true**
* `neos_flag = False`
* `ampl_flag = False`

The function `run_optimization_problem_given_solver` is defined in the script `comparison_utils.py`. When a new optimization
problem is to be compared for different solvers, then an `if` structure should be added, which is `True` if the `problem` parameter is equal
to the name of the new problem, and which runs the optimization problem. An example of the
new structure can be seen here:
```
if problem == "new_problem":
        run_new_problem(solver = solver,
                        problem = problem,
                        neos_flag = neos_flag,
                        number_of_variables = number_of_variables,
                        number_of_constraints = number_of_constraints,
                        sense_opt_problem = sense_opt_problem,
                        maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                        folder_results = folder_results,
                        csv_file_name_multistart = csv_file_name_multistart,
                        ampl_flag = ampl_flag) 
```

Moreover, the function `run_new_problem` includes the model definition as well as the resolution. We suggest including
in the call to the solver, an `if` statement which varies depending if the Neos server is used or not. The code structure should be similar to this one:

```
solver_name = 'conopt'
if neos_flag:
    solver = pe.SolverManagerFactory("neos")
else:
    solver = pe.SolverFactory(solver_name)

if neos_flag:
    results_solver = solver.solve(multistart_model,
                                  tee = True,
                                  opt = solver_name)
else:
    results_solver = solver.solve(multistart_model,
                                  tee = True)
```

In addition, we suggest saving the output results of the multistart in a binary `.pydata` file or similar, as well as to save 
the objective value and computational times (or any other performance measure) for each of the runs of the multistart. Moreover, it will be nice to
have a summary of the results. An example
of the function which writes the results can be seen in the function `write_results_minlp_trigonometric_functions` in the
 [comparison_utils.py](comparison_utils.py) file.
 
 Finally, it just remains to run the experiments and get conclusions about which solver is the best for the new optimization problem.
 
 
 ## Do you want to contribute? 🙋‍♂️🙋‍♀️
 
 Please, do it 😋 Any feedback is welcome 🤗 so feel free to ask or comment anything you want via a Pull Request in this repo.
 
 ## Contributors 🌬☀
 
 * [OASYS group](http://oasys.uma.es) -  groupoasys@gmail.com
 
 ## Developed by 👩‍💻👨‍💻
 * [Asunción Jiménez Cordero](https://www.researchgate.net/profile/Asuncion_Jimenez-Cordero/research) - asuncionjc@uma.es
 
 (Please add your name here if you have contributed to the repo)
 
 ## License 📝
 
    Copyright 2019 Optimization and Analytics for Sustainable energY Systems (OASYS)

    Licensed under the GNU General Public License, Version 3 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.gnu.org/licenses/gpl-3.0.en.html

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
 

