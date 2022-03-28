# lipo-b--annealing

Pile of code for IC layout

B*-Tree Annealer partially based on Prof. Yao-Wen Chang's implementation and materials

Includes Python interface supporting multi-start and parameter tuning via Lipschitz Optimization (LIPO)

Wirelength (HPWL)-driven Mixed Integer Linear Program (MILP) detailed placer supporting rotation
and relative positioning constraints defined by the O-Tree. By exploiting the relative position induced by
the O-Tree, we can get rid of variables necessary for satisfying pairwise overlap constraints

Parallelism handled with Python through joblib

Remember to set the root path variable in python scripts.
Multistart relies on a './tmp' directory which should initially contain all design
files (.nets, .block, etc) and will store the resulting .block and .rprt files
on completion.

The log dir \& plots.ipynb is for lipo iterations, not annealing iterations.

deps for Python:
- joblib (parallism)
- tqdm   (nice progress bars)
- dlib   (for lcb-lipo tuning framework)
- CVXPY & Coin-OR for detailed placement


### Build Annealer
>./clean

>cmake .

>make
