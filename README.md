# NLboxsolve.jl

Introduction
------------

NLboxsolve.jl is a package containing a small collection of algorithms for solving systems of non-linear equations subject to box-constraints: ```F(x) = 0```, ``` l <= x <= u```, where it is assumed that the box-constraint admits a solution. This problem is similar, but different to mixed complementarity problems (for those see Complementarity.jl or NLsolve.jl).

So far the collection contains nine algorithms: one based on Newton's method, four based on Levenberg-Marquardt, one based on Powell's dogleg method, and three based on Newton-Krylov methods, one of which is Jacobian-free.

Installing
----------

NLboxsolve.jl is a registered package that can be installed using the package manager.  To do so, type the following in the REPL:

```julia
using Pkg
Pkg.add("NLboxsolve")
```

Formulating a problem
---------------------

The key elements to a problem are a vector-function containing the system of equations to be solved: ```F(x)```, an initial guess at the solution, ```x``` (1d-array), and the lower, ```l``` (1d-array), and upper, ```u``` (1d-array) bounds that form the box-constraint.  With these defined, we solve the system using:

```julia
soln = nlboxsolve(F,x,l,u)
```

Of course there are optional arguments.  The general function call allows up to six keyword arguments, for example:

```julia
soln = nlboxsolve(F,x,l,u,xtol=1e-10,ftol=1e-10,maxiters=200,method=:jfnk,sparse=:yes,krylovdim=20)
```

where ```xtol``` is the convergence tolerance applied to the solution point, ```x```, (default = 1e-8), ```ftol``` is the convergence tolerance applied to ```F(x)``` (default = 1e-8), ```maxiters``` is the maximum number of iterations (default = 100), ```method``` specifies the algorithm used (default = :lm_ar), ```sparsejac``` selects whether a sparse Jacobian should be used (default = :no), and ```krylovdim``` (default = 30) is specific to the three Newton-Krylov methods (and ignored for the other methods).

The solution algorithms
-----------------------

The solution algorithms are the following:

- constrained Newton (method = :newton)
- constrained Levenberg-Marquardt (method = :lm)
- Kanzow, Yamashita, and Fukushima (2004) (method = :lm_kyf)
- Fan (2013) (method = :lm_fan)
- Amini and Rostami (2016) (method = :lm_ar) (this is the default method)
- Bellavia, Macconi, and Pieraccini (2012) (method = :dogleg)
- Chen and Vuik (2016) (method = :nk)
- a constrained globalized Newton-Krylov method based on Frontini and Sormani (2004) (method = :nk_fs)
- Jacobian-free Newton Krylov (method = :jfnk)

Each algorithm returns the solution in a structure that has the following fields:

- solution_method
- initial
- zero
- fzero
- xdist
- fdist
- iters

which are (hopefully) self-explanatory, but to be explicit the value for ```x``` that satisfies the problem is given by the ```zero``` field.  The nature of the convergence (or non-convergence) can be determined from ```fzero```, ```xdist```, ```fdist```, and ```iters```.

References
----------

Amini, K., and F. Rostami, (2016), "Three-Steps Modified Levenberg-Marquardt Method with a New Line Search for Systems of Nonlinear Equations", *Journal of Computational and Applied Mathematics*, 300, pp. 30–42.

Bellavia, S., Macconi, M., and S. Pieraccinin, (2012), "Constrained Dogleg Methods for Nonlinear Systems with Simple Bounds", *Computational Optimization and Applications*, 53, pp. 771–794.

Chen, J., and C. Vuik, (2016), "Globalization Technique for Projected Newton-Krylov Methods", *International Journal for Numerical Methods in Engineering*, 110, pp.661–674. 

Fan, J., (2013), "On the Levenberg-Marquardt Methods for Convex Constrained Nonlinear Equations", *Journal of Industrial and Management Optimization*, 9, 1, pp. 227–241.

Frontini, M., and E. Sormani, (2004), "Third-order Methods from Quadrature Formulae for Solving Systems of Nonlinear Equations", *Applied Mathematics and Computation*, 149, pp. 771–782.

Kanzow, C., Yamashita, N., and M. Fukushima, (2004), "Levenberg–Marquardt Methods with Strong Local Convergence Properties for Solving Nonlinear Equations with Convex Constraints", *Journal of Computational and Applied Mathematics*, 172, pp. 375–397.
