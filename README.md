# NLboxsolve.jl

Introduction
------------

Nlboxsolve.jl is a package containing a small collection of algorithms for soving systems of non-linear equations subject to box-constraints: ```F(x) = 0```, ``` l <= x <= u```, where it is assumed that the box-constraint admits a solution. This problem is similar, but different to mixed complementarity problems (for those see Complementarity.jl or NLsolve.jl).

So far the collection contains six algorithms: one based on Newton's method, four based on Levenberg-Marquardt, and one based on Powell's dogleg method.

Formulating a problem
---------------------

The key elements to a problem are a function containing the system of equations to be solved: ```F(x)```, an initial guess at the solution, ```x``` (1d-array), and the lower, ```l``` (1d-array), and upper, ```u``` (1d-array) bounds that form the box-constraint.  With these defined, we solve the system using:

```julia
soln = nlboxsolve(F,x,l,u)
```

Of course there are optional arguments.  The general function call is:

```julia
soln = nlboxsolve(F,x,l,u,xtol,ftol,maxiters,method)
```

where ```xtol``` is the convergence tolerance applied to the solution point (default = 1e-8), ```ftol``` is the convergence tolerance applied to ```F(x)``` (default = 1e-8), ```maxiters``` is the maximum number of iterations (default = 100), and ```method``` specifies the algorithm used (defualt = :lm_ar).

The solution algorithms
-----------------------

The solutions algorithms are the following:

- constrained Newton (method = :newton)
- constrained Levenberg-Marquardt (method = :lm)
- Kanzow, Yamashita, and Fukushima (2004) (method = :lm_kyf)
- Fan (2013) (method = :lm_fan)
- Amini and Rostami (2016) (method = :lm_ar) (this is the default method)
- Bellavia, Macconi, and Pieraccini (2012) (method = :dogleg)

Each algorithm returns the solution in a structure that has the following fields:

- solution_method
- initial
- zero
- fzero
- xdist
- fdist
- iters

which are (hopefully) self-explanatory, but to be explicit the value for ```x``` that satisfies the problem is given by ```soln.zero```.  The nature of the convergence (or non-convergence) can be determined from ```fzero```, ```xdist```, ```fdist```, and ```iters```.