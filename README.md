# NLboxsolve.jl

Introduction
------------

NLboxsolve.jl is a package containing a small collection of algorithms for solving systems of non-linear equations subject to box-constraints: ```F(x) = 0```, ``` lb <= x <= ub``` (element-by-element), where it is assumed that the box-constraint admits a solution. This problem is similar, but different, to mixed complementarity problems (for those see Complementarity.jl or NLsolve.jl).

So far the collection contains eight algorithms: one based on Newton's (or Newton-Raphson's) method, three based on Levenberg-Marquardt, two trust region methods, and two based on Newton-Krylov methods, one of which is Jacobian-free.  Experience suggests that some algorithms can work better than others for some problems, or some algorithms work on a problem where other fail, and no one algorithm dominates for all problems.

Installing
----------

NLboxsolve.jl is a registered package that can be installed using the package manager.  To do so, type the following in the REPL:

```julia
using Pkg
Pkg.add("NLboxsolve")
```

Formulating a problem
---------------------

The key elements to a problem are a vector-function containing the system of equations to be solved: ```F(x)```, an initial guess at the solution, ```x``` (1d-array), and the lower, ```lb``` (1d-array with default enteries equaling -Inf), and upper, ```ub``` (1d-array with default enteries equaling Inf) bounds that form the box-constraint.  With these objects defined, we solve the system using:

```julia
soln = nlboxsolve(F,x,lb,ub)
```

A Jacobian function can also be provided:

```julia
soln = nlboxsolve(F,J,x,lb,ub)
```

The function, ```F``` and the Jacobian function, ``` J``` can be in-place, meaning that they can take as their first argument a preallocated array.

Of course there are optional arguments.  The general function call allows up to six keyword arguments, for example:

```julia
soln = nlboxsolve(F,x,l,u,xtol=1e-10,ftol=1e-10,maxiters=200,method=:jfnk,sparsejac=:yes,krylovdim=20)
```

where ```xtol``` is the convergence tolerance applied to the solution point, ```x```, (default = 1e-8), ```ftol``` is the convergence tolerance applied to ```F(x)``` (default = 1e-8), ```maxiters``` is the maximum number of iterations (default = 100), ```method``` specifies the algorithm used (default = :lm_ar), ```sparsejac``` selects whether a sparse Jacobian should be used (default = :no), and ```krylovdim``` (default = 30) is specific to the three Newton-Krylov methods (and ignored for the other methods).

The solution algorithms
-----------------------

The solution algorithms are the following:

- Constrained Newton-Raphson (method = :nr)
- Constrained Levenberg-Marquardt (method = :lm)
- Kanzow, Yamashita, and Fukushima (2004) (method = :lm_kyf)
- Amini and Rostami (2016) (method = :lm_ar) (this is the default method)
- Trust region Dogleg method (method = :dogleg) (based on Nocedal and Wright, 2006)
- Bellavia, Macconi, and Pieraccini (2012) (method = :dogleg_bmp)
- Chen and Vuik (2016) (method = :nk)
- Jacobian-free Newton Krylov (method = :jfnk)

Each algorithm returns the solution in a structure that has the following fields:

- solution_method
- initial
- zero
- fzero
- xdist
- fdist
- iters
- trace

which are (hopefully) self-explanatory, but to be explicit the value for ```x``` that satisfies ```F(x) = 0``` is given by the ```zero``` field.  The nature of the convergence (or non-convergence) can be determined from ```fzero```, ```xdist```, ```fdist```, and ```iters```. The path taken by the solver is stored in the ```trace``` field.

Examples
--------

As a first example, consider the following 'fivediagonal' function:

```julia
function fivediagonal(x)

    f = similar(x)

    f[1]     = 4.0*(x[1] - x[2]^2) + x[2] - x[3]^2
    f[2]     = 8.0*x[2]*(x[2]^2 - x[1]) - 2.0*(1.0 - x[2]) + 4.0*(x[2] - x[3]^2) + x[3] - x[4]^2
    f[end-1] = 8.0*x[end-1]*(x[end-1]^2 - x[end-2]) - 2.0*(1.0 - x[end-1]) + 4.0*(x[end-1] - x[end]^2) 
             + x[end-2]^2 - x[end-3]
    f[end]   = 8.0*x[end]*(x[end]^2 - x[end-1]) - 2*(1.0 - x[end]) + x[end-1]^2 - x[end-2]    
    for i = 3:length(x)-2
        f[i] = 8.0*x[i]*(x[i]^2 - x[i-1]) - 2.0*(1.0 - x[i]) + 4.0*(x[i] - x[i+1]^2) + x[i-1]^2 
             - x[i-2] + x[i+1] - x[i+2]^2
    end

    return f

end

function fivediagonal!(f,x)

    f[1]     = 4.0*(x[1] - x[2]^2) + x[2] - x[3]^2
    f[2]     = 8.0*x[2]*(x[2]^2 - x[1]) - 2.0*(1.0 - x[2]) + 4.0*(x[2] - x[3]^2) + x[3] - x[4]^2
    f[end-1] = 8.0*x[end-1]*(x[end-1]^2 - x[end-2]) - 2.0*(1.0 - x[end-1]) + 4.0*(x[end-1] - x[end]^2) 
             + x[end-2]^2 - x[end-3]
    f[end]   = 8.0*x[end]*(x[end]^2 - x[end-1]) - 2*(1.0 - x[end]) + x[end-1]^2 - x[end-2]    
    for i = 3:length(x)-2
        f[i] = 8.0*x[i]*(x[i]^2 - x[i-1]) - 2.0*(1.0 - x[i]) + 4.0*(x[i] - x[i+1]^2) + x[i-1]^2 
            - x[i-2] + x[i+1] - x[i+2]^2
    end

end

n = 5000
x0 = [2.0 for _ in 1:n]
soln_a = nlboxsolve(fivediagonal,x0,xtol=1e-15,ftol=1e-15,krylovdim=80,method=:jfnk)
soln_b = nlboxsolve(fivediagonal!,x0,xtol=1e-15,ftol=1e-15,krylovdim=80,method=:jfnk)
```

Now consider the smaller problem:

```julia
function example(x)

    f = similar(x)

    f[1] = x[1]^2 + x[2]^2 - x[1]
    f[2] = x[1]^2 - x[2]^2 - x[2]

    return f

end

function example!(f,x)

    f[1] = x[1]^2 + x[2]^2 - x[1]
    f[2] = x[1]^2 - x[2]^2 - x[2]

end
```

To obtain one solution we can use:

```julia
x0 = [-0.6, 0.5]
lb = [-0.5, -0.2]
ub = [0.5, 0.4]
soln_c = nlboxsolve(example,x0,lb,ub,ftol=1e-15,xtol=1e-15,method=:lm)
```

To obtain a second solution:

```julia
x0 = [0.8, 0.6]
lb = [0.5, 0.0]
ub = [1.0, 1.0]
soln_d = nlboxsolve(example!,x0,lb,ub,ftol=1e-15,xtol=1e-15,method=:lm)
```

As a final example---one involving the use of a user defined Jacobian---, consider the problem borrowed from the package NLsolve.jl:

```julia
function f(x)

    F = similar(x)

    F[1] = (x[1]+3)*(x[2]^3-7)+18
    F[2] = sin(x[2]*exp(x[1])-1)

    return F

end

function j(x)

    J = zeros(Number,2,2)

    J[1, 1] = x[2]^3-7
    J[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    J[2, 1] = x[2]*u
    J[2, 2] = u

    return J

end

function f!(F, x)
    F[1] = (x[1]+3)*(x[2]^3-7)+18
    F[2] = sin(x[2]*exp(x[1])-1)
end

function j!(J, x)
    J[1, 1] = x[2]^3-7
    J[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    J[2, 1] = x[2]*u
    J[2, 2] = u
end

x0 = [0.1, 1.2]
lb = [0.0, 0.0]
ub = [5.0, 5.0]
soln_e = nlboxsolve(f,j,x0,lb,ub,xtol=1e-15,ftol=1e-15,method=:nr)
soln_f = nlboxsolve(f,j!,x0,lb,ub,xtol=1e-15,ftol=1e-15,method=:nr)
soln_g = nlboxsolve(f!,j,x0,lb,ub,xtol=1e-15,ftol=1e-15,method=:nr)
soln_h = nlboxsolve(f!,j!,x0,lb,ub,xtol=1e-15,ftol=1e-15,method=:nr)
```

Related packages
----------------

- NLsolve.jl
- Complementarity.jl
- NonlinearSolvers,jl
- NonlinearSolve.jl

References
----------

Amini, K., and F. Rostami, (2016), "Three-Steps Modified Levenberg-Marquardt Method with a New Line Search for Systems of Nonlinear Equations", *Journal of Computational and Applied Mathematics*, 300, pp. 30–42.

Bellavia, S., Macconi, M., and S. Pieraccini, (2012), "Constrained Dogleg Methods for Nonlinear Systems with Simple Bounds", *Computational Optimization and Applications*, 53, pp. 771–794.

Chen, J., and C. Vuik, (2016), "Globalization Technique for Projected Newton-Krylov Methods", *International Journal for Numerical Methods in Engineering*, 110, pp.661–674. 

Kanzow, C., Yamashita, N., and M. Fukushima, (2004), "Levenberg–Marquardt Methods with Strong Local Convergence Properties for Solving Nonlinear Equations with Convex Constraints", *Journal of Computational and Applied Mathematics*, 172, pp. 375–397.

Nocedal, J and S. Wright, (2006), Numerical Optimization, second edition, Springer.