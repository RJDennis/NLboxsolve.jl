# NLboxsolve.jl

Introduction
============

NLboxsolve.jl is a package containing a small collection of algorithms for solving systems of non-linear equations subject to box-constraints: ```F(x) = 0```, ``` lb <= x <= ub``` (element-by-element), where it is assumed that the box-constraint admits a solution.  The package can also solve mixed complementarity problems, leveraging the non-linear box-solvers to do so.

The collection contains seven algorithms for solving box-constrained non-linear systems: one based on Newton's (or Newton-Raphson's) method, two based on Levenberg-Marquardt, two trust region methods, and two based on Newton-Krylov methods, one of which is Jacobian-free.

Installing
==========

NLboxsolve.jl is a registered package that can be installed using the package manager.  To do so, type the following in the REPL:

```julia
using Pkg
Pkg.add("NLboxsolve")
```

Solving box-constrained systems of equations
============================================

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
soln = nlboxsolve(F,x,l,u,xtol=1e-10,ftol=1e-10,iterations=200,method=:jfnk,sparsejac=:yes,krylovdim=20)
```

where ```xtol``` is the convergence tolerance applied to the solution point, ```x```, (default = 1e-8), ```ftol``` is the convergence tolerance applied to ```F(x)``` (default = 1e-8), ```iterations``` is the maximum number of iterations (default = 100), ```method``` specifies the algorithm used (default = :lm_ar), ```sparsejac``` selects whether a sparse Jacobian should be used (default = :no), and ```krylovdim``` (default = 30) is specific to the two Newton-Krylov methods (and ignored for the other methods).

The solution algorithms
-----------------------

The solution algorithms are the following:

- Constrained Newton-Raphson (method = :nr)
- Kanzow, Yamashita, and Fukushima (2004) (method = :lm_kyf)
- Amini and Rostami (2016) (method = :lm_ar) (this is the default method)
- Kimiaei (2017) (method = :tr) (this is a nonmonotone adaptive trust region method)
- Bellavia, Macconi, and Pieraccini (2012) (method = :dogleg) (Sometimes known as CoDoSol)
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

Solving Mixed Complementarity Problems
======================================

Mixed complementarity problems are problems that can be expressed as: $x_i \ge 0$, $f_i(x) \ge 0$ and $x_i f_i(x) = 0$, for all $i = 1,...n$, with $l \le x \le u$.  These problems can be reformulated in different ways allowing them to be solved using the tools used to solve box-constrained systems of nonlinear equations.  This package allows two reformulations:

- The "mid" reformulation recasts the problem as: $h_i(x) = x_i - mid(l_i,u_i,x_i-f_i(x))$ and seeks to solve $h(x) = 0$, $l \le x \le u$.  This reformulation is selected with ```reformulation = :mid```.
- The Fischer-Burmeister reformulation makes use of the transform: $h_i(x) = sqrt(x_i^2 + f_(x)^2) - x_i - f_i(x).  This reformulation is selected with ```reformulation = :fb``` (this reformulation is the default).

Formulating a problem
---------------------

The key elements to a problem are a vector-function containing the system of equations to be solved: ```F(x)```, an initial guess at the solution, ```x``` (1d-array), and the lower, ```lb``` (1d-array with default enteries equaling -Inf), and upper, ```ub``` (1d-array with default enteries equaling Inf) bounds that form the box-constraint.  With these objects defined, we solve the system using:

```julia
soln = mcpsolve(F,x,lb,ub)
```

The solvers that underpin ```mcpsolve()``` are those accessable through the ```nlboxsolve()``` function.

The general function call allows up to seven keyword arguments, for example:

```julia
soln = mcpsolve(F,x,l,u,xtol=1e-10,ftol=1e-10,iterations=200,reformulation=:mid,method=:nr,sparsejac=:yes,krylovdim=20)
```

where ```xtol``` is the convergence tolerance applied to the solution point, ```x```, (default = 1e-8), ```ftol``` is the convergence tolerance applied to ```F(x)``` (default = 1e-8), ```iterations``` is the maximum number of iterations (default = 100), ```reformulation``` selects the transform used to reformulate the problem (default = :fb), ```method``` specifies the algorithm used (default = :lm_ar), ```sparsejac``` selects whether a sparse Jacobian should be used (default = :no), and ```krylovdim``` (default = 30) is specific to the Newton-Krylov methods (and ignored for the other methods).

Example
-------

Consider the following function:

```julia
function simple(x::Array{T,1}) where {T<:Number}

    f = Array{T,1}(undef,length(x))

    f[1] = x[1]^3 - 8
    f[2] = x[2] - x[3] + x[2]^3 + 3
    f[3] = x[2] + x[3] + 2*x[3]^3 - 3
    f[4] = x[4] + 2*x[4]^3

    return f

end

x0 = [0.5,0.5,0.5,0.5]
lb = [-1.0,-1.0,-1.0,-1.0]
ub = [1.0,1.0,1.0,1.0]
soln = mcpsolve(simple,x0,lb,ub,xtol = 1e-8,ftol=1e-8,reformulation=:mid,method=:nr)
```

Related packages
================

- NLsolve.jl
- Complementarity.jl
- NonlinearSolvers,jl
- NonlinearSolve.jl

References
==========

Amini, K., and F. Rostami, (2016), "Three-Steps Modified Levenberg-Marquardt Method with a New Line Search for Systems of Nonlinear Equations", *Journal of Computational and Applied Mathematics*, 300, pp. 30–42.

Bellavia, S., Macconi, M., and S. Pieraccini, (2012), "Constrained Dogleg Methods for Nonlinear Systems with Simple Bounds", *Computational Optimization and Applications*, 53, pp. 771–794.

Chen, J., and C. Vuik, (2016), "Globalization Technique for Projected Newton-Krylov Methods", *International Journal for Numerical Methods in Engineering*, 110, pp.661–674. 

Kanzow, C., Yamashita, N., and M. Fukushima, (2004), "Levenberg–Marquardt Methods with Strong Local Convergence Properties for Solving Nonlinear Equations with Convex Constraints", *Journal of Computational and Applied Mathematics*, 172, pp. 375–397.

Kimiaei, M., (2017), "A New-Class of Nonmonotone Adaptive Trust-Region Methods for Nonlinear Equations with Box Constraints", *Calcolo*, 54, pp. 769-812.