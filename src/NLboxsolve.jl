module NLboxsolve

using ForwardDiff, LinearAlgebra, SparseArrays

include("structs.jl")
include("givens.jl")
include("arnoldi.jl")
include("gmres.jl")
include("jacvec.jl")
include("boxsolvers.jl")
include("mcpsolvers.jl")

export SolverState,
       SolverTrace,
       BoxSolverResults,
       MCPSolverResults

export constrained_newton,
       constrained_levenberg_marquardt_kyf,
       constrained_levenberg_marquardt_ar,
       constrained_trust_region,
       constrained_dogleg_solver,
       constrained_newton_krylov,
       constrained_jacobian_free_newton_krylov,
       constrained_newton_sparse,
       constrained_levenberg_marquardt_kyf_sparse,
       constrained_levenberg_marquardt_ar_sparse,
       constrained_trust_region_sparse,
       constrained_dogleg_solver_sparse,
       constrained_newton_krylov_sparse,
       nlboxsolve,
       mcpsolve
       
    end
