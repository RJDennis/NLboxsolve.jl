module NLboxsolve

using ForwardDiff, LinearAlgebra, SparseArrays

include("structs.jl")
include("givens.jl")
include("arnoldi.jl")
include("gmres.jl")
include("jacvec.jl")
include("boxsolvers.jl")

export SolverState,
       SolverTrace,
       SolverResults

export constrained_newton,
       constrained_newton_ms,
       constrained_levenberg_marquardt,
       constrained_levenberg_marquardt_kyf,
       constrained_levenberg_marquardt_fan,
       constrained_levenberg_marquardt_ar,
       constrained_dogleg_solver,
       constrained_newton_krylov,
       constrained_newton_krylov_fs,
       constrained_jacobian_free_newton_krylov,
       constrained_newton_sparse,
       constrained_newton_ms_sparse,
       constrained_levenberg_marquardt_sparse,
       constrained_levenberg_marquardt_kyf_sparse,
       constrained_levenberg_marquardt_fan_sparse,
       constrained_levenberg_marquardt_ar_sparse,
       constrained_dogleg_solver_sparse,
       constrained_newton_krylov_sparse,
       constrained_newton_krylov_fs_sparse,
       nlboxsolve

end
