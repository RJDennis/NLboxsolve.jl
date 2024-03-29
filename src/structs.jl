struct SolverState

    iteration::Integer
    xdist::Float64
    fdist::Float64

end

struct SolverTrace

    trace::Array{SolverState,1}

end

struct BoxSolverResults

    solution_method::Symbol
    initial::Array{Float64,1}
    zero::Array{Float64,1}
    fzero::Array{Float64,1}
    xdist::Float64
    fdist::Float64
    iters::Integer
    trace::SolverTrace

end

struct MCPSolverResults

    solution_method::Symbol
    reformulation::Symbol
    initial::Array{Float64,1}
    zero::Array{Float64,1}
    fzero::Array{Float64,1}
    xdist::Float64
    fdist::Float64
    iters::Integer
    trace::SolverTrace

end