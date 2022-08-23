function mid(x,y,z)

    middle_value = MCPSolverResults(soln.solution_method,:mid,x,soln.zero,f(soln.zero),soln.xdist,soln.fdist,soln.iters,soln.trace)
    
    return results

end

function fischer_burmeister(x,l,u,y)

    fischer_burmeister(a,b) = sqrt(a^2+b^2) - a - b

    if l > -Inf && u == Inf
        return fischer_burmeister(x-l,y)
    elseif l == -Inf && u < Inf
        return -fischer_burmeister(u-x,-y)
    elseif l > -Inf && u < Inf
        return fischer_burmeister(x-l,fischer_burmeister(u-x,-y))
    else
        return x
    end

end

function MCP_mid(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1},xtol::T,ftol::T,maxiters::S,method::Symbol,sparsejac::Symbol) where {T <: AbstractFloat, S<:Integer}

    h(x) = x - mid.(lb,ub,x-f(x))
    soln = nlboxsolve(h,x,lb,ub,xtol=xtol,ftol=ftol,iterations=maxiters,method = method,sparsejac = sparsejac)

    results = MCPSolverResults(soln.solution_method,:mid,x,soln.zero,f(soln.zero),soln.xdist,soln.fdist,soln.iters,soln.trace)
    
    return results

end

function MCP_fischer_burmeister(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1},xtol::T,ftol::T,maxiters::S,method::Symbol,sparsejac::Symbol) where {T <: AbstractFloat, S<:Integer}

    h(x) = fischer_burmeister.(x,lb,ub,f(x))

    soln = nlboxsolve(h,x,lb,ub,xtol=xtol,ftol=ftol,iterations=maxiters,method = method,sparsejac = sparsejac)

    results = MCPSolverResults(soln.solution_method,:fb,x,soln.zero,f(soln.zero),soln.xdist,soln.fdist,soln.iters,soln.trace)
    
    return results

end

function mcpsolve(f::Function,x::Array{T,1},lb::Array{T,1} = [-Inf for _ in eachindex(x)],ub::Array{T,1}= [Inf for _ in eachindex(x)];xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,reformulation::Symbol=:fb,method::Symbol=:lm_ar,sparsejac::Symbol=:no) where {T <: AbstractFloat, S <: Integer}

    if reformulation == :mid
        return MCP_mid(f,x,lb,ub,xtol,ftol,iterations,method,sparsejac)
    elseif reformulation == :fb
        return MCP_fischer_burmeister(f,x,lb,ub,xtol,ftol,iterations,method,sparsejac)
    else
        error("Your chosen reformulation is not supported")
    end

end
