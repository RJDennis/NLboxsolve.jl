function mid(x,y,z)

    middle_value = sort(real.([x,y,z]))[2] # real is only needed here to support the :jfnk method

    return middle_value

end

function fischer_burmeister(x,l,u,y)

    fischer_burmeister(a,b) = sqrt(a^2+b^2 + eps()) - a - b # eps() added to avoid non-differentiability at (0,0)

    if l > -Inf && u == Inf
        return fischer_burmeister(x-l,y)
    elseif l == -Inf && u < Inf
        return -fischer_burmeister(u-x,-y)
    elseif l > -Inf && u < Inf
        return fischer_burmeister(x-l,fischer_burmeister(u-x,-y))
    else
        return -y
    end

end

function chks(x,l,u,y)

    chks(a,b) = sqrt((a-b)^2 + eps()) - a - b # eps() added to avoid non-differentiability at (0,0)

    if l > -Inf && u == Inf
        return chks(x-l,y)
    elseif l == -Inf && u < Inf
        return -chks(u-x,-y)
    elseif l > -Inf && u < Inf
        return chks(x-l,chks(u-x,-y))
    else
        return -y
    end

end

function closure_mid(g,x,lb,ub,f::Function)

    function h!(g,x)
    
        ff = similar(x)
        f(ff,x)
        g .= x - mid.(lb,ub,x-ff)
    
    end
    
    return h!

end

function closure_fischer_burmeister(g,x,lb,ub,f::Function)

    function h!(g,x)
    
        ff = similar(x)
        f(ff,x)
        g .= fischer_burmeister.(x,lb,ub,ff)
    
    end
    
    return h!

end

function closure_chks(g,x,lb,ub,f::Function)

    function h!(g,x)
    
        ff = similar(x)
        f(ff,x)
        g .= chks.(x,lb,ub,ff)
    
    end
    
    return h!

end

function MCP_mid(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1},xtol::T,ftol::T,maxiters::S,method::Symbol,sparsejac::Symbol) where {T <: AbstractFloat, S<:Integer}

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false

        h(x) = x - mid.(lb,ub,x-f(x))
        soln = nlboxsolve(h,x,lb,ub,xtol=xtol,ftol=ftol,iterations=maxiters,method = method,sparsejac = sparsejac)

        results = MCPSolverResults(soln.solution_method,:mid,x,soln.zero,f(soln.zero),soln.xdist,soln.fdist,soln.iters,soln.trace)
    
        return results

    else

        g = similar(x)
        h! = closure_mid(g,x,lb,ub,f)
        soln = nlboxsolve(h!,x,lb,ub,xtol=xtol,ftol=ftol,iterations=maxiters,method = method,sparsejac = sparsejac)

        ff = similar(x)
        f(ff,soln.zero)
        results = MCPSolverResults(soln.solution_method,:mid,x,soln.zero,ff,soln.xdist,soln.fdist,soln.iters,soln.trace)
    
        return results

    end

end

function MCP_fischer_burmeister(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1},xtol::T,ftol::T,maxiters::S,method::Symbol,sparsejac::Symbol) where {T <: AbstractFloat, S<:Integer}

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false

        h(x) = fischer_burmeister.(x,lb,ub,f(x))
        soln = nlboxsolve(h,x,lb,ub,xtol=xtol,ftol=ftol,iterations=maxiters,method=method,sparsejac=sparsejac)

        results = MCPSolverResults(soln.solution_method,:fb,x,soln.zero,f(soln.zero),soln.xdist,soln.fdist,soln.iters,soln.trace)
    
        return results

    else

        g = similar(x)
        h! = closure_fischer_burmeister(g,x,lb,ub,f)
        soln = nlboxsolve(h!,x,lb,ub,xtol=xtol,ftol=ftol,iterations=maxiters,method = method,sparsejac = sparsejac)

        ff = similar(x)
        f(ff,soln.zero)
        results = MCPSolverResults(soln.solution_method,:fb,x,soln.zero,ff,soln.xdist,soln.fdist,soln.iters,soln.trace)
    
        return results

    end

end

function MCP_chks(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1},xtol::T,ftol::T,maxiters::S,method::Symbol,sparsejac::Symbol) where {T <: AbstractFloat, S<:Integer}

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        
        h(x) = chks.(x,lb,ub,f(x))
        soln = nlboxsolve(h,x,lb,ub,xtol=xtol,ftol=ftol,iterations=maxiters,method=method,sparsejac=sparsejac)
        
        results = MCPSolverResults(soln.solution_method,:chks,x,soln.zero,f(soln.zero),soln.xdist,soln.fdist,soln.iters,soln.trace)
    
        return results

    else

        g = similar(x)
        h! = closure_chks(g,x,lb,ub,f)
        soln = nlboxsolve(h!,x,lb,ub,xtol=xtol,ftol=ftol,iterations=maxiters,method = method,sparsejac = sparsejac)

        ff = similar(x)
        f(ff,soln.zero)
        results = MCPSolverResults(soln.solution_method,:chks,x,soln.zero,ff,soln.xdist,soln.fdist,soln.iters,soln.trace)
    
        return results

    end

end

function mcpsolve(f::Function,x::Array{T,1},lb::Array{T,1} = [-Inf for _ in eachindex(x)],ub::Array{T,1}= [Inf for _ in eachindex(x)];xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,reformulation::Symbol=:mid,method::Symbol=:lm_ar,sparsejac::Symbol=:no) where {T <: AbstractFloat, S <: Integer}

    if reformulation == :mid
        return MCP_mid(f,x,lb,ub,xtol,ftol,iterations,method,sparsejac)
    elseif reformulation == :fb
        return MCP_fischer_burmeister(f,x,lb,ub,xtol,ftol,iterations,method,sparsejac)
    elseif reformulation == :chks
        return MCP_chks(f,x,lb,ub,xtol,ftol,iterations,method,sparsejac)
    else
        error("Your chosen reformulation is not supported")
    end

end