function box_projection(x::Array{T,1},lb::Array{T,1},ub::Array{T,1}) where {T <: AbstractFloat}

    y = copy(x)

    for i in eachindex(x)
        if y[i] < lb[i]
            y[i] = lb[i]
        elseif y[i] > ub[i]
            y[i] = ub[i]
        end
    end

    return y

end

function box_projection!(x::Array{T,1},lb::Array{T,1},ub::Array{T,1}) where {T <: AbstractFloat}

    for i in eachindex(x)
        if x[i] < lb[i]
            x[i] = lb[i]
        elseif x[i] > ub[i]
            x[i] = ub[i]
        end
    end

end

function box_check(lb::Array{T,1},ub::Array{T,1}) where {T<:AbstractFloat}

    if length(lb) != length(ub)
        error(" 'lb' and 'ub' must have the same length")
    end

    satisfied = true
    for i in eachindex(lb)
        if ub[i] <= lb[i]
            satisfied = false
            break
        end
    end

    return satisfied

end

################ Solvers ###################

function constrained_newton(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_newton_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_newton_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_newton_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        xn .= xk - jk\f(xk)
  
        box_projection!(xn,lb,ub)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:nr,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    jk = Array{T,2}(undef,n,n)

    ffk = Array{T,1}(undef,n)

    lenx = zero(T)
    lenf = zero(T)
    
    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,ffk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        xn .= xk - jk\ffk
  
        box_projection!(xn,lb,ub)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffk)
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:nr,x,xn,ffk,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_newton_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_newton_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_newton_outplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    j_inplace = !applicable(j,x)

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    iter = 0
    while true

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        xn .= xk - jk\f(xk)
  
        box_projection!(xn,lb,ub)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:nr,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton_inplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    j_inplace = !applicable(j,x)

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    jk = Array{T,2}(undef,n,n)

    ffk = Array{T,1}(undef,n)

    lenx = zero(T)
    lenf = zero(T)
    
    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    iter = 0
    while true

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        f(ffk,xk)
        xn .= xk - jk\ffk
  
        box_projection!(xn,lb,ub)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffk)
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:nr,x,xn,ffk,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton_sparse(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_newton_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_newton_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_newton_sparse_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    jk = sparse(Array{T,2}(undef,n,n))

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end

        xn .= xk - jk\f(xk)
  
        box_projection!(xn,lb,ub)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:nr,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton_sparse_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    jk = sparse(Array{T,2}(undef,n,n))

    ffk = Array{T,1}(undef,n)

    lenx = zero(T)
    lenf = zero(T)
    
    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,ffk,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        xn .= xk - jk\ffk
  
        box_projection!(xn,lb,ub)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffk)
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:nr,x,xn,ffk,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton_ms(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_newton_ms_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_newton_ms_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_newton_ms_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        xn .= xk - jk\f(xk)
        xn .= xn - jk\f(xn)
        xn .= xn - jk\f(xn)
  
        box_projection!(xn,lb,ub)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:nr_ms,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton_ms_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    jk = Array{T,2}(undef,n,n)

    ffk = Array{T,1}(undef,n)

    lenx = zero(T)
    lenf = zero(T)
    
    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,ffk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        xn .= xk - jk\ffk
        f(ffk,xn)
        xn .= xn - jk\ffk
        f(ffk,xn)
        xn .= xn - jk\ffk

        box_projection!(xn,lb,ub)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffk)
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:nr_ms,x,xn,ffk,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton_ms(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_newton_ms_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_newton_ms_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_newton_ms_outplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    j_inplace = !applicable(j,x)

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    iter = 0
    while true

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        xn .= xk - jk\f(xk)
        xn .= xn - jk\f(xn)
        xn .= xn - jk\f(xn)
  
        box_projection!(xn,lb,ub)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:nr_ms,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton_ms_inplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    j_inplace = !applicable(j,x)

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    jk = Array{T,2}(undef,n,n)

    ffk = Array{T,1}(undef,n)

    lenx = zero(T)
    lenf = zero(T)
    
    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    iter = 0
    while true

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        f(ffk,xk)
        xn .= xk - jk\ffk
        f(ffk,xn)
        xn .= xn - jk\ffk
        f(ffk,xn)
        xn .= xn - jk\ffk

        box_projection!(xn,lb,ub)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffk)
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:nr_ms,x,xn,ffk,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton_ms_sparse(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_newton_ms_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_newton_ms_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_newton_ms_sparse_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    jk = sparse(Array{T,2}(undef,n,n))

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end

        xn .= xk - jk\f(xk)
        xn .= xn - jk\f(xn)
        xn .= xn - jk\f(xn)

        box_projection!(xn,lb,ub)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:nr_ms,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton_ms_sparse_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    jk = sparse(Array{T,2}(undef,n,n))

    ffk = Array{T,1}(undef,n)

    lenx = zero(T)
    lenf = zero(T)
    
    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,ffk,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        xn .= xk - jk\ffk
        f(ffk,xn)
        xn .= xn - jk\ffk
        f(ffk,xn)
        xn .= xn - jk\ffk

        box_projection!(xn,lb,ub)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffk)
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:nr_ms,x,xn,ffk,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_levenberg_marquardt(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_levenberg_marquardt_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_levenberg_marquardt_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_levenberg_marquardt_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    dk = similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    muk = 0.5*10^(-8)*norm(f(xk))^2

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + muk*I)\(jk'f(xk))
        xn .= xk + dk

        box_projection!(xn,lb,ub)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

        muk = min(muk,norm(f(xk))^2)

    end
  
    results = SolverResults(:lm,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    dk = similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)
    
    # Initialize solution trace
    f(ffk,x)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    muk = 0.5*10^(-8)*norm(ffk)^2

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,ffk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + muk*I)\(jk'ffk)
        xn .= xk + dk

        box_projection!(xn,lb,ub)

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

        muk = min(muk,norm(ffn)^2)

    end
  
    f(ffn,xn)
    results = SolverResults(:lm,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_levenberg_marquardt_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_levenberg_marquardt_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_levenberg_marquardt_outplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    j_inplace = !applicable(j,x)

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    dk = similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    muk = 0.5*10^(-8)*norm(f(xk))^2

    iter = 0
    while true

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        dk .= -(jk'jk + muk*I)\(jk'f(xk))
        xn .= xk + dk

        box_projection!(xn,lb,ub)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

        muk = min(muk,norm(f(xk))^2)

    end
  
    results = SolverResults(:lm,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_inplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    j_inplace = !applicable(j,x)

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    dk = similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)
    
    # Initialize solution trace
    f(ffk,x)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    muk = 0.5*10^(-8)*norm(ffk)^2

    iter = 0
    while true

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        f(ffk,xk)
        dk .= -(jk'jk + muk*I)\(jk'ffk)
        xn .= xk + dk

        box_projection!(xn,lb,ub)

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

        muk = min(muk,norm(ffn)^2)

    end
  
    f(ffn,xn)
    results = SolverResults(:lm,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_sparse(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_levenberg_marquardt_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_levenberg_marquardt_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_levenberg_marquardt_sparse_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    dk = similar(x)
    jk = sparse(Array{T,2}(undef,n,n))

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    muk = 0.5*10^(-8)*norm(f(xk))^2

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + muk*I)\(jk'f(xk))
        xn .= xk + dk

        box_projection!(xn,lb,ub)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

        muk = min(muk,norm(f(xk))^2)

    end
  
    results = SolverResults(:lm,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_sparse_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    dk = similar(x)
    jk = sparse(Array{T,2}(undef,n,n))

    lenx = zero(T)
    lenf = zero(T)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)
    
    # Initialize solution trace
    f(ffk,x)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    muk = 0.5*10^(-8)*norm(ffk)^2

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,ffk,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + muk*I)\(jk'ffk)
        xn .= xk + dk

        box_projection!(xn,lb,ub)

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

        muk = min(muk,norm(ffn)^2)

    end
  
    f(ffn,xn)
    results = SolverResults(:lm,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_kyf(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_levenberg_marquardt_kyf_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_levenberg_marquardt_kyf_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_levenberg_marquardt_kyf_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 3.12 from Kanzow, Yamashita, and Fukushima (2004) "Levenberg-Marquardt methods
    # with strong local convergence properties for solving nonlinear equations with convex constraints", Journal of 
    # Computational and Applied Mathematics, 172, pp. 375--397.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)
    dk = similar(x)
    g =  similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    beta  = 0.9
    gamma = 0.99995
    sigma = 10^(-4)
    rho   = 10^(-8)
    p     = 2.1

    muk = 0.5*10^(-8)*norm(f(xk))^2

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + muk*I)\(jk'f(xk))
        xn .= xk + dk
        box_projection!(xn,lb,ub)

        if norm(f(xn)) > gamma*norm(f(xk))
            g .= jk'f(xk)
            if g'dk <= -rho*norm(dk)^p
                alpha = 1.0
                while norm(f(xk+alpha*dk))^2 > norm(f(xk))^2 + 2*alpha*beta*g'dk
                    alpha = beta*alpha
                end
                xn .= xk + alpha*dk
            else
                alpha = 1.0
                while true
                    xt .= xk-alpha*g
                    box_projection!(xt,lb,ub)
                    if norm(f(xt))^2 <= norm(f(xk))^2 + 2*sigma*g'(xt-xk)
                        xn .=  xt
                        break
                    else
                        alpha = beta*alpha
                    end
                end
            end
        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

        muk = min(muk,norm(f(xk))^2)

    end
  
    results = SolverResults(:lm_kyf,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_kyf_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 3.12 from Kanzow, Yamashita, and Fukushima (2004) "Levenberg-Marquardt methods
    # with strong local convergence properties for solving nonlinear equations with convex constraints", Journal of 
    # Computational and Applied Mathematics, 172, pp. 375--397.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)
    dk = similar(x)
    g =  similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)
 
    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    beta  = 0.9
    gamma = 0.99995
    sigma = 10^(-4)
    rho   = 10^(-8)
    p     = 2.1

    muk = 0.5*10^(-8)*norm(ffk)^2

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,ffk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + muk*I)\(jk'ffk)
        xn .= xk + dk
        box_projection!(xn,lb,ub)

        f(ffn,xn)
        if norm(ffn) > gamma*norm(ffk)
            g .= jk'ffk
            if g'dk <= -rho*norm(dk)^p
                alpha = 1.0
                while true
                    f(ffn,xk+alpha*dk)
                    if norm(ffn)^2 > norm(ffk)^2 + 2*alpha*beta*g'dk
                        alpha = beta*alpha
                    else
                        break
                    end
                end
                xn .= xk + alpha*dk
            else
                alpha = 1.0
                while true
                    xt .= xk-alpha*g
                    box_projection!(xt,lb,ub)
                    f(ffn,xt)
                    if norm(ffn)^2 <= norm(ffk)^2 + 2*sigma*g'(xt-xk)
                        xn .=  xt
                        break
                    else
                        alpha = beta*alpha
                    end
                end
            end
        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

        muk = min(muk,norm(ffn)^2)

    end
  
    f(ffn,xn)
    results = SolverResults(:lm_kyf,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_kyf(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_levenberg_marquardt_kyf_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_levenberg_marquardt_kyf_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_levenberg_marquardt_kyf_outplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 3.12 from Kanzow, Yamashita, and Fukushima (2004) "Levenberg-Marquardt methods
    # with strong local convergence properties for solving nonlinear equations with convex constraints", Journal of 
    # Computational and Applied Mathematics, 172, pp. 375--397.

    j_inplace = !applicable(j,x)

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)
    dk = similar(x)
    g =  similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    beta  = 0.9
    gamma = 0.99995
    sigma = 10^(-4)
    rho   = 10^(-8)
    p     = 2.1

    muk = 0.5*10^(-8)*norm(f(xk))^2

    iter = 0
    while true

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        dk .= -(jk'jk + muk*I)\(jk'f(xk))
        xn .= xk + dk
        box_projection!(xn,lb,ub)

        if norm(f(xn)) > gamma*norm(f(xk))
            g .= jk'f(xk)
            if g'dk <= -rho*norm(dk)^p
                alpha = 1.0
                while norm(f(xk+alpha*dk))^2 > norm(f(xk))^2 + 2*alpha*beta*g'dk
                    alpha = beta*alpha
                end
                xn .= xk + alpha*dk
            else
                alpha = 1.0
                while true
                    xt .= xk-alpha*g
                    box_projection!(xt,lb,ub)
                    if norm(f(xt))^2 <= norm(f(xk))^2 + 2*sigma*g'(xt-xk)
                        xn .=  xt
                        break
                    else
                        alpha = beta*alpha
                    end
                end
            end
        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

        muk = min(muk,norm(f(xk))^2)

    end
  
    results = SolverResults(:lm_kyf,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_kyf_inplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 3.12 from Kanzow, Yamashita, and Fukushima (2004) "Levenberg-Marquardt methods
    # with strong local convergence properties for solving nonlinear equations with convex constraints", Journal of 
    # Computational and Applied Mathematics, 172, pp. 375--397.

    j_inplace = !applicable(j,x)

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)
    dk = similar(x)
    g =  similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)
 
    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    beta  = 0.9
    gamma = 0.99995
    sigma = 10^(-4)
    rho   = 10^(-8)
    p     = 2.1

    muk = 0.5*10^(-8)*norm(ffk)^2

    iter = 0
    while true

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        f(ffk,xk)
        dk .= -(jk'jk + muk*I)\(jk'ffk)
        xn .= xk + dk
        box_projection!(xn,lb,ub)

        f(ffn,xn)
        if norm(ffn) > gamma*norm(ffk)
            g .= jk'ffk
            if g'dk <= -rho*norm(dk)^p
                alpha = 1.0
                while true
                    f(ffn,xk+alpha*dk)
                    if norm(ffn)^2 > norm(ffk)^2 + 2*alpha*beta*g'dk
                        alpha = beta*alpha
                    else
                        break
                    end
                end
                xn .= xk + alpha*dk
            else
                alpha = 1.0
                while true
                    xt .= xk-alpha*g
                    box_projection!(xt,lb,ub)
                    f(ffn,xt)
                    if norm(ffn)^2 <= norm(ffk)^2 + 2*sigma*g'(xt-xk)
                        xn .=  xt
                        break
                    else
                        alpha = beta*alpha
                    end
                end
            end
        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

        muk = min(muk,norm(ffn)^2)

    end
  
    f(ffn,xn)
    results = SolverResults(:lm_kyf,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_kyf_sparse(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_levenberg_marquardt_kyf_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_levenberg_marquardt_kyf_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_levenberg_marquardt_kyf_sparse_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 3.12 from Kanzow, Yamashita, and Fukushima (2004) "Levenberg-Marquardt methods
    # with strong local convergence properties for solving nonlinear equations with convex constraints", Journal of 
    # Computational and Applied Mathematics, 172, pp. 375--397.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)
    dk = similar(x)
    g =  similar(x)
    jk = sparse(Array{T,2}(undef,n,n))

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    beta  = 0.9
    gamma = 0.99995
    sigma = 10^(-4)
    rho   = 10^(-8)
    p     = 2.1

    muk = 0.5*10^(-8)*norm(f(xk))^2

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + muk*I)\(jk'f(xk))
        xn .= xk + dk
        box_projection!(xn,lb,ub)

        if norm(f(xn)) > gamma*norm(f(xk))
            g .= jk'f(xk)
            if g'dk <= -rho*norm(dk)^p
                alpha = 1.0
                while norm(f(xk+alpha*dk))^2 > norm(f(xk))^2 + 2*alpha*beta*g'dk
                    alpha = beta*alpha
                end
                xn .= xk + alpha*dk
            else
                alpha = 1.0
                while true
                    xt .= xk-alpha*g
                    box_projection!(xt,lb,ub)
                    if norm(f(xt))^2 <= norm(f(xk))^2 + 2*sigma*g'(xt-xk)
                        xn .=  xt
                        break
                    else
                        alpha = beta*alpha
                    end
                end
            end
        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

        muk = min(muk,norm(f(xk))^2)

    end
  
    results = SolverResults(:lm_kyf,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_kyf_sparse_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 3.12 from Kanzow, Yamashita, and Fukushima (2004) "Levenberg-Marquardt methods
    # with strong local convergence properties for solving nonlinear equations with convex constraints", Journal of 
    # Computational and Applied Mathematics, 172, pp. 375--397.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)
    dk = similar(x)
    g =  similar(x)
    jk = sparse(Array{T,2}(undef,n,n))

    lenx = zero(T)
    lenf = zero(T)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    # Initialize solution trace
    f(ffk,x)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    beta  = 0.9
    gamma = 0.99995
    sigma = 10^(-4)
    rho   = 10^(-8)
    p     = 2.1

    muk = 0.5*10^(-8)*norm(ffk)^2

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,ffk,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + muk*I)\(jk'ffk)
        xn .= xk + dk
        box_projection!(xn,lb,ub)

        f(ffn,xn)
        if norm(ffn) > gamma*norm(ffk)
            g .= jk'ffk
            if g'dk <= -rho*norm(dk)^p
                alpha = 1.0
                while true
                    f(ffn,xk+alpha*dk)
                    if norm(ffn)^2 > norm(ffk)^2 + 2*alpha*beta*g'dk
                        alpha = beta*alpha
                    else
                        break
                    end
                end
                xn .= xk + alpha*dk
            else
                alpha = 1.0
                while true
                    xt .= xk-alpha*g
                    box_projection!(xt,lb,ub)
                    f(ffn,xt)
                    if norm(ffn)^2 <= norm(ffk)^2 + 2*sigma*g'(xt-xk)
                        xn .=  xt
                        break
                    else
                        alpha = beta*alpha
                    end
                end
            end
        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

        muk = min(muk,norm(ffn)^2)

    end
  
    f(ffn,xn)
    results = SolverResults(:lm_kyf,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_fan(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_levenberg_marquardt_fan_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_levenberg_marquardt_fan_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_levenberg_marquardt_fan_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 2.1 from Fan (2013) "On the Levenberg-Marquardt methods for convex constrained
    # nonlinear equations", Journal of Industrial and Management Optimization, 9, 1, pp. 227--241.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)
    dk = similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    delta = 1.0
    mu    = 10^(-4)
    gamma = 0.99995
    beta  = 0.9
    sigma = 10^(-4)

    iter = 0
    while true

        muk = mu*norm(f(xk))^delta

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + muk*I)\(jk'f(xk))
        xn .= xk+dk
        box_projection!(xn,lb,ub)

        if norm(f(xn)) > gamma*norm(f(xk))
            alpha = 1.0
            while true
                xt .= xk-alpha*jk'f(xk)
                box_projection!(xt,lb,ub)
                if norm(f(xt))^2 <= norm(f(xk))^2 + sigma*(f(xk)'jk*(xt-xk))
                    xn .= xt
                    break
                else
                    alpha = beta*alpha
                end
            end
        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)


        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:lm_fan,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_levenberg_marquardt_fan_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 2.1 from Fan (2013) "On the Levenberg-Marquardt methods for convex constrained
    # nonlinear equations", Journal of Industrial and Management Optimization, 9, 1, pp. 227--241.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)
    dk = similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    delta = 1.0
    mu    = 10^(-4)
    gamma = 0.99995
    beta  = 0.9
    sigma = 10^(-4)

    iter = 0
    while true

        f(ffk,xk)
        muk = mu*norm(ffk)^delta

        jk .= ForwardDiff.jacobian(f,ffk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + muk*I)\(jk'ffk)
        xn .= xk+dk
        box_projection!(xn,lb,ub)

        f(ffn,xn)
        if norm(ffn) > gamma*norm(ffk)
            alpha = 1.0
            while true
                xt .= xk-alpha*jk'ffk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if norm(ffn)^2 <= norm(ffk)^2 + sigma*(ffk'jk*(xt-xk))
                    xn .= xt
                    break
                else
                    alpha = beta*alpha
                end
            end
        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:lm_fan,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_levenberg_marquardt_fan(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_levenberg_marquardt_fan_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_levenberg_marquardt_fan_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_levenberg_marquardt_fan_outplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 2.1 from Fan (2013) "On the Levenberg-Marquardt methods for convex constrained
    # nonlinear equations", Journal of Industrial and Management Optimization, 9, 1, pp. 227--241.

    j_inplace = !applicable(j,x)

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)
    dk = similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    delta = 1.0
    mu    = 10^(-4)
    gamma = 0.99995
    beta  = 0.9
    sigma = 10^(-4)

    iter = 0
    while true

        muk = mu*norm(f(xk))^delta

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        dk .= -(jk'jk + muk*I)\(jk'f(xk))
        xn .= xk+dk
        box_projection!(xn,lb,ub)

        if norm(f(xn)) > gamma*norm(f(xk))
            alpha = 1.0
            while true
                xt .= xk-alpha*jk'f(xk)
                box_projection!(xt,lb,ub)
                if norm(f(xt))^2 <= norm(f(xk))^2 + sigma*(f(xk)'jk*(xt-xk))
                    xn .= xt
                    break
                else
                    alpha = beta*alpha
                end
            end
        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:lm_fan,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_levenberg_marquardt_fan_inplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 2.1 from Fan (2013) "On the Levenberg-Marquardt methods for convex constrained
    # nonlinear equations", Journal of Industrial and Management Optimization, 9, 1, pp. 227--241.

    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    j_inplace = !applicable(j,x)
    
    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)
    dk = similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    delta = 1.0
    mu    = 10^(-4)
    gamma = 0.99995
    beta  = 0.9
    sigma = 10^(-4)

    iter = 0
    while true

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        f(ffk,xk)
        muk = mu*norm(ffk)^delta

        dk .= -(jk'jk + muk*I)\(jk'ffk)
        xn .= xk+dk
        box_projection!(xn,lb,ub)

        f(ffn,xn)
        if norm(ffn) > gamma*norm(ffk)
            alpha = 1.0
            while true
                xt .= xk-alpha*jk'ffk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if norm(ffn)^2 <= norm(ffk)^2 + sigma*(ffk'jk*(xt-xk))
                    xn .= xt
                    break
                else
                    alpha = beta*alpha
                end
            end
        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:lm_fan,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_levenberg_marquardt_fan_sparse(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_levenberg_marquardt_fan_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_levenberg_marquardt_fan_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_levenberg_marquardt_fan_sparse_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 2.1 from Fan (2013) "On the Levenberg-Marquardt methods for convex constrained
    # nonlinear equations", Journal of Industrial and Management Optimization, 9, 1, pp. 227--241.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)
    dk = similar(x)
    jk = sparse(Array{T,2}(undef,n,n))

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    delta = 1.0
    mu    = 10^(-4)
    gamma = 0.99995
    beta  = 0.9
    sigma = 10^(-4)

    iter = 0
    while true

        muk = mu*norm(f(xk))^delta

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + muk*I)\(jk'f(xk))
        xn .= xk+dk
        box_projection!(xn,lb,ub)

        if norm(f(xn)) > gamma*norm(f(xk))
            alpha = 1.0
            while true
                xt .= xk-alpha*jk'f(xk)
                box_projection!(xt,lb,ub)
                if norm(f(xt))^2 <= norm(f(xk))^2 + sigma*(f(xk)'jk*(xt-xk))
                    xn .= xt
                    break
                else
                    alpha = beta*alpha
                end
            end
        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:lm_fan,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_levenberg_marquardt_fan_sparse_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 2.1 from Fan (2013) "On the Levenberg-Marquardt methods for convex constrained
    # nonlinear equations", Journal of Industrial and Management Optimization, 9, 1, pp. 227--241.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)
    dk = similar(x)
    jk = sparse(Array{T,2}(undef,n,n))

    lenx = zero(T)
    lenf = zero(T)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    delta = 1.0
    mu    = 10^(-4)
    gamma = 0.99995
    beta  = 0.9
    sigma = 10^(-4)

    iter = 0
    while true

        f(ffk,xk)
        muk = mu*norm(ffk)^delta

        jk .= sparse(ForwardDiff.jacobian(f,ffk,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + muk*I)\(jk'ffk)
        xn .= xk+dk
        box_projection!(xn,lb,ub)

        f(ffn,xn)
        if norm(ffn) > gamma*norm(ffk)
            alpha = 1.0
            while true
                xt .= xk-alpha*jk'ffk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if norm(ffn)^2 <= norm(ffk)^2 + sigma*(ffk'jk*(xt-xk))
                    xn .= xt
                    break
                else
                    alpha = beta*alpha
                end
            end
        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:lm_fan,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_levenberg_marquardt_ar(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_levenberg_marquardt_ar_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_levenberg_marquardt_ar_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_levenberg_marquardt_ar_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 2.1 from Amini and Rostami (2016), "Three-steps modified Levenberg-Marquardt 
    # method with a new line search for systems of nonlinear equations", Journal of Computational and Applied Mathematics, 
    # 300, pp. 30--42.

    # Modified to allow for box-constraints by Richard Dennis.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    z  = similar(x)
    s  = similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    sigma1 = 0.005
    sigma2 = 0.005
    rho    = 0.8
    r      = 0.5
    gamma  = 10^(-16)
    mu     = 10^(-4)

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        lambdak = mu*norm(f(xk))^2

        d1k = -(jk'jk + lambdak*I)\(jk'f(xk))
        d2k = -(jk'jk + lambdak*I)\(jk'f(xk+d1k))
        d3k = -(jk'jk + lambdak*I)\(jk'f(xk+d1k+d2k))

        dk = d1k+d2k+d3k

        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk

        if norm(f(z)) <= rho*norm(f(xk))
            alpha = 1.0
        else
            if f(xk)'jk*dk > -gamma
                dk = d1k
                z .= xk+dk
                box_projection!(z,lb,ub)
                s .= z-xk
            end
            alpha = 1.0
            epsilon = 1/10
            while true
                if norm(f(xk+alpha*s))^2 > (1+epsilon)*norm(f(xk))^2 - sigma1*alpha^2*norm(s)^2 - sigma2*alpha^2*norm(f(xk))^2
                    alpha = r*alpha
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + alpha*s

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:lm_ar,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_levenberg_marquardt_ar_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 2.1 from Amini and Rostami (2016), "Three-steps modified Levenberg-Marquardt 
    # method with a new line search for systems of nonlinear equations", Journal of Computational and Applied Mathematics, 
    # 300, pp. 30--42.

    # Modified to allow for box-constraints by Richard Dennis.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    z  = similar(x)
    s  = similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    sigma1 = 0.005
    sigma2 = 0.005
    rho    = 0.8
    r      = 0.5
    gamma  = 10^(-16)
    mu     = 10^(-4)

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,ffk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        lambdak = mu*norm(ffk)^2

        d1k = -(jk'jk + lambdak*I)\(jk'ffk)
        f(ffn,xk+d1k)
        d2k = -(jk'jk + lambdak*I)\(jk'ffn)
        f(ffn,xk+d1k+d2k)
        d3k = -(jk'jk + lambdak*I)\(jk'ffn)

        dk = d1k+d2k+d3k

        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk

        f(ffn,z)
        if norm(ffn) <= rho*norm(ffk)
            alpha = 1.0
        else
            if ffk'jk*dk > -gamma
                dk = d1k
                z .= xk+dk
                box_projection!(z,lb,ub)
                s .= z-xk
            end
            alpha = 1.0
            epsilon = 1/10
            while true
                f(ffn,xk+alpha*s)
                if norm(ffn)^2 > (1+epsilon)*norm(ffk)^2 - sigma1*alpha^2*norm(s)^2 - sigma2*alpha^2*norm(ffk)^2
                    alpha = r*alpha
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + alpha*s

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:lm_ar,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_levenberg_marquardt_ar(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_levenberg_marquardt_ar_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_levenberg_marquardt_ar_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_levenberg_marquardt_ar_outplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 2.1 from Amini and Rostami (2016), "Three-steps modified Levenberg-Marquardt 
    # method with a new line search for systems of nonlinear equations", Journal of Computational and Applied Mathematics, 
    # 300, pp. 30--42.

    # Modified to allow for box-constraints by Richard Dennis.

    j_inplace = !applicable(j,x)

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    z  = similar(x)
    s  = similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    sigma1 = 0.005
    sigma2 = 0.005
    rho    = 0.8
    r      = 0.5
    gamma  = 10^(-16)
    mu     = 10^(-4)

    iter = 0
    while true

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        lambdak = mu*norm(f(xk))^2

        d1k = -(jk'jk + lambdak*I)\(jk'f(xk))
        d2k = -(jk'jk + lambdak*I)\(jk'f(xk+d1k))
        d3k = -(jk'jk + lambdak*I)\(jk'f(xk+d1k+d2k))

        dk = d1k+d2k+d3k

        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk

        if norm(f(z)) <= rho*norm(f(xk))
            alpha = 1.0
        else
            if f(xk)'jk*dk > -gamma
                dk = d1k
                z .= xk+dk
                box_projection!(z,lb,ub)
                s .= z-xk
            end
            alpha = 1.0
            epsilon = 1/10
            while true
                if norm(f(xk+alpha*s))^2 > (1+epsilon)*norm(f(xk))^2 - sigma1*alpha^2*norm(s)^2 - sigma2*alpha^2*norm(f(xk))^2
                    alpha = r*alpha
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + alpha*s

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:lm_ar,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_levenberg_marquardt_ar_inplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 2.1 from Amini and Rostami (2016), "Three-steps modified Levenberg-Marquardt 
    # method with a new line search for systems of nonlinear equations", Journal of Computational and Applied Mathematics, 
    # 300, pp. 30--42.

    # Modified to allow for box-constraints by Richard Dennis.

    j_inplace = !applicable(j,x)

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    z  = similar(x)
    s  = similar(x)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    sigma1 = 0.005
    sigma2 = 0.005
    rho    = 0.8
    r      = 0.5
    gamma  = 10^(-16)
    mu     = 10^(-4)

    iter = 0
    while true

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        f(ffk,xk)
        lambdak = mu*norm(ffk)^2

        d1k = -(jk'jk + lambdak*I)\(jk'ffk)
        f(ffn,xk+d1k)
        d2k = -(jk'jk + lambdak*I)\(jk'ffn)
        f(ffn,xk+d1k+d2k)
        d3k = -(jk'jk + lambdak*I)\(jk'ffn)

        dk = d1k+d2k+d3k

        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk

        f(ffn,z)
        if norm(ffn) <= rho*norm(ffk)
            alpha = 1.0
        else
            if ffk'jk*dk > -gamma
                dk = d1k
                z .= xk+dk
                box_projection!(z,lb,ub)
                s .= z-xk
            end
            alpha = 1.0
            epsilon = 1/10
            while true
                f(ffn,xk+alpha*s)
                if norm(ffn)^2 > (1+epsilon)*norm(ffk)^2 - sigma1*alpha^2*norm(s)^2 - sigma2*alpha^2*norm(ffk)^2
                    alpha = r*alpha
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + alpha*s

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:lm_ar,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_levenberg_marquardt_ar_sparse(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_levenberg_marquardt_ar_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_levenberg_marquardt_ar_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_levenberg_marquardt_ar_sparse_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 2.1 from Amini and Rostami (2016), "Three-steps modified Levenberg-Marquardt 
    # method with a new line search for systems of nonlinear equations", Journal of Computational and Applied Mathematics, 
    # 300, pp. 30--42.

    # Modified to allow for box-constraints by Richard Dennis.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    z  = similar(x)
    s  = similar(x)
    jk = sparse(Array{T,2}(undef,n,n))

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    sigma1 = 0.005
    sigma2 = 0.005
    rho    = 0.8
    r      = 0.5
    gamma  = 10^(-16)
    mu     = 10^(-4)

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        lambdak = mu*norm(f(xk))^2

        d1k = -(jk'jk + lambdak*I)\(jk'f(xk))
        d2k = -(jk'jk + lambdak*I)\(jk'f(xk+d1k))
        d3k = -(jk'jk + lambdak*I)\(jk'f(xk+d1k+d2k))

        dk = d1k+d2k+d3k

        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk

        if norm(f(z)) <= rho*norm(f(xk))
            alpha = 1.0
        else
            if f(xk)'jk*dk > -gamma
                dk = d1k
                z .= xk+dk
                box_projection!(z,lb,ub)
                s .= z-xk
            end
            alpha = 1.0
            epsilon = 1/10
            while true
                if norm(f(xk+alpha*s))^2 > (1+epsilon)*norm(f(xk))^2 - sigma1*alpha^2*norm(s)^2 - sigma2*alpha^2*norm(f(xk))^2
                    alpha = r*alpha
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + alpha*s

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:lm_ar,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_levenberg_marquardt_ar_sparse_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 2.1 from Amini and Rostami (2016), "Three-steps modified Levenberg-Marquardt 
    # method with a new line search for systems of nonlinear equations", Journal of Computational and Applied Mathematics, 
    # 300, pp. 30--42.

    # Modified to allow for box-constraints by Richard Dennis.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    z  = similar(x)
    s  = similar(x)
    jk = sparse(Array{T,2}(undef,n,n))

    lenx = zero(T)
    lenf = zero(T)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    sigma1 = 0.005
    sigma2 = 0.005
    rho    = 0.8
    r      = 0.5
    gamma  = 10^(-16)
    mu     = 10^(-4)

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,ffk,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        lambdak = mu*norm(ffk)^2

        d1k = -(jk'jk + lambdak*I)\(jk'ffk)
        f(ffn,xk+d1k)
        d2k = -(jk'jk + lambdak*I)\(jk'ffn)
        f(ffn,xk+d1k+d2k)
        d3k = -(jk'jk + lambdak*I)\(jk'ffn)

        dk = d1k+d2k+d3k

        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk

        f(ffn,z)
        if norm(ffn) <= rho*norm(ffk)
            alpha = 1.0
        else
            if ffk'jk*dk > -gamma
                dk = d1k
                z .= xk+dk
                box_projection!(z,lb,ub)
                s .= z-xk
            end
            alpha = 1.0
            epsilon = 1/10
            while true
                f(ffn,xk+alpha*s)
                if norm(ffn)^2 > (1+epsilon)*norm(ffk)^2 - sigma1*alpha^2*norm(s)^2 - sigma2*alpha^2*norm(ffk)^2
                    alpha = r*alpha
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + alpha*s

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:lm_ar,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
  
end

#=
function coleman_li(f::Function,j::Array{T,2},x::Array{T,1},lb::Array{T,1},ub::Array{T,1}) where {T <: AbstractFloat}

    n = length(x)
    D = zeros(n,n)

    df = j'*f(x)

    for i = 1:n
        if df[i] < 0.0 && ub[i] < Inf
            D[i,i] = ub[i] - x[i]
        elseif df[i] > 0.0 && lb[i] > -Inf
            D[i,i] = x[i] - lb[i]
        elseif df[i] == 0.0 && (lb[i] > -Inf || ub[i] < Inf)
            D[i,i] = min(x[i]-lb[i],ub[i]-x[i])
        else
            D[i,i] = 1.0
        end
    end

    return D

end
=#
function coleman_li_outplace(f::Function,j::AbstractArray{T,2},x::Array{T,1},lb::Array{T,1},ub::Array{T,1}) where {T <: AbstractFloat}

    df = j'f(x)

    for i in eachindex(df)
        if df[i] < 0.0 && ub[i] < Inf
            df[i] = ub[i] - x[i]
        elseif df[i] > 0.0 && lb[i] > -Inf
            df[i] = x[i] - lb[i]
        elseif df[i] == 0.0 && (lb[i] > -Inf || ub[i] < Inf)
            df[i] = min(x[i]-lb[i],ub[i]-x[i])
        else
            df[i] = 1.0
        end
    end
    return df
end

function coleman_li_inplace(f::Function,j::AbstractArray{T,2},x::Array{T,1},lb::Array{T,1},ub::Array{T,1}) where {T <: AbstractFloat}

    ff = Array{T,1}(undef,length(x))

    f(ff,x)
    df = j'ff

    for i in eachindex(df)
        if df[i] < 0.0 && ub[i] < Inf
            df[i] = ub[i] - x[i]
        elseif df[i] > 0.0 && lb[i] > -Inf
            df[i] = x[i] - lb[i]
        elseif df[i] == 0.0 && (lb[i] > -Inf || ub[i] < Inf)
            df[i] = min(x[i]-lb[i],ub[i]-x[i])
        else
            df[i] = 1.0
        end
    end
    return df
end

function step_selection_outplace(f::Function,x::Array{T,1},Gk::AbstractArray{T,2},jk::AbstractArray{T,2},gk::Array{T,1},pkn::Array{T,1},lb::Array{T,1},ub::Array{T,1},deltak::T,theta::T) where {T <: AbstractFloat}

    lambdak = Inf
    for i in eachindex(gk)
        if gk[i] != 0.0
            lambdak = min(max(((lb[i] - x[i])/gk[i]),((ub[i]-x[i])/gk[i])),lambdak)
        else
            lambdak = minimum((Inf,lambdak))
        end
    end

    taukprime = min(-(f(x)'*jk*gk)/norm(jk*gk)^2,deltak/norm(Gk*gk))
    tauk = taukprime
    if !isequal(x+taukprime*gk, box_projection(x+taukprime*gk,lb,ub))
        tauk = theta*lambdak
    end

    pc = tauk*gk

    pc_pkn = pc-pkn
    
    if norm(-pc_pkn) < 1e-13 # "division by zero errors can otherwise occur"
        return (pc+pkn)/2
    else
        gammahat = -(f(x)+jk*pc)'*jk*(-pc_pkn)/norm(jk*(-pc_pkn))^2
        if (pc'Gk^2*(pc_pkn))^2 - norm(Gk*(pc_pkn))^2*(norm(Gk*pc)^2-deltak^2) >= 0.0
            r = ((pc'Gk^2*(pc_pkn))^2 - norm(Gk*(pc_pkn))^2*(norm(Gk*pc)^2-deltak^2))^0.5
            gammaplus  = (pc'*Gk^2*(pc_pkn) + r)/norm(Gk*(pc_pkn))^2
            gammaminus = (pc'*Gk^2*(pc_pkn) - r)/norm(Gk*(pc_pkn))^2
        else
            gammaplus  = pc'*Gk^2*(pc_pkn)/norm(Gk*(pc_pkn))^2
            gammaminus = pc'*Gk^2*(pc_pkn)/norm(Gk*(pc_pkn))^2
        end

        if gammahat > 1.0
            gammatildaplus = Inf
            for i in eachindex(pkn)
                if (pkn[i]-pc[i]) != 0.0
                    gammatildaplus = min(max(((lb[i] - x[i]- pc[i])/(pkn[i]-pc[i])),((ub[i] - x[i] - pc[i])/(pkn[i]-pc[i]))),gammatildaplus)
                else
                    gammatildaplus = min(Inf,gammatildaplus)
                end
            end
            gamma = min(gammahat,gammaplus,theta*gammatildaplus)
        elseif gammahat < 0.0
            gammatildaminus = -Inf
            for i in eachindex(pkn)
                if (-pkn[i]+pc[i]) != 0.0
                    gammatildaminus = min(max(-((lb[i] - x[i]- pc[i])/(-(pkn[i]-pc[i]))),-((ub[i] - x[i] - pc[i])/(-(pkn[i]-pc[i])))),gammatildaminus)
                else
                    gammatildaminus = min(Inf,gammatildaminus)
                end
            end
            gamma = max(gammahat,gammaminus,theta*gammatildaminus)
        else
            gamma = gammahat
        end

        p = (1.0-gamma)*pc + gamma*pkn

        return p

    end

end

function step_selection_inplace(f::Function,x::Array{T,1},Gk::AbstractArray{T,2},jk::AbstractArray{T,2},gk::Array{T,1},pkn::Array{T,1},lb::Array{T,1},ub::Array{T,1},deltak::T,theta::T) where {T <: AbstractFloat}

    ff = Array{T,1}(undef,length(x))

    lambdak = Inf
    for i in eachindex(gk)
        if gk[i] != 0.0
            lambdak = min(max(((lb[i] - x[i])/gk[i]),((ub[i]-x[i])/gk[i])),lambdak)
        else
            lambdak = minimum((Inf,lambdak))
        end
    end

    f(ff,x)
    taukprime = min(-(ff'*jk*gk)/norm(jk*gk)^2,deltak/norm(Gk*gk))

    tauk = taukprime
    if !isequal(x+taukprime*gk, box_projection(x+taukprime*gk,lb,ub))
        tauk = theta*lambdak
    end

    pc = tauk*gk
    
    pc_pkn = pc-pkn
    
    if norm(-pc_pkn) < 1e-13 # "division by zero errors can otherwise occur"
        return (pc+pkn)/2
    else
        gammahat = -(ff+jk*pc)'*jk*(-pc_pkn)/norm(jk*(-pc_pkn))^2
        if (pc'Gk^2*(pc_pkn))^2 - norm(Gk*(pc_pkn))^2*(norm(Gk*pc)^2-deltak^2) >= 0.0
            r = ((pc'Gk^2*(pc_pkn))^2 - norm(Gk*(pc_pkn))^2*(norm(Gk*pc)^2-deltak^2))^0.5
            gammaplus  = (pc'*Gk^2*(pc_pkn) + r)/norm(Gk*(pc_pkn))^2
            gammaminus = (pc'*Gk^2*(pc_pkn) - r)/norm(Gk*(pc_pkn))^2
        else
            gammaplus  = pc'*Gk^2*(pc_pkn)/norm(Gk*(pc_pkn))^2
            gammaminus = pc'*Gk^2*(pc_pkn)/norm(Gk*(pc_pkn))^2
        end

        if gammahat > 1.0
            gammatildaplus = Inf
            for i in eachindex(pkn)
                if (pkn[i]-pc[i]) != 0.0
                    gammatildaplus = min(max(((lb[i] - x[i]- pc[i])/(pkn[i]-pc[i])),((ub[i] - x[i] - pc[i])/(pkn[i]-pc[i]))),gammatildaplus)
                else
                    gammatildaplus = min(Inf,gammatildaplus)
                end
            end
            gamma = min(gammahat,gammaplus,theta*gammatildaplus)
        elseif gammahat < 0.0
            gammatildaminus = -Inf
            for i in eachindex(pkn)
                if (-pkn[i]+pc[i]) != 0.0
                    gammatildaminus = min(max(-((lb[i] - x[i]- pc[i])/(-(pkn[i]-pc[i]))),-((ub[i] - x[i] - pc[i])/(-(pkn[i]-pc[i])))),gammatildaminus)
                else
                    gammatildaminus = min(Inf,gammatildaminus)
                end
            end
            gamma = max(gammahat,gammaminus,theta*gammatildaminus)
        else
            gamma = gammahat
        end

        p = (1.0-gamma)*pc + gamma*pkn

        return p

    end

end

function constrained_dogleg_solver(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_dogleg_solver_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_dogleg_solver_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_dogleg_solver_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of the algorithm from Bellavia, Macconi, and Pieraccini (2012), "Constrained 
    # dogleg methods for nonlinear systems with simple bounds", Computational Optimization and Applications, 
    # 53, pp. 771--794 

    n = length(x)
    xk  = copy(x)
    xn  = similar(x)
    pkn = similar(x)
    p   = similar(x)
    df  = similar(x)

    jk = Array{T,2}(undef,n,n)
    Gk = Array{T,2}(undef,n,n)
    gk = similar(x)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)
   
    # Initialize solver-parameters
    deltak = 1.0 
    theta  = 0.99995 
    beta1  = 0.5#0.25 
    beta2  = 0.9 

    # Replace infinities with largest possible Float64
    for i in eachindex(x)
        if lb[i] == -Inf
            lb[i] = -1/eps(T)
        elseif lb[i] == Inf
            lb[i] = 1/eps(T)
        end
        if ub[i] == -Inf
            ub[i] = -1/eps(T)
        elseif ub[i] == Inf
            ub[i] = 1/eps(T)
        end
    end

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        df .= coleman_li_outplace(f,jk,xk,lb,ub)
        Gk .= Diagonal(df.^(-1/2))
        gk .= -df.*jk'f(xk)
   
        alphak = max(theta,1.0-norm(f(xk)))
        pkn .= xk-jk\f(xk)
        box_projection!(pkn,lb,ub)
        pkn .= alphak*(pkn-xk)

        p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,deltak,theta)
    
        while true
            rhof = (norm(f(xk)) - norm(f(xk+p))) / (norm(f(xk)) - norm(f(xk) + jk'*p))
            if rhof < beta1 # linear approximation is poor fit so reduce the trust region
                deltak = min(0.25*deltak,0.5*norm(p))
                p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,deltak,theta)
            elseif rhof > beta2 # linear approximation is good fit so expand the trust region
                deltak = max(deltak,2*norm(p))
                p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,deltak,theta)
            end
            xn .= xk .+ p
            break
        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:dogleg,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_solver_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of the algorithm from Bellavia, Macconi, and Pieraccini (2012), "Constrained 
    # dogleg methods for nonlinear systems with simple bounds", Computational Optimization and Applications, 
    # 53, pp. 771--794 

    n = length(x)
    xk  = copy(x)
    xn  = similar(x)
    pkn = similar(x)
    p   = similar(x)
    df  = similar(x)

    jk = Array{T,2}(undef,n,n)
    Gk = Array{T,2}(undef,n,n)
    gk = similar(x)

    lenx = zero(T)
    lenf = zero(T)
    
    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    deltak = 1.0 
    theta  = 0.99995 
    beta1  = 0.5#0.25 
    beta2  = 0.9 

    # Replace infinities with largest possible Float64
    for i in eachindex(x)
        if lb[i] == -Inf
            lb[i] = -1/eps(T)
        elseif lb[i] == Inf
            lb[i] = 1/eps(T)
        end
        if ub[i] == -Inf
            ub[i] = -1/eps(T)
        elseif ub[i] == Inf
            ub[i] = 1/eps(T)
        end
    end

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,ffk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        df .= coleman_li_inplace(f,jk,xk,lb,ub)
        Gk .= Diagonal(df.^(-1/2))
        gk .= -df.*jk'ffk
   
        alphak = max(theta,1.0-norm(ffk))
        pkn .= xk-jk\ffk
        box_projection!(pkn,lb,ub)
        pkn .= alphak*(pkn-xk)

        p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,deltak,theta)
    
        while true
            f(ffn,xk+p)
            rhof = (norm(ffk) - norm(ffn)) / (norm(ffk) - norm(ffk + jk'p))
            if rhof < beta1 # linear approximation is poor fit so reduce the trust region
                deltak = min(0.25*deltak,0.5*norm(p))
                p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,deltak,theta)
            elseif rhof > beta2 # linear approximation is good fit so expand the trust region
                deltak = max(deltak,2*norm(p))
                p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,deltak,theta)
            end
            xn .= xk .+ p
            break
        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:dogleg,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_solver(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_dogleg_solver_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_dogleg_solver_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_dogleg_solver_outplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of the algorithm from Bellavia, Macconi, and Pieraccini (2012), "Constrained 
    # dogleg methods for nonlinear systems with simple bounds", Computational Optimization and Applications, 
    # 53, pp. 771--794 

    j_inplace = !applicable(j,x)

    n = length(x)
    xk  = copy(x)
    xn  = similar(x)
    pkn = similar(x)
    p   = similar(x)
    df  = similar(x)

    jk = Array{T,2}(undef,n,n)
    Gk = Array{T,2}(undef,n,n)
    gk = similar(x)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)
    
    # Initialize solver-parameters
    deltak = 1.0 
    theta  = 0.99995 
    beta1  = 0.5#0.25 
    beta2  = 0.9 

    # Replace infinities with largest possible Float64
    for i in eachindex(x)
        if lb[i] == -Inf
            lb[i] = -1/eps(T)
        elseif lb[i] == Inf
            lb[i] = 1/eps(T)
        end
        if ub[i] == -Inf
            ub[i] = -1/eps(T)
        elseif ub[i] == Inf
            ub[i] = 1/eps(T)
        end
    end

    iter = 0
    while true

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        df .= coleman_li_outplace(f,jk,xk,lb,ub)
        Gk .= Diagonal(df.^(-1/2))
        gk .= -df.*jk'f(xk)
   
        alphak = max(theta,1.0-norm(f(xk)))
        pkn .= xk-jk\f(xk)
        box_projection!(pkn,lb,ub)
        pkn .= alphak*(pkn-xk)

        p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,deltak,theta)
    
        while true
            rhof = (norm(f(xk)) - norm(f(xk+p))) / (norm(f(xk)) - norm(f(xk) + jk'*p))
            if rhof < beta1 # linear approximation is poor fit so reduce the trust region
                deltak = min(0.25*deltak,0.5*norm(p))
                p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,deltak,theta)
            elseif rhof > beta2 # linear approximation is good fit so expand the trust region
                deltak = max(deltak,2*norm(p))
                p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,deltak,theta)
            end
            xn .= xk .+ p
            break
        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:dogleg,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_solver_inplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of the algorithm from Bellavia, Macconi, and Pieraccini (2012), "Constrained 
    # dogleg methods for nonlinear systems with simple bounds", Computational Optimization and Applications, 
    # 53, pp. 771--794 

    j_inplace = !applicable(j,x)

    n = length(x)
    xk  = copy(x)
    xn  = similar(x)
    pkn = similar(x)
    p   = similar(x)
    df  = similar(x)

    jk = Array{T,2}(undef,n,n)
    Gk = Array{T,2}(undef,n,n)
    gk = similar(x)

    lenx = zero(T)
    lenf = zero(T)
    
    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    deltak = 1.0 
    theta  = 0.99995 
    beta1  = 0.5#0.25 
    beta2  = 0.9 

    # Replace infinities with largest possible Float64
    for i in eachindex(x)
        if lb[i] == -Inf
            lb[i] = -1/eps(T)
        elseif lb[i] == Inf
            lb[i] = 1/eps(T)
        end
        if ub[i] == -Inf
            ub[i] = -1/eps(T)
        elseif ub[i] == Inf
            ub[i] = 1/eps(T)
        end
    end

    iter = 0
    while true

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        f(ffk,xk)
        df .= coleman_li_inplace(f,jk,xk,lb,ub)
        Gk .= Diagonal(df.^(-1/2))
        gk .= -df.*jk'ffk
   
        alphak = max(theta,1.0-norm(ffk))
        pkn .= xk-jk\ffk
        box_projection!(pkn,lb,ub)
        pkn .= alphak*(pkn-xk)

        p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,deltak,theta)
    
        while true
            f(ffn,xk+p)
            rhof = (norm(ffk) - norm(ffn)) / (norm(ffk) - norm(ffk + jk'p))
            if rhof < beta1 # linear approximation is poor fit so reduce the trust region
                deltak = min(0.25*deltak,0.5*norm(p))
                p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,deltak,theta)
            elseif rhof > beta2 # linear approximation is good fit so expand the trust region
                deltak = max(deltak,2*norm(p))
                p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,deltak,theta)
            end
            xn .= xk .+ p
            break
        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:dogleg,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_solver_sparse(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_dogleg_solver_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_dogleg_solver_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_dogleg_solver_sparse_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of the algorithm from Bellavia, Macconi, and Pieraccini (2012), "Constrained 
    # dogleg methods for nonlinear systems with simple bounds", Computational Optimization and Applications, 
    # 53, pp. 771--794 

    n = length(x)
    xk  = copy(x)
    xn  = similar(x)
    pkn = similar(x)
    p   = similar(x)
    df  = similar(x)

    jk = sparse(Array{T,2}(undef,n,n))
    Gk = similar(jk)
    gk = similar(x)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)
    
    # Initialize solver-parameters
    deltak = 1.0 
    theta  = 0.99995 
    beta1  = 0.5#0.25 
    beta2  = 0.9 

    # Replace infinities with largest possible Float64  
    for i in eachindex(x)
        if lb[i] == -Inf
            lb[i] = -1/eps(T)
        elseif lb[i] == Inf
            lb[i] = 1/eps(T)
        end
        if ub[i] == -Inf
            ub[i] = -1/eps(T)
        elseif ub[i] == Inf
            ub[i] = 1/eps(T)
        end
    end

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        df .= coleman_li_outplace(f,jk,xk,lb,ub)
        Gk .= sparse(Diagonal(df.^(-1/2)))
        gk .= -df.*jk'f(xk)
   
        alphak = max(theta,1.0-norm(f(xk)))
        pkn .= xk-jk\f(xk)
        box_projection!(pkn,lb,ub)
        pkn .= alphak*(pkn-xk)

        p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,deltak,theta)
    
        while true
            rhof = (norm(f(xk)) - norm(f(xk+p))) / (norm(f(xk)) - norm(f(xk) + jk'*p))
            if rhof < beta1 # linear approximation is poor fit so reduce the trust region
                deltak = min(0.25*deltak,0.5*norm(p))
                p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,deltak,theta)
            elseif rhof > beta2 # linear approximation is good fit so expand the trust region
                deltak = max(deltak,2*norm(p))
                p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,deltak,theta)
            end
            xn .= xk .+ p
            break
        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:dogleg,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_solver_sparse_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of the algorithm from Bellavia, Macconi, and Pieraccini (2012), "Constrained 
    # dogleg methods for nonlinear systems with simple bounds", Computational Optimization and Applications, 
    # 53, pp. 771--794 

    n = length(x)
    xk  = copy(x)
    xn  = similar(x)
    pkn = similar(x)
    p   = similar(x)
    df  = similar(x)

    jk = sparse(Array{T,2}(undef,n,n))
    Gk = similar(jk)
    gk = similar(x)

    lenx = zero(T)
    lenf = zero(T)
    
    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    deltak = 1.0 
    theta  = 0.99995 
    beta1  = 0.5#0.25 
    beta2  = 0.9 

    # Replace infinities with largest possible Float64
    for i in eachindex(x)
        if lb[i] == -Inf
            lb[i] = -1/eps(T)
        elseif lb[i] == Inf
            lb[i] = 1/eps(T)
        end
        if ub[i] == -Inf
            ub[i] = -1/eps(T)
        elseif ub[i] == Inf
            ub[i] = 1/eps(T)
        end
    end

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,ffk,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        df .= coleman_li_inplace(f,jk,xk,lb,ub)
        Gk .= sparse(Diagonal(df.^(-1/2)))
        gk .= -df.*jk'ffk
   
        alphak = max(theta,1.0-norm(ffk))
        pkn .= xk-jk\ffk
        box_projection!(pkn,lb,ub)
        pkn .= alphak*(pkn-xk)

        p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,deltak,theta)
    
        while true
            f(ffn,xk+p)
            rhof = (norm(ffk) - norm(ffn)) / (norm(ffk) - norm(ffk + jk'p))
            if rhof < beta1 # linear approximation is poor fit so reduce the trust region
                deltak = min(0.25*deltak,0.5*norm(p))
                p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,deltak,theta)
            elseif rhof > beta2 # linear approximation is good fit so expand the trust region
                deltak = max(deltak,2*norm(p))
                p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,deltak,theta)
            end
            xn .= xk .+ p
            break
        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:dogleg,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_newton_krylov_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
    else
        return constrained_newton_krylov_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
    end

end

function constrained_newton_krylov_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 1 from Chen and Vuik (2016) "Globalization Technique for 
    # Projected Newton-Krylov Methods", International Journal for Numerical Methods in Engineering, 
    # 110, pp. 661--674.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)

    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    etak = 1e-4
    beta = 0.9
    t = 1e-4
    sigma = 1e-4
    mmax = 50

    g(x) = 0.5*norm(f(x))^2

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false
            jk .= ForwardDiff.jacobian(f,xk)
            if !all(isfinite,jk)
                error("The jacobian has non-finite elements")
            end    
            dk, status = gmres(jk,-f(xk),xk,etak,krylovdim)

            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                if norm(f(xt)) <= (1.0 - t*alpha*(1-etak))*norm(f(xk))
                    xn .= xt
                    etak = (1.0 - alpha*(1.0 - etak))
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                    flag_ng = true
                end
            end
        else
            dk = -jk'f(xk)
            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                if g(xt) <= g(xk) + sigma*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                end
            end
        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:nk,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 1 from Chen and Vuik (2016) "Globalization Technique for 
    # Projected Newton-Krylov Methods", International Journal for Numerical Methods in Engineering, 
    # 110, pp. 661--674.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)

    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    etak = 1e-4
    beta = 0.9
    t = 1e-4
    sigma = 1e-4
    mmax = 50

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false
            jk .= ForwardDiff.jacobian(f,ffk,xk)
            if !all(isfinite,jk)
                error("The jacobian has non-finite elements")
            end    
            dk, status = gmres(jk,-ffk,xk,etak,krylovdim)

            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if norm(ffn) <= (1.0 - t*alpha*(1-etak))*norm(ffk)
                    xn .= xt
                    etak = (1.0 - alpha*(1.0 - etak))
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                    flag_ng = true
                end
            end
        else
            dk = -jk'ffk
            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if 0.5*norm(ffn)^2 <=  0.5*norm(ffk)^2 + sigma*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                end
            end
        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:nk,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_newton_krylov_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
    else
        return constrained_newton_krylov_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
    end

end

function constrained_newton_krylov_outplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 1 from Chen and Vuik (2016) "Globalization Technique for 
    # Projected Newton-Krylov Methods", International Journal for Numerical Methods in Engineering, 
    # 110, pp. 661--674.

    j_inplace = !applicable(j,x)
    
    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)

    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    etak = 1e-4
    beta = 0.9
    t = 1e-4
    sigma = 1e-4
    mmax = 50

    g(x) = 0.5*norm(f(x))^2

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false
            if j_inplace == false
                jk .= j(xk)
            else
                j(jk,xk)
            end
            if !all(isfinite,jk)
                error("The jacobian has non-finite elements")
            end
    
            dk, status = gmres(jk,-f(xk),xk,etak,krylovdim)

            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                if norm(f(xt)) <= (1.0 - t*alpha*(1-etak))*norm(f(xk))
                    xn .= xt
                    etak = (1.0 - alpha*(1.0 - etak))
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                    flag_ng = true
                end
            end
        else
            dk = -jk'f(xk)
            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                if g(xt) <= g(xk) + sigma*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                end
            end
        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:nk,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov_inplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 1 from Chen and Vuik (2016) "Globalization Technique for 
    # Projected Newton-Krylov Methods", International Journal for Numerical Methods in Engineering, 
    # 110, pp. 661--674.

    j_inplace = !applicable(j,x)

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)

    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    etak = 1e-4
    beta = 0.9
    t = 1e-4
    sigma = 1e-4
    mmax = 50

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false
            if j_inplace == false
                jk .= j(xk)
            else
                j(jk,xk)
            end
            if !all(isfinite,jk)
                error("The jacobian has non-finite elements")
            end
    
            f(ffk,xk)
            dk, status = gmres(jk,-ffk,xk,etak,krylovdim)

            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if norm(ffn) <= (1.0 - t*alpha*(1-etak))*norm(ffk)
                    xn .= xt
                    etak = (1.0 - alpha*(1.0 - etak))
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                    flag_ng = true
                end
            end
        else
            dk = -jk'ffk
            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if 0.5*norm(ffn)^2 <=  0.5*norm(ffk)^2 + sigma*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                end
            end
        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:nk,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov_sparse(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_newton_krylov_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
    else
        return constrained_newton_krylov_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
    end

end

function constrained_newton_krylov_sparse_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 1 from Chen and Vuik (2016) "Globalization Technique for 
    # Projected Newton-Krylov Methods", International Journal  for Numerical Methods in Engineering, 
    # 110, pp. 661--674.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)

    jk = sparse(Array{T,2}(undef,n,n))

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    etak = 1e-4
    beta = 0.9
    t = 1e-4
    sigma = 1e-4
    mmax = 50

    g(x) = 0.5*norm(f(x))^2

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false
            jk .= sparse(ForwardDiff.jacobian(f,xk))
            if !all(isfinite,nonzeros(jk))
                error("The jacobian has non-finite elements")
            end    
            dk, status = gmres(jk,-f(xk),xk,etak,krylovdim)

            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                if norm(f(xt)) <= (1.0 - t*alpha*(1-etak))*norm(f(xk))
                    xn .= xt
                    etak = (1.0 - alpha*(1.0 - etak))
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                    flag_ng = true
                end
            end
        else
            dk = -jk'f(xk)
            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                if g(xt) <= g(xk) + sigma*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                end
            end
        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:nk,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov_sparse_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 1 from Chen and Vuik (2016) "Globalization Technique for 
    # Projected Newton-Krylov Methods", International Journal  for Numerical Methods in Engineering, 
    # 110, pp. 661--674.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)

    jk = sparse(Array{T,2}(undef,n,n))

    lenx = zero(T)
    lenf = zero(T)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    etak = 1e-4
    beta = 0.9
    t = 1e-4
    sigma = 1e-4
    mmax = 50

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false
            jk .= sparse(ForwardDiff.jacobian(f,ffk,xk))
            if !all(isfinite,nonzeros(jk))
                error("The jacobian has non-finite elements")
            end    
            dk, status = gmres(jk,-ffk,xk,etak,krylovdim)

            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if norm(ffn) <= (1.0 - t*alpha*(1-etak))*norm(ffk)
                    xn .= xt
                    etak = (1.0 - alpha*(1.0 - etak))
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                    flag_ng = true
                end
            end
        else
            dk = -jk'ffk
            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if 0.5*norm(ffn)^2 <=  0.5*norm(ffk)^2 + sigma*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                end
            end
        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:nk,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov_fs(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_newton_krylov_fs_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
    else
        return constrained_newton_krylov_fs_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
    end

end

function constrained_newton_krylov_fs_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of the approach in Frontini and Sormani (2004) "Third-order methods from
    # quadrature formulae for solving systems of nonlinear equations", Applied Mathematics and Computation, 
    # 149, pp. 771--782.

    # Modified to use Krylov methods, a globalization step, and to allow for box-constraints by Richard Dennis.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)

    lenx = zero(T)
    lenf = zero(T)

    jk = Array{T,2}(undef,n,n)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    etak = 1e-4
    beta = 0.9
    t = 1e-4
    sigma = 1e-4
    mmax = 50

    g(x) = 0.5*norm(f(x))^2

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false

            jk .= ForwardDiff.jacobian(f,xk)
            if !all(isfinite,jk)
                error("The jacobian has non-finite elements")
            end    
            kk, status = gmres(jk,-0.5*f(xk),xk,etak,krylovdim)
            jk .= ForwardDiff.jacobian(f,xk-kk) # Here xk-kk is not guaranteed to be inside the box
            if !all(isfinite,jk)
                error("The jacobian has non-finite elements")
            end
            dk, status = gmres(jk,-f(xk),xk,etak,krylovdim)

            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                if norm(f(xt)) <= (1.0 - t*alpha*(1-etak))*norm(f(xk))
                    xn .= xt
                    etak = (1.0 - alpha*(1.0 - etak))
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                    flag_ng = true
                end
            end
        else
            dk = -jk'f(xk)
            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                if g(xt) <= g(xk) + sigma*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                end
            end
        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:nk_fs,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov_fs_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of the approach in Frontini and Sormani (2004) "Third-order methods from
    # quadrature formulae for solving systems of nonlinear equations", Applied Mathematics and Computation, 
    # 149, pp. 771--782.

    # Modified to use Krylov methods, a globalization step, and to allow for box-constraints by Richard Dennis.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)

    lenx = zero(T)
    lenf = zero(T)

    jk = Array{T,2}(undef,n,n)
     
    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    etak = 1e-4
    beta = 0.9
    t = 1e-4
    sigma = 1e-4
    mmax = 50

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false

            jk .= ForwardDiff.jacobian(f,ffk,xk)
            if !all(isfinite,jk)
                error("The jacobian has non-finite elements")
            end    
            kk, status = gmres(jk,-0.5*ffk,xk,etak,krylovdim)
            jk .= ForwardDiff.jacobian(f,ffn,xk-kk) # Here xk-kk is not guaranteed to be inside the box
            if !all(isfinite,jk)
                error("The jacobian has non-finite elements")
            end    
            dk, status = gmres(jk,-ffk,xk,etak,krylovdim)

            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if norm(ffn) <= (1.0 - t*alpha*(1-etak))*norm(ffk)
                    xn .= xt
                    etak = (1.0 - alpha*(1.0 - etak))
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                    flag_ng = true
                end
            end
        else
            dk = -jk'ffk
            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if 0.5*norm(ffn)^2 <= 0.5*norm(ffk)^2 + sigma*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                end
            end
        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:nk_fs,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov_fs(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_newton_krylov_fs_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
    else
        return constrained_newton_krylov_fs_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
    end

end

function constrained_newton_krylov_fs_outplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of the approach in Frontini and Sormani (2004) "Third-order methods from
    # quadrature formulae for solving systems of nonlinear equations", Applied Mathematics and Computation, 
    # 149, pp. 771--782.

    # Modified to use Krylov methods, a globalization step, and to allow for box-constraints by Richard Dennis.

    j_inplace = !applicable(j,x)
    
    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)

    lenx = zero(T)
    lenf = zero(T)

    jk = Array{T,2}(undef,n,n)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    etak = 1e-4
    beta = 0.9
    t = 1e-4
    sigma = 1e-4
    mmax = 50

    g(x) = 0.5*norm(f(x))^2

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false

            if j_inplace == false
                jk .= j(xk)
            else
                j(jk,xk)
            end
            if !all(isfinite,jk)
                error("The jacobian has non-finite elements")
            end
    
            kk, status = gmres(jk,-0.5*f(xk),xk,etak,krylovdim)

            if j_inplace == false
                jk .= j(xk-kk)  # Here xk-kk is not guaranteed to be inside the box
            else
                j(jk,xk-kk)  # Here xk-kk is not guaranteed to be inside the box
            end
            if !all(isfinite,jk)
                error("The jacobian has non-finite elements")
            end
    
            dk, status = gmres(jk,-f(xk),xk,etak,krylovdim)

            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                if norm(f(xt)) <= (1.0 - t*alpha*(1-etak))*norm(f(xk))
                    xn .= xt
                    etak = (1.0 - alpha*(1.0 - etak))
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                    flag_ng = true
                end
            end
        else
            dk = -jk'f(xk)
            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                if g(xt) <= g(xk) + sigma*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                end
            end
        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:nk_fs,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov_fs_inplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of the approach in Frontini and Sormani (2004) "Third-order methods from
    # quadrature formulae for solving systems of nonlinear equations", Applied Mathematics and Computation, 
    # 149, pp. 771--782.

    # Modified to use Krylov methods, a globalization step, and to allow for box-constraints by Richard Dennis.

    j_inplace = !applicable(j,x)

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)

    lenx = zero(T)
    lenf = zero(T)

    jk = Array{T,2}(undef,n,n)
     
    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    etak = 1e-4
    beta = 0.9
    t = 1e-4
    sigma = 1e-4
    mmax = 50

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false

            if j_inplace == false
                jk .= j(xk)
            else
                j(jk,xk)
            end
            if !all(isfinite,jk)
                error("The jacobian has non-finite elements")
            end
    
            f(ffk,xk)
            kk, status = gmres(jk,-0.5*ffk,xk,etak,krylovdim)

            if j_inplace == false
                jk .= j(xk-kk) # Here xk-kk is not guaranteed to be inside the box
            else
                j(jk,xk-kk) # Here xk-kk is not guaranteed to be inside the box
            end
            if !all(isfinite,jk)
                error("The jacobian has non-finite elements")
            end
    
            f(ffn,xk-kk)
            dk, status = gmres(jk,-ffk,xk,etak,krylovdim)

            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if norm(ffn) <= (1.0 - t*alpha*(1-etak))*norm(ffk)
                    xn .= xt
                    etak = (1.0 - alpha*(1.0 - etak))
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                    flag_ng = true
                end
            end
        else
            dk = -jk'ffk
            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if 0.5*norm(ffn)^2 <= 0.5*norm(ffk)^2 + sigma*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                end
            end
        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:nk_fs,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov_fs_sparse(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_newton_krylov_fs_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
    else
        return constrained_newton_krylov_fs_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
    end

end

function constrained_newton_krylov_fs_sparse_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of the approach in Frontini and Sormani (2004) "Third-order methods from
    # quadrature formulae for solving systems of nonlinear equations", Applied Mathematics and Computation, 
    # 149, pp. 771--782.

    # Modified to use Krylov methods, a globalization step, and to allow for box-constraints by Richard Dennis.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)

    lenx = zero(T)
    lenf = zero(T)

    jk = sparse(Array{T,2}(undef,n,n))

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    etak = 1e-4
    beta = 0.9
    t = 1e-4
    sigma = 1e-4
    mmax = 50

    g(x) = 0.5*norm(f(x))^2

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false

            jk .= sparse(ForwardDiff.jacobian(f,xk))
            if !all(isfinite,nonzeros(jk))
                error("The jacobian has non-finite elements")
            end    
            kk, status = gmres(jk,-0.5*f(xk),xk,etak,krylovdim)
            jk .= sparse(ForwardDiff.jacobian(f,xk-kk)) # Here xk-kk is not guaranteed to be inside the box
            if !all(isfinite,nonzeros(jk))
                error("The jacobian has non-finite elements")
            end    
            dk, status = gmres(jk,-f(xk),xk,etak,krylovdim)

            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                if norm(f(xt)) <= (1.0 - t*alpha*(1-etak))*norm(f(xk))
                    xn .= xt
                    etak = (1.0 - alpha*(1.0 - etak))
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                    flag_ng = true
                end
            end
        else
            dk = -jk'f(xk)
            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                if g(xt) <= g(xk) + sigma*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                end
            end
        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:nk_fs,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov_fs_sparse_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of the approach in Frontini and Sormani (2004) "Third-order methods from
    # quadrature formulae for solving systems of nonlinear equations", Applied Mathematics and Computation, 
    # 149, pp. 771--782.

    # Modified to use Krylov methods, a globalization step, and to allow for box-constraints by Richard Dennis.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)

    lenx = zero(T)
    lenf = zero(T)

    jk = sparse(Array{T,2}(undef,n,n))
     
    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    etak = 1e-4
    beta = 0.9
    t = 1e-4
    sigma = 1e-4
    mmax = 50

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false

            jk .= sparse(ForwardDiff.jacobian(f,ffk,xk))
            if !all(isfinite,nonzeros(jk))
                error("The jacobian has non-finite elements")
            end    
            kk, status = gmres(jk,-0.5*ffk,xk,etak,krylovdim)
            jk .= sparse(ForwardDiff.jacobian(f,ffn,xk-kk)) # Here xk-kk is not guaranteed to be inside the box
            if !all(isfinite,nonzeros(jk))
                error("The jacobian has non-finite elements")
            end    
            dk, status = gmres(jk,-ffk,xk,etak,krylovdim)

            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if norm(ffn) <= (1.0 - t*alpha*(1-etak))*norm(ffk)
                    xn .= xt
                    etak = (1.0 - alpha*(1.0 - etak))
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                    flag_ng = true
                end
            end
        else
            dk = -jk'ffk
            alpha = 1.0
            m = 1
            while m <= mmax
                xt .= xk+alpha*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if 0.5*norm(ffn)^2 <= 0.5*norm(ffk)^2 + sigma*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                end
            end
        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:nk_fs,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_jacobian_free_newton_krylov(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    for i in eachindex(x)
        if x[i] < lb[i] || x[i] > ub[i]
            println("Warning: Initial point is outside the box.")
            box_projection!(x,lb,ub) # Put the initial guess inside the box
            break
        end
    end

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_jacobian_free_newton_krylov_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
    else
        return constrained_jacobian_free_newton_krylov_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
    end

end

function constrained_jacobian_free_newton_krylov_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This function is my implementation of a constrained Jacobian-free Newton-Krylov solver

    xk = copy(x)
    xn = similar(x)
    dk = similar(x)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    etak  = 1e-4
    beta  = 0.9
    t     = 1e-4
    sigma = 1e-4
    mmax  = 50

    g(x) = 0.5*norm(f(x))^2

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false
            dk, status = jacobian_free_gmres(f,xk,etak,krylovdim) # Using inexact Arnoldi (could use restarted)

            if status == false
                error("Restarted GMRES did not converge")
            end

            alpha = 1.0
            m = 1
            while m <= mmax
                if norm(f(box_projection(xk+alpha*dk,lb,ub))) <= (1.0 - t*alpha*(1-etak))*norm(f(xk))
                    xn = box_projection(xk+alpha*dk,lb,ub)
                    etak = (1.0 - alpha*(1.0 - etak))
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                    flag_ng = true
                end
            end
        else
            dk = - ForwardDiff.gradient(g,xk)
            alpha = 1.0
            m = 1
            while m <= mmax
                if g(box_projection(xk+alpha*dk,lb,ub)) <= g(xk) + sigma*dk'*(box_projection(xk+alpha*dk,lb,ub) .- xk)
                    xn = box_projection(xk+alpha*dk,lb,ub)
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                end
            end
        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:jfnk,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_jacobian_free_newton_krylov_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This function is my implementation of a constrained Jacobian-free Newton-Krylov solver

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    dk = similar(x)

    lenx = zero(T)
    lenf = zero(T)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    etak  = 1e-4
    beta  = 0.9
    t     = 1e-4
    sigma = 1e-4
    mmax  = 50

    g(x) = 0.5*norm(f(x))^2

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false
            dk, status = jacobian_free_gmres_inplace(f,xk,etak,krylovdim) # Using inexact Arnoldi (could use restarted)

            if status == false
                error("Restarted GMRES did not converge")
            end

            alpha = 1.0
            m = 1
            while m <= mmax
                f(ffk,xk)
                f(ffn,box_projection(xk+alpha*dk,lb,ub))
                if norm(ffn) <= (1.0 - t*alpha*(1-etak))*norm(ffk)
                    xn = box_projection(xk+alpha*dk,lb,ub)
                    etak = (1.0 - alpha*(1.0 - etak))
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                    flag_ng = true
                end
            end
        else
            dk = - ForwardDiff.gradient(g,ffk,xk)
            alpha = 1.0
            m = 1
            while m <= mmax
                f(ffn,box_projection(xk+alpha*dk,lb,ub))
                if 0.5*norm(ffn)^2 <= 0.5*norm(ffk) + sigma*dk'*(box_projection(xk+alpha*dk,lb,ub) .- xk)
                    xn = box_projection(xk+alpha*dk,lb,ub)
                    flag_ng = false
                    break
                else
                    alpha = beta*alpha
                    m += 1
                end
            end
        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:jfnk,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function nlboxsolve(f::Function,x::Array{T,1},lb::Array{T,1} = [-Inf for _ in eachindex(x)],ub::Array{T,1}= [Inf for _ in eachindex(x)];xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,method::Symbol=:lm_ar,krylovdim::S=30,sparsejac::Symbol=:no) where {T <: AbstractFloat, S <: Integer}

    if method == :nr
        if sparsejac == :no
            return constrained_newton(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
        elseif sparsejac == :yes
            return constrained_newton_sparse(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
        end
    elseif method == :nr_ms
        if sparsejac == :no
            return constrained_newton_ms(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
        elseif sparsejac == :yes
            return constrained_newton_ms_sparse(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
        end
    elseif method == :lm
        if sparsejac == :no
            return constrained_levenberg_marquardt(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
        elseif sparsejac == :yes
            return constrained_levenberg_marquardt_sparse(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
        end
    elseif method == :lm_kyf
        if sparsejac == :no
            return constrained_levenberg_marquardt_kyf(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
        elseif sparsejac == :yes
            return constrained_levenberg_marquardt_kyf_sparse(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
        end
    elseif method == :lm_fan
        if sparsejac == :no
            return constrained_levenberg_marquardt_fan(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
        elseif sparsejac == :yes
            return constrained_levenberg_marquardt_fan_sparse(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
        end
    elseif method == :lm_ar
        if sparsejac == :no
            return constrained_levenberg_marquardt_ar(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
        elseif sparsejav == :yes
            return constrained_levenberg_marquardt_ar_sparse(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
        end
    elseif method == :dogleg
        if sparsejac == :no
            return constrained_dogleg_solver(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
        elseif sparsejac == :yes
            return constrained_dogleg_solver_sparse(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
        end
    elseif method == :nk
        if sparsejac == :no
            return constrained_newton_krylov(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
        elseif sparsejac == :yes
            return constrained_newton_krylov_sparse(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
        end
    elseif method == :nk_fs
        if sparsejac == :no
            return constrained_newton_krylov_fs(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
        elseif sparsejac == :yes
            return constrained_newton_krylov_fs_sparse(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
        end
    elseif method == :jfnk
        return constrained_jacobian_free_newton_krylov(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        error("Your chosen solution method is not supported")
    end

end

function nlboxsolve(f::Function,j::Function,x::Array{T,1},lb::Array{T,1} = [-Inf for _ in eachindex(x)],ub::Array{T,1}= [Inf for _ in eachindex(x)];xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,method::Symbol=:lm_ar,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    if method == :nr
        return constrained_newton(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    elseif method == :nr_ms
        return constrained_newton_ms(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    elseif method == :lm
        return constrained_levenberg_marquardt(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    elseif method == :lm_kyf
        return constrained_levenberg_marquardt_kyf(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    elseif method == :lm_fan
        return constrained_levenberg_marquardt_fan(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    elseif method == :lm_ar
        return constrained_levenberg_marquardt_ar(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    elseif method == :dogleg
        return constrained_dogleg_solver(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    elseif method == :nk
        return constrained_newton_krylov(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
    elseif method == :nk_fs
        return constrained_newton_krylov_fs(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
    else
        error("Your chosen solution method is not supported")
    end

end