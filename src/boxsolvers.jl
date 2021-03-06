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

### Constrained Newton-Raphson

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
       
        f(ffk,xn)
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

    f(ffk,xn)
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

        f(ffk,xn)
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
  
    f(ffk,xn)
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

        f(ffk,xn)
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
  
    f(ffk,xn)
    results = SolverResults(:nr,x,xn,ffk,lenx,lenf,iter,solution_trace)

    return results
  
end

### Constrained Levenberg-Marquardt

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

    ??k = max(0.5*10^(-8)*norm(f(xk))^2,1e-4)

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + ??k*I)\(jk'f(xk))
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

        ??k = min(??k,norm(f(xk))^2)

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

    ??k = max(0.5*10^(-8)*norm(ffk)^2,1e-4)

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,ffk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + ??k*I)\(jk'ffk)
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

        ??k = min(??k,norm(ffn)^2)

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

    ??k = max(0.5*10^(-8)*norm(f(xk))^2,1e-4)

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

        dk .= -(jk'jk + ??k*I)\(jk'f(xk))
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

        ??k = min(??k,norm(f(xk))^2)

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

    ??k = max(0.5*10^(-8)*norm(ffk)^2,1e-4)

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
        dk .= -(jk'jk + ??k*I)\(jk'ffk)
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

        ??k = min(??k,norm(ffn)^2)

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

    ??k = max(0.5*10^(-8)*norm(f(xk))^2,1e-4)

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + ??k*I)\(jk'f(xk))
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

        ??k = min(??k,norm(f(xk))^2)

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

    ??k = max(0.5*10^(-8)*norm(ffk)^2,1e-4)

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,ffk,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + ??k*I)\(jk'ffk)
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

        ??k = min(??k,norm(ffn)^2)

    end
  
    f(ffn,xn)
    results = SolverResults(:lm,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
   
end

### Constrained Levenberg-Marquardt-kyf

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
    ?? = 0.9
    ?? = 0.99995
    ?? = 10^(-4)
    ?? = 10^(-8)
    p = 2.1

    ??k = max(0.5*10^(-8)*norm(f(xk))^2,1e-4)

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + ??k*I)\(jk'f(xk))
        xn .= xk + dk
        box_projection!(xn,lb,ub)

        if norm(f(xn)) > ??*norm(f(xk))
            g .= jk'f(xk)
            if g'dk <= -??*norm(dk)^p
                ?? = 1.0
                while norm(f(xk+??*dk))^2 > norm(f(xk))^2 + 2*??*??*g'dk
                    ?? = ??*??
                end
                xn .= xk + ??*dk
            else
                ?? = 1.0
                while true
                    xt .= xk-??*g
                    box_projection!(xt,lb,ub)
                    if norm(f(xt))^2 <= norm(f(xk))^2 + 2*??*g'*(xt-xk)
                        xn .=  xt
                        break
                    else
                        ?? = ??*??
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

        ??k = min(??k,norm(f(xk))^2)

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
    ?? = 0.9
    ?? = 0.99995
    ?? = 10^(-4)
    ?? = 10^(-8)
    p = 2.1

    ??k = max(0.5*10^(-8)*norm(ffk)^2,1e-4)

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,ffk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + ??k*I)\(jk'ffk)
        xn .= xk + dk
        box_projection!(xn,lb,ub)

        f(ffn,xn)
        if norm(ffn) > ??*norm(ffk)
            g .= jk'ffk
            if g'dk <= -??*norm(dk)^p
                ?? = 1.0
                while true
                    f(ffn,xk+??*dk)
                    if norm(ffn)^2 > norm(ffk)^2 + 2*??*??*g'dk
                        ?? = ??*??
                    else
                        break
                    end
                end
                xn .= xk + ??*dk
            else
                ?? = 1.0
                while true
                    xt .= xk-??*g
                    box_projection!(xt,lb,ub)
                    f(ffn,xt)
                    if norm(ffn)^2 <= norm(ffk)^2 + 2*??*g'*(xt-xk)
                        xn .=  xt
                        break
                    else
                        ?? = ??*??
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

        ??k = min(??k,norm(ffn)^2)

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
    ?? = 0.9
    ?? = 0.99995
    ?? = 10^(-4)
    ?? = 10^(-8)
    p = 2.1

    ??k = max(0.5*10^(-8)*norm(f(xk))^2,1e-4)

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

        dk .= -(jk'jk + ??k*I)\(jk'f(xk))
        xn .= xk + dk
        box_projection!(xn,lb,ub)

        if norm(f(xn)) > ??*norm(f(xk))
            g .= jk'f(xk)
            if g'dk <= -??*norm(dk)^p
                ?? = 1.0
                while norm(f(xk+??*dk))^2 > norm(f(xk))^2 + 2*??*??*g'dk
                    ?? = ??*??
                end
                xn .= xk + ??*dk
            else
                ?? = 1.0
                while true
                    xt .= xk-??*g
                    box_projection!(xt,lb,ub)
                    if norm(f(xt))^2 <= norm(f(xk))^2 + 2*??*g'*(xt-xk)
                        xn .=  xt
                        break
                    else
                        ?? = ??*??
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

        ??k = min(??k,norm(f(xk))^2)

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
    ?? = 0.9
    ?? = 0.99995
    ?? = 10^(-4)
    ?? = 10^(-8)
    p = 2.1

    ??k = max(0.5*10^(-8)*norm(ffk)^2,1e-4)

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
        dk .= -(jk'jk + ??k*I)\(jk'ffk)
        xn .= xk + dk
        box_projection!(xn,lb,ub)

        f(ffn,xn)
        if norm(ffn) > ??*norm(ffk)
            g .= jk'ffk
            if g'dk <= -??*norm(dk)^p
                ?? = 1.0
                while true
                    f(ffn,xk+??*dk)
                    if norm(ffn)^2 > norm(ffk)^2 + 2*??*??*g'dk
                        ?? = ??*??
                    else
                        break
                    end
                end
                xn .= xk + ??*dk
            else
                ?? = 1.0
                while true
                    xt .= xk-??*g
                    box_projection!(xt,lb,ub)
                    f(ffn,xt)
                    if norm(ffn)^2 <= norm(ffk)^2 + 2*??*g'*(xt-xk)
                        xn .=  xt
                        break
                    else
                        ?? = ??*??
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

        ??k = min(??k,norm(ffn)^2)

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
    ?? = 0.9
    ?? = 0.99995
    ?? = 10^(-4)
    ?? = 10^(-8)
    p = 2.1

    ??k = max(0.5*10^(-8)*norm(f(xk))^2,1e-4)

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + ??k*I)\(jk'f(xk))
        xn .= xk + dk
        box_projection!(xn,lb,ub)

        if norm(f(xn)) > ??*norm(f(xk))
            g .= jk'f(xk)
            if g'dk <= -??*norm(dk)^p
                ?? = 1.0
                while norm(f(xk+??*dk))^2 > norm(f(xk))^2 + 2*??*??*g'dk
                    ?? = ??*??
                end
                xn .= xk + ??*dk
            else
                ?? = 1.0
                while true
                    xt .= xk-??*g
                    box_projection!(xt,lb,ub)
                    if norm(f(xt))^2 <= norm(f(xk))^2 + 2*??*g'*(xt-xk)
                        xn .=  xt
                        break
                    else
                        ?? = ??*??
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

        ??k = min(??k,norm(f(xk))^2)

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
    ?? = 0.9
    ?? = 0.99995
    ?? = 10^(-4)
    ?? = 10^(-8)
    p = 2.1

    ??k = max(0.5*10^(-8)*norm(ffk)^2,1e-4)

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,ffk,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + ??k*I)\(jk'ffk)
        xn .= xk + dk
        box_projection!(xn,lb,ub)

        f(ffn,xn)
        if norm(ffn) > ??*norm(ffk)
            g .= jk'ffk
            if g'dk <= -??*norm(dk)^p
                ?? = 1.0
                while true
                    f(ffn,xk+??*dk)
                    if norm(ffn)^2 > norm(ffk)^2 + 2*??*??*g'dk
                        ?? = ??*??
                    else
                        break
                    end
                end
                xn .= xk + ??*dk
            else
                ?? = 1.0
                while true
                    xt .= xk-??*g
                    box_projection!(xt,lb,ub)
                    f(ffn,xt)
                    if norm(ffn)^2 <= norm(ffk)^2 + 2*??*g'*(xt-xk)
                        xn .=  xt
                        break
                    else
                        ?? = ??*??
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

        ??k = min(??k,norm(ffn)^2)

    end
  
    f(ffn,xn)
    results = SolverResults(:lm_kyf,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
   
end

### Constrained Levenberg-Marquardt-ar

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
    ??1 = 0.005
    ??2 = 0.005
    ??  = 0.8
    r  = 0.5
    ??  = 10^(-16)
    ??  = 10^(-4)

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        ??k = ??*norm(f(xk))^2

        d1k = -(jk'jk + ??k*I)\(jk'f(xk))
        d2k = -(jk'jk + ??k*I)\(jk'f(xk+d1k))
        d3k = -(jk'jk + ??k*I)\(jk'f(xk+d1k+d2k))

        dk = d1k+d2k+d3k

        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk

        if norm(f(z)) <= ??*norm(f(xk))
            ?? = 1.0
        else
            if f(xk)'jk*dk > -??
                dk = d1k
                z .= xk+dk
                box_projection!(z,lb,ub)
                s .= z-xk
            end
            ?? = 1.0
            epsilon = 1/10
            while true
                if norm(f(xk+??*s))^2 > (1+epsilon)*norm(f(xk))^2 - ??1*??^2*norm(s)^2 - ??2*??^2*norm(f(xk))^2
                    ?? = r*??
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + ??*s

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
    ??1 = 0.005
    ??2 = 0.005
    ??  = 0.8
    r  = 0.5
    ??  = 10^(-16)
    ??  = 10^(-4)

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,ffk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        ??k = ??*norm(ffk)^2

        d1k = -(jk'jk + ??k*I)\(jk'ffk)
        f(ffn,xk+d1k)
        d2k = -(jk'jk + ??k*I)\(jk'ffn)
        f(ffn,xk+d1k+d2k)
        d3k = -(jk'jk + ??k*I)\(jk'ffn)

        dk = d1k+d2k+d3k

        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk

        f(ffn,z)
        if norm(ffn) <= ??*norm(ffk)
            ?? = 1.0
        else
            if ffk'jk*dk > -??
                dk = d1k
                z .= xk+dk
                box_projection!(z,lb,ub)
                s .= z-xk
            end
            ?? = 1.0
            epsilon = 1/10
            while true
                f(ffn,xk+??*s)
                if norm(ffn)^2 > (1+epsilon)*norm(ffk)^2 - ??1*??^2*norm(s)^2 - ??2*??^2*norm(ffk)^2
                    ?? = r*??
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + ??*s

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
    ??1 = 0.005
    ??2 = 0.005
    ??  = 0.8
    r  = 0.5
    ??  = 10^(-16)
    ??  = 10^(-4)

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

        ??k = ??*norm(f(xk))^2

        d1k = -(jk'jk + ??k*I)\(jk'f(xk))
        d2k = -(jk'jk + ??k*I)\(jk'f(xk+d1k))
        d3k = -(jk'jk + ??k*I)\(jk'f(xk+d1k+d2k))

        dk = d1k+d2k+d3k

        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk

        if norm(f(z)) <= ??*norm(f(xk))
            ?? = 1.0
        else
            if f(xk)'jk*dk > -??
                dk = d1k
                z .= xk+dk
                box_projection!(z,lb,ub)
                s .= z-xk
            end
            ?? = 1.0
            epsilon = 1/10
            while true
                if norm(f(xk+??*s))^2 > (1+epsilon)*norm(f(xk))^2 - ??1*??^2*norm(s)^2 - ??2*??^2*norm(f(xk))^2
                    ?? = r*??
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + ??*s

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
    ??1 = 0.005
    ??2 = 0.005
    ??  = 0.8
    r  = 0.5
    ??  = 10^(-16)
    ??  = 10^(-4)

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
        ??k = ??*norm(ffk)^2

        d1k = -(jk'jk + ??k*I)\(jk'ffk)
        f(ffn,xk+d1k)
        d2k = -(jk'jk + ??k*I)\(jk'ffn)
        f(ffn,xk+d1k+d2k)
        d3k = -(jk'jk + ??k*I)\(jk'ffn)

        dk = d1k+d2k+d3k

        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk

        f(ffn,z)
        if norm(ffn) <= ??*norm(ffk)
            ?? = 1.0
        else
            if ffk'jk*dk > -??
                dk = d1k
                z .= xk+dk
                box_projection!(z,lb,ub)
                s .= z-xk
            end
            ?? = 1.0
            epsilon = 1/10
            while true
                f(ffn,xk+??*s)
                if norm(ffn)^2 > (1+epsilon)*norm(ffk)^2 - ??1*??^2*norm(s)^2 - ??2*??^2*norm(ffk)^2
                    ?? = r*??
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + ??*s

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
    ??1 = 0.005
    ??2 = 0.005
    ??  = 0.8
    r  = 0.5
    ??  = 10^(-16)
    ??  = 10^(-4)

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        ??k = ??*norm(f(xk))^2

        d1k = -(jk'jk + ??k*I)\(jk'f(xk))
        d2k = -(jk'jk + ??k*I)\(jk'f(xk+d1k))
        d3k = -(jk'jk + ??k*I)\(jk'f(xk+d1k+d2k))

        dk = d1k+d2k+d3k

        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk

        if norm(f(z)) <= ??*norm(f(xk))
            ?? = 1.0
        else
            if f(xk)'jk*dk > -??
                dk = d1k
                z .= xk+dk
                box_projection!(z,lb,ub)
                s .= z-xk
            end
            ?? = 1.0
            epsilon = 1/10
            while true
                if norm(f(xk+??*s))^2 > (1+epsilon)*norm(f(xk))^2 - ??1*??^2*norm(s)^2 - ??2*??^2*norm(f(xk))^2
                    ?? = r*??
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + ??*s

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
    ??1 = 0.005
    ??2 = 0.005
    ??  = 0.8
    r  = 0.5
    ??  = 10^(-16)
    ??  = 10^(-4)

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,ffk,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        ??k = ??*norm(ffk)^2

        d1k = -(jk'jk + ??k*I)\(jk'ffk)
        f(ffn,xk+d1k)
        d2k = -(jk'jk + ??k*I)\(jk'ffn)
        f(ffn,xk+d1k+d2k)
        d3k = -(jk'jk + ??k*I)\(jk'ffn)

        dk = d1k+d2k+d3k

        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk

        f(ffn,z)
        if norm(ffn) <= ??*norm(ffk)
            ?? = 1.0
        else
            if ffk'jk*dk > -??
                dk = d1k
                z .= xk+dk
                box_projection!(z,lb,ub)
                s .= z-xk
            end
            ?? = 1.0
            epsilon = 1/10
            while true
                f(ffn,xk+??*s)
                if norm(ffn)^2 > (1+epsilon)*norm(ffk)^2 - ??1*??^2*norm(s)^2 - ??2*??^2*norm(ffk)^2
                    ?? = r*??
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + ??*s

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

### Constrained doglog

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
        return constrained_dogleg_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_dogleg_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_dogleg_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = Array{T,1}(undef,n)
    xt = Array{T,1}(undef,n)
    pnk = Array{T,1}(undef,n)
    pck = Array{T,1}(undef,n)
    p = Array{T,1}(undef,n)
    pdiff = Array{T,1}(undef,n)
    gk = Array{T,1}(undef,n)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    ??  = 1.0
    ??1 = 2.0
    ??2 = 0.25
    ??1 = 0.25
    ??2 = 0.75
    ??hat = 1/(100*eps())

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        pnk .= -jk\f(xk) # This is the Newton step
        xn .= xk .+ pnk
        box_projection!(xn,lb,ub)
        pnk .= xn .- xk
        if norm(pnk) <= ?? # The Newton step is inside the trust region
            p .= pnk
        else # The Newton step is outside the trust region
            gk .= jk'f(xk)
            #pck .= -((gk'gk)/(gk'jk'jk*gk))*gk # This is the Cauchy step
            pck .= -gk # This is the Cauchy step
            xt .= xk .+ pck
            box_projection!(xt,lb,ub)
            pck .= xt .- xk
            if norm(pck) > ?? # The Cauchy step is outside the trust region
                p .= (??/norm(pck))*pck
            else # The Cauchy step is inside the trust region
                pdiff .= pnk-pck
                c = norm(pck)^2-??^2
                b = 2*pck'pdiff
                a = norm(pdiff)
                if b^2-4*a*c > 0.0
                    ??1 = (1/(2*a))*(-b + sqrt(b^2-4*a*c))
                    ??2 = (1/(2*a))*(-b - sqrt(b^2-4*a*c))
                    ?? = max(??1,??2)
                else
                    ?? = 0.0
                end
                p .= pck .+ ??*pdiff
            end
        end

        # Now check and update the trust region

        ?? = (norm(f(xk)) - norm(f(xk+p)))/(norm(f(xk)) - norm(f(xk) + jk'*p))
        if ?? >= ??1
            xn .= xk .+ p
            box_projection!(xn,lb,ub)    
            if ?? >= ??2 && isapprox(norm(p),??)
                ?? = min(??1*norm(p),??hat)
            else
                ?? = norm(p)
            end
        else
            ?? = ??2*??
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

function constrained_dogleg_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = Array{T,1}(undef,n)
    xt = Array{T,1}(undef,n)
    pnk = Array{T,1}(undef,n)
    pck = Array{T,1}(undef,n)
    p = Array{T,1}(undef,n)
    pdiff = Array{T,1}(undef,n)
    gk = Array{T,1}(undef,n)
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
    ??  = 1.0
    ??1 = 2.0
    ??2 = 0.25
    ??1 = 0.25
    ??2 = 0.75
    ??hat = 1/(100*eps())

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,ffk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        pnk .= -jk\ffk # This is the Newton step
        xn .= xk .+ pnk
        box_projection!(xn,lb,ub)
        pnk .= xn .- xk
        if norm(pnk) <= ?? # The Newton step is inside the trust region
            p .= pnk
        else # The Newton step is outside the trust region
            gk .= jk'ffk
            #pck .= -((gk'gk)/(gk'jk'jk*gk))*gk # This is the Cauchy step
            pck .= -gk # This is the Cauchy step
            xt .= xk .+ pck
            box_projection!(xt,lb,ub)
            pck .= xt .- xk
            if norm(pck) > ?? # The Cauchy step is outside the trust region
                p .= -(??/norm(pck))*pck
            else # The Cauchy step is inside the trust region
                pdiff .= pnk-pck
                c = norm(pck)^2-??^2
                b = 2*pck'pdiff
                a = norm(pdiff)
                if b^2-4*a*c > 0.0
                    ??1 = (1/(2*a))*(-b + sqrt(b^2-4*a*c))
                    ??2 = (1/(2*a))*(-b - sqrt(b^2-4*a*c))
                    ?? = max(??1,??2)
                else
                    ?? = 0.0
                end
                p .= pck .+ ??*pdiff
            end
        end

        # Now check and update the trust region

        f(ffn,xk+p)
        ?? = (norm(ffk) - norm(ffn))/(norm(ffk) - norm(ffk + jk'*p))
        if ?? >= ??1
            xn .= xk .+ p
            box_projection!(xn,lb,ub)    
            if ?? >= ??2 && isapprox(norm(p),??)
                ?? = min(??1*norm(p),??hat)
            else
                ?? = norm(p)
            end
        else
            ?? = ??2*??
        end
       
        f(ffk,xn)
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

    f(ffk,xn)
    results = SolverResults(:dogleg,x,xn,ffk,lenx,lenf,iter,solution_trace)

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
        return constrained_dogleg_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_dogleg_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_dogleg_outplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    j_inplace = !applicable(j,x)

    n = length(x)
    xk = copy(x)
    xn = Array{T,1}(undef,n)
    xt = Array{T,1}(undef,n)
    pnk = Array{T,1}(undef,n)
    pck = Array{T,1}(undef,n)
    p = Array{T,1}(undef,n)
    pdiff = Array{T,1}(undef,n)
    gk = Array{T,1}(undef,n)
    jk = Array{T,2}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    ??  = 1.0
    ??1 = 2.0
    ??2 = 0.25
    ??1 = 0.25
    ??2 = 0.75
    ??hat = 1/(100*eps())

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

        pnk .= -jk\f(xk) # This is the Newton step
        xn .= xk .+ pnk
        box_projection!(xn,lb,ub)
        pnk .= xn .- xk
        if norm(pnk) <= ?? # The Newton step is inside the trust region
            p .= pnk
        else # The Newton step is outside the trust region
            gk .= jk'f(xk)
            #pck .= -((gk'gk)/(gk'jk'jk*gk))*gk # This is the Cauchy step
            pck .= -gk # This is the Cauchy step
            xt .= xk .+ pck
            box_projection!(xt,lb,ub)
            pck .= xt .- xk
            if norm(pck) > ?? # The Cauchy step is outside the trust region
                p .= -(??/norm(gk))*gk
            else # The Cauchy step is inside the trust region
                pdiff .= pnk-pck
                c = norm(pck)^2-??^2
                b = 2*pck'pdiff
                a = norm(pdiff)
                if b^2-4*a*c > 0.0
                    ??1 = (1/(2*a))*(-b + sqrt(b^2-4*a*c))
                    ??2 = (1/(2*a))*(-b - sqrt(b^2-4*a*c))
                    ?? = max(??1,??2)
                else
                    ?? = 0.0
                end
                p .= pck .+ ??*pdiff
            end
        end

        # Now check and update the trust region

        ?? = (norm(f(xk)) - norm(f(xk+p)))/(norm(f(xk)) - norm(f(xk) + jk'*p))
        if ?? >= ??1
            xn .= xk .+ p
            box_projection!(xn,lb,ub)    
            if ?? >= ??2 && isapprox(norm(p),??)
                ?? = min(??1*norm(p),??hat)
            else
                ?? = norm(p)
            end
        else
            ?? = ??2*??
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

function constrained_dogleg_inplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    j_inplace = !applicable(j,x)

    n = length(x)
    xk = copy(x)
    xn = Array{T,1}(undef,n)
    xt = Array{T,1}(undef,n)
    pnk = Array{T,1}(undef,n)
    pck = Array{T,1}(undef,n)
    p = Array{T,1}(undef,n)
    pdiff = Array{T,1}(undef,n)
    gk = Array{T,1}(undef,n)
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
    ??  = 1.0
    ??1 = 2.0
    ??2 = 0.25
    ??1 = 0.25
    ??2 = 0.75
    ??hat = 1/(100*eps())

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
        pnk .= -jk\ffk # This is the Newton step
        xn .= xk .+ pnk
        box_projection!(xn,lb,ub)
        pnk .= xn .- xk
        if norm(pnk) <= ?? # The Newton step is inside the trust region
            p .= pnk
        else # The Newton step is outside the trust region
            gk .= jk'ffk
            #pck .= -((gk'gk)/(gk'jk'jk*gk))*gk # This is the Cauchy step
            pck .= -gk # This is the Cauchy step
            xt .= xk .+ pck
            box_projection!(xt,lb,ub)
            pck .= xt .- xk
            if norm(pck) > ?? # The Cauchy step is outside the trust region
                p .= -(??/norm(pck))*pck
            else # The Cauchy step is inside the trust region
                pdiff .= pnk-pck
                c = norm(pck)^2-??^2
                b = 2*pck'pdiff
                a = norm(pdiff)
                if b^2-4*a*c > 0.0
                    ??1 = (1/(2*a))*(-b + sqrt(b^2-4*a*c))
                    ??2 = (1/(2*a))*(-b - sqrt(b^2-4*a*c))
                    ?? = max(??1,??2)
                else
                    ?? = 0.0
                end
                p .= pck .+ ??*pdiff
            end
        end

        # Now check and update the trust region

        f(ffn,xk+p)
        ?? = (norm(ffk) - norm(ffn))/(norm(ffk) - norm(ffk + jk'*p))
        if ?? >= ??1
            xn .= xk .+ p
            box_projection!(xn,lb,ub)    
            if ?? >= ??2 && isapprox(norm(p),??)
                ?? = min(??1*norm(p),??hat)
            else
                ?? = norm(p)
            end
        else
            ?? = ??2*??
        end
  
        f(ffk,xn)
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
  
    f(ffk,xn)
    results = SolverResults(:dogleg,x,xn,ffk,lenx,lenf,iter,solution_trace)

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
        return constrained_dogleg_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_dogleg_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_dogleg_sparse_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = Array{T,1}(undef,n)
    xt = Array{T,1}(undef,n)
    pnk = Array{T,1}(undef,n)
    pck = Array{T,1}(undef,n)
    p = Array{T,1}(undef,n)
    pdiff = Array{T,1}(undef,n)
    gk = Array{T,1}(undef,n)
    jk = sparse(Array{T,2}(undef,n,n))

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    ??  = 1.0
    ??1 = 2.0
    ??2 = 0.25
    ??1 = 0.25
    ??2 = 0.75
    ??hat = 1/(100*eps())

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end

        pnk .= -jk\f(xk) # This is the Newton step
        xn .= xk .+ pnk
        box_projection!(xn,lb,ub)
        pnk .= xn .- xk
        if norm(pnk) <= ?? # The Newton step is inside the trust region
            p .= pnk
        else # The Newton step is outside the trust region
            gk .= jk'f(xk)
            #pck .= -((gk'gk)/(gk'jk'jk*gk))*gk # This is the Cauchy step
            pck .= -gk # This is the Cauchy step
            xt .= xk .+ pck
            box_projection!(xt,lb,ub)
            pck .= xt .- xk
            if norm(pck) > ?? # The Cauchy step is outside the trust region
                p .= -(??/norm(pck))*pck
            else # The Cauchy step is inside the trust region
                pdiff .= pnk-pck
                c = norm(pck)^2-??^2
                b = 2*pck'pdiff
                a = norm(pdiff)
                if b^2-4*a*c > 0.0
                    ??1 = (1/(2*a))*(-b + sqrt(b^2-4*a*c))
                    ??2 = (1/(2*a))*(-b - sqrt(b^2-4*a*c))
                    ?? = max(??1,??2)
                else
                    ?? = 0.0
                end
                p .= pck .+ ??*pdiff
            end
        end

        # Now check and update the trust region

        ?? = (norm(f(xk)) - norm(f(xk+p)))/(norm(f(xk)) - norm(f(xk) + jk'*p))
        if ?? >= ??1
            xn .= xk .+ p
            box_projection!(xn,lb,ub)    
            if ?? >= ??2 && isapprox(norm(p),??)
                ?? = min(??1*norm(p),??hat)
            else
                ?? = norm(p)
            end
        else
            ?? = ??2*??
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

function constrained_dogleg_sparse_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = Array{T,1}(undef,n)
    xt = Array{T,1}(undef,n)
    pnk = Array{T,1}(undef,n)
    pck = Array{T,1}(undef,n)
    p = Array{T,1}(undef,n)
    pdiff = Array{T,1}(undef,n)
    gk = Array{T,1}(undef,n)
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
    ??  = 1.0
    ??1 = 2.0
    ??2 = 0.25
    ??1 = 0.25
    ??2 = 0.75
    ??hat = 1/(100*eps())

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,ffk,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        pnk .= -jk\ffk # This is the Newton step
        xn .= xk .+ pnk
        box_projection!(xn,lb,ub)
        pnk .= xn .- xk
        if norm(pnk) <= ?? # The Newton step is inside the trust region
            p .= pnk
        else # The Newton step is outside the trust region
            gk .= jk'ffk
            #pck .= -((gk'gk)/(gk'jk'jk*gk))*gk # This is the Cauchy step
            pck .= -gk # This is the Cauchy step
            xt .= xk .+ pck
            box_projection!(xt,lb,ub)
            pck .= xt .- xk
            if norm(pck) > ?? # The Cauchy step is outside the trust region
                p .= -(??/norm(pck))*pck
            else # The Cauchy step is inside the trust region
                pdiff .= pnk-pck
                c = norm(pck)^2-??^2
                b = 2*pck'pdiff
                a = norm(pdiff)
                if b^2-4*a*c > 0.0
                    ??1 = (1/(2*a))*(-b + sqrt(b^2-4*a*c))
                    ??2 = (1/(2*a))*(-b - sqrt(b^2-4*a*c))
                    ?? = max(??1,??2)
                else
                    ?? = 0.0
                end
                p .= pck .+ ??*pdiff
            end
        end

        # Now check and update the trust region

        f(ffn,xk+p)
        ?? = (norm(ffk) - norm(ffn))/(norm(ffk) - norm(ffk + jk'*p))
        if ?? >= ??1
            xn .= xk .+ p
            box_projection!(xn,lb,ub)    
            if ?? >= ??2 && isapprox(norm(p),??)
                ?? = min(??1*norm(p),??hat)
            else
                ?? = norm(p)
            end
        else
            ?? = ??2*??
        end

        f(ffk,xn)
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
  
    f(ffk,xn)
    results = SolverResults(:dogleg,x,xn,ffk,lenx,lenf,iter,solution_trace)

    return results
  
end

### Constrained dogleg-bmp

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

function step_selection_outplace(f::Function,x::Array{T,1},Gk::AbstractArray{T,2},jk::AbstractArray{T,2},gk::Array{T,1},pkn::Array{T,1},lb::Array{T,1},ub::Array{T,1},??::T,??::T) where {T <: AbstractFloat}

    ??k = Inf
    for i in eachindex(gk)
        if gk[i] != 0.0
            ??k = min(max(((lb[i] - x[i])/gk[i]),((ub[i]-x[i])/gk[i])),??k)
        else
            ??k = minimum((Inf,??k))
        end
    end

    taukprime = min(-(f(x)'*jk*gk)/norm(jk*gk)^2,??/norm(Gk*gk))
    tauk = taukprime
    if !isequal(x+taukprime*gk, box_projection(x+taukprime*gk,lb,ub))
        tauk = ??*??k
    end

    pc = tauk*gk

    pc_pkn = pc-pkn
    
    if norm(-pc_pkn) < 1e-13 # "division by zero errors can otherwise occur"
        return (pc+pkn)/2
    else
        ??hat = -(f(x)+jk*pc)'*jk*(-pc_pkn)/norm(jk*(-pc_pkn))^2
        if (pc'Gk^2*(pc_pkn))^2 - norm(Gk*(pc_pkn))^2*(norm(Gk*pc)^2-??^2) >= 0.0
            r = ((pc'Gk^2*(pc_pkn))^2 - norm(Gk*(pc_pkn))^2*(norm(Gk*pc)^2-??^2))^0.5
            ??plus  = (pc'*Gk^2*(pc_pkn) + r)/norm(Gk*(pc_pkn))^2
            ??minus = (pc'*Gk^2*(pc_pkn) - r)/norm(Gk*(pc_pkn))^2
        else
            ??plus  = pc'*Gk^2*(pc_pkn)/norm(Gk*(pc_pkn))^2
            ??minus = pc'*Gk^2*(pc_pkn)/norm(Gk*(pc_pkn))^2
        end

        if ??hat > 1.0
            ??tildaplus = Inf
            for i in eachindex(pkn)
                if (pkn[i]-pc[i]) != 0.0
                    ??tildaplus = min(max(((lb[i] - x[i]- pc[i])/(pkn[i]-pc[i])),((ub[i] - x[i] - pc[i])/(pkn[i]-pc[i]))),??tildaplus)
                else
                    ??tildaplus = min(Inf,??tildaplus)
                end
            end
            ?? = min(??hat,??plus,??*??tildaplus)
        elseif ??hat < 0.0
            ??tildaminus = -Inf
            for i in eachindex(pkn)
                if (-pkn[i]+pc[i]) != 0.0
                    ??tildaminus = min(max(-((lb[i] - x[i]- pc[i])/(-(pkn[i]-pc[i]))),-((ub[i] - x[i] - pc[i])/(-(pkn[i]-pc[i])))),??tildaminus)
                else
                    ??tildaminus = min(Inf,??tildaminus)
                end
            end
            ?? = max(??hat,??minus,??*??tildaminus)
        else
            ?? = ??hat
        end

        p = (1.0-??)*pc + ??*pkn

        return p

    end

end

function step_selection_inplace(f::Function,x::Array{T,1},Gk::AbstractArray{T,2},jk::AbstractArray{T,2},gk::Array{T,1},pkn::Array{T,1},lb::Array{T,1},ub::Array{T,1},??::T,??::T) where {T <: AbstractFloat}

    ff = Array{T,1}(undef,length(x))

    ??k = Inf
    for i in eachindex(gk)
        if gk[i] != 0.0
            ??k = min(max(((lb[i] - x[i])/gk[i]),((ub[i]-x[i])/gk[i])),??k)
        else
            ??k = minimum((Inf,??k))
        end
    end

    f(ff,x)
    taukprime = min(-(ff'*jk*gk)/norm(jk*gk)^2,??/norm(Gk*gk))

    tauk = taukprime
    if !isequal(x+taukprime*gk, box_projection(x+taukprime*gk,lb,ub))
        tauk = ??*??k
    end

    pc = tauk*gk
    
    pc_pkn = pc-pkn
    
    if norm(-pc_pkn) < 1e-13 # "division by zero errors can otherwise occur"
        return (pc+pkn)/2
    else
        ??hat = -(ff+jk*pc)'*jk*(-pc_pkn)/norm(jk*(-pc_pkn))^2
        if (pc'Gk^2*(pc_pkn))^2 - norm(Gk*(pc_pkn))^2*(norm(Gk*pc)^2-??^2) >= 0.0
            r = ((pc'Gk^2*(pc_pkn))^2 - norm(Gk*(pc_pkn))^2*(norm(Gk*pc)^2-??^2))^0.5
            ??plus  = (pc'*Gk^2*(pc_pkn) + r)/norm(Gk*(pc_pkn))^2
            ??minus = (pc'*Gk^2*(pc_pkn) - r)/norm(Gk*(pc_pkn))^2
        else
            ??plus  = pc'*Gk^2*(pc_pkn)/norm(Gk*(pc_pkn))^2
            ??minus = pc'*Gk^2*(pc_pkn)/norm(Gk*(pc_pkn))^2
        end

        if ??hat > 1.0
            ??tildaplus = Inf
            for i in eachindex(pkn)
                if (pkn[i]-pc[i]) != 0.0
                    ??tildaplus = min(max(((lb[i] - x[i]- pc[i])/(pkn[i]-pc[i])),((ub[i] - x[i] - pc[i])/(pkn[i]-pc[i]))),??tildaplus)
                else
                    ??tildaplus = min(Inf,??tildaplus)
                end
            end
            ?? = min(??hat,??plus,??*??tildaplus)
        elseif ??hat < 0.0
            ??tildaminus = -Inf
            for i in eachindex(pkn)
                if (-pkn[i]+pc[i]) != 0.0
                    ??tildaminus = min(max(-((lb[i] - x[i]- pc[i])/(-(pkn[i]-pc[i]))),-((ub[i] - x[i] - pc[i])/(-(pkn[i]-pc[i])))),??tildaminus)
                else
                    ??tildaminus = min(Inf,??tildaminus)
                end
            end
            ?? = max(??hat,??minus,??*??tildaminus)
        else
            ?? = ??hat
        end

        p = (1.0-??)*pc + ??*pkn

        return p

    end

end

### Constrained dogleg-bmp solvers

function constrained_dogleg_bmp_solver(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_dogleg_bmp_solver_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_dogleg_bmp_solver_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_dogleg_bmp_solver_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

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
    ??  = 1.0 
    ??1 = 2.0
    ??2 = 0.25
    ??  = 0.99995 
    ??1 = 0.5 
    ??2 = 0.9 

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
   
        ?? = max(??,1.0-norm(f(xk)))
        pkn .= xk-jk\f(xk)
        box_projection!(pkn,lb,ub)
        pkn .= ??*(pkn-xk)

        p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,??,??)
    
        while true
            ?? = (norm(f(xk)) - norm(f(xk+p))) / (norm(f(xk)) - norm(f(xk) + jk'*p))
            if ?? < ??1 # linear approximation is poor fit so reduce the trust region
                ?? = min(??2*??,0.5*norm(p))
                p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,??,??)
            elseif ?? > ??2 # linear approximation is good fit so expand the trust region
                ?? = max(??,??1*norm(p))
                p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,??,??)
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
  
    results = SolverResults(:dogleg_bmp,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_bmp_solver_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

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
    ??  = 1.0 
    ??1 = 2.0
    ??2 = 0.25
    ??  = 0.99995 
    ??1 = 0.5 
    ??2 = 0.9 

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
   
        ?? = max(??,1.0-norm(ffk))
        pkn .= xk-jk\ffk
        box_projection!(pkn,lb,ub)
        pkn .= ??*(pkn-xk)

        p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,??,??)
    
        while true
            f(ffn,xk+p)
            ?? = (norm(ffk) - norm(ffn)) / (norm(ffk) - norm(ffk + jk'p))
            if ?? < ??1 # linear approximation is poor fit so reduce the trust region
                ?? = min(??2*??,0.5*norm(p))
                p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,??,??)
            elseif ?? > ??2 # linear approximation is good fit so expand the trust region
                ?? = max(??,??1*norm(p))
                p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,??,??)
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
    results = SolverResults(:dogleg_bmp,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_bmp_solver(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_dogleg_bmp_solver_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_dogleg_bmp_solver_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_dogleg_bmp_solver_outplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

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
    ??  = 1.0 
    ??1 = 2.0
    ??2 = 0.25
    ??  = 0.99995 
    ??1 = 0.5 
    ??2 = 0.9 

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
   
        ?? = max(??,1.0-norm(f(xk)))
        pkn .= xk-jk\f(xk)
        box_projection!(pkn,lb,ub)
        pkn .= ??*(pkn-xk)

        p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,??,??)
    
        while true
            ?? = (norm(f(xk)) - norm(f(xk+p))) / (norm(f(xk)) - norm(f(xk) + jk'*p))
            if ?? < ??1 # linear approximation is poor fit so reduce the trust region
                ?? = min(??2*??,0.5*norm(p))
                p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,??,??)
            elseif ?? > ??2 # linear approximation is good fit so expand the trust region
                ?? = max(??,??1*norm(p))
                p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,??,??)
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
  
    results = SolverResults(:dogleg_bmp,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_bmp_solver_inplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

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
    ??  = 1.0 
    ??1 = 2.0
    ??2 = 0.25
    ??  = 0.99995 
    ??1 = 0.5 
    ??2 = 0.9 

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
   
        ?? = max(??,1.0-norm(ffk))
        pkn .= xk-jk\ffk
        box_projection!(pkn,lb,ub)
        pkn .= ??*(pkn-xk)

        p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,??,??)
    
        while true
            f(ffn,xk+p)
            ?? = (norm(ffk) - norm(ffn)) / (norm(ffk) - norm(ffk + jk'p))
            if ?? < ??1 # linear approximation is poor fit so reduce the trust region
                ?? = min(??2*??,0.5*norm(p))
                p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,??,??)
            elseif ?? > ??2 # linear approximation is good fit so expand the trust region
                ?? = max(??,??1*norm(p))
                p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,??,??)
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
    results = SolverResults(:dogleg_bmp,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_bmp_solver_sparse(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_dogleg_bmp_solver_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    else
        return constrained_dogleg_bmp_solver_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end

function constrained_dogleg_bmp_solver_sparse_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

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
    ??  = 1.0 
    ??1 = 2.0
    ??2 = 0.25
    ??  = 0.99995 
    ??1 = 0.5 
    ??2 = 0.9 

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
   
        ?? = max(??,1.0-norm(f(xk)))
        pkn .= xk-jk\f(xk)
        box_projection!(pkn,lb,ub)
        pkn .= ??*(pkn-xk)

        p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,??,??)
    
        while true
            ?? = (norm(f(xk)) - norm(f(xk+p))) / (norm(f(xk)) - norm(f(xk) + jk'*p))
            if ?? < ??1 # linear approximation is poor fit so reduce the trust region
                ?? = min(??2*??,0.5*norm(p))
                p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,??,??)
            elseif ?? > ??2 # linear approximation is good fit so expand the trust region
                ?? = max(??,??1*norm(p))
                p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,??,??)
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
  
    results = SolverResults(:dogleg_bmp,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_bmp_solver_sparse_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

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
    ??  = 1.0 
    ??1 = 2.0
    ??2 = 0.25
    ??  = 0.99995 
    ??1 = 0.5 
    ??2 = 0.9 

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
   
        ?? = max(??,1.0-norm(ffk))
        pkn .= xk-jk\ffk
        box_projection!(pkn,lb,ub)
        pkn .= ??*(pkn-xk)

        p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,??,??)
    
        while true
            f(ffn,xk+p)
            ?? = (norm(ffk) - norm(ffn)) / (norm(ffk) - norm(ffk + jk'p))
            if ?? < ??1 # linear approximation is poor fit so reduce the trust region
                ?? = min(??2*??,0.5*norm(p))
                p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,??,??)
            elseif ?? > ??2 # linear approximation is good fit so expand the trust region
                ?? = max(??,??1*norm(p))
                p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,??,??)
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
    results = SolverResults(:dogleg_bmp,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end


### Constrained Newton-Krylov

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
    ??k = 1e-4#1e-4
    ??  = 0.9
    t  = 1e-4
    ??  = 1e-4
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
            dk, status = gmres(jk,-f(xk),xk,??k,krylovdim) # Inexact restarted

            ?? = 1.0
            m = 1
            while m <= mmax
                xt .= xk+??*dk
                box_projection!(xt,lb,ub)
                if norm(f(xt)) <= (1.0 - t*??*(1-??k))*norm(f(xk))
                    xn .= xt
                    ??k = min(1.0 - ??*(1.0 - ??k),1e-3)
                    flag_ng = false
                    break
                else
                    ?? = ??*??
                    m += 1
                    flag_ng = true
                end
            end
        else # Take a Newton step
            dk = -jk'f(xk)
            ?? = 1.0
            m = 1
            while m <= mmax
                xt .= xk+??*dk
                box_projection!(xt,lb,ub)
                if g(xt) <= g(xk) + ??*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    ?? = ??*??
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
    ??k = 1e-4#1e-4
    ??  = 0.9
    t  = 1e-4
    ??  = 1e-4
    mmax = 50

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false
            jk .= ForwardDiff.jacobian(f,ffk,xk)
            if !all(isfinite,jk)
                error("The jacobian has non-finite elements")
            end    
            dk, status = gmres(jk,-ffk,xk,??k,krylovdim) # Inexact restarted

            ?? = 1.0
            m = 1
            while m <= mmax
                xt .= xk+??*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if norm(ffn) <= (1.0 - t*??*(1-??k))*norm(ffk)
                    xn .= xt
                    ??k = min(1.0 - ??*(1.0 - ??k),1e-3)
                    flag_ng = false
                    break
                else
                    ?? = ??*??
                    m += 1
                    flag_ng = true
                end
            end
        else # Take a Newton step
            dk = -jk'ffk
            ?? = 1.0
            m = 1
            while m <= mmax
                xt .= xk+??*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if 0.5*norm(ffn)^2 <=  0.5*norm(ffk)^2 + ??*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    ?? = ??*??
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
    ??k = 1e-4#1e-4
    ??  = 0.9
    t  = 1e-4
    ??  = 1e-4
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
    
            dk, status = gmres(jk,-f(xk),xk,??k,krylovdim) # Inexact restarted

            ?? = 1.0
            m = 1
            while m <= mmax
                xt .= xk+??*dk
                box_projection!(xt,lb,ub)
                if norm(f(xt)) <= (1.0 - t*??*(1-??k))*norm(f(xk))
                    xn .= xt
                    ??k = min(1.0 - ??*(1.0 - ??k),1e-3)
                    flag_ng = false
                    break
                else
                    ?? = ??*??
                    m += 1
                    flag_ng = true
                end
            end
        else # Take a Newton step
            dk = -jk'f(xk)
            ?? = 1.0
            m = 1
            while m <= mmax
                xt .= xk+??*dk
                box_projection!(xt,lb,ub)
                if g(xt) <= g(xk) + ??*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    ?? = ??*??
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
    ??k = 1e-4#1e-4
    ??  = 0.9
    t  = 1e-4
    ??  = 1e-4
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
            dk, status = gmres(jk,-ffk,xk,??k,krylovdim) # Inexact restarted

            ?? = 1.0
            m = 1
            while m <= mmax
                xt .= xk+??*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if norm(ffn) <= (1.0 - t*??*(1-??k))*norm(ffk)
                    xn .= xt
                    ??k = min(1.0 - ??*(1.0 - ??k),1e-3)
                    flag_ng = false
                    break
                else
                    ?? = ??*??
                    m += 1
                    flag_ng = true
                end
            end
        else # Take a Newton step
            dk = -jk'ffk
            ?? = 1.0
            m = 1
            while m <= mmax
                xt .= xk+??*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if 0.5*norm(ffn)^2 <=  0.5*norm(ffk)^2 + ??*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    ?? = ??*??
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
    ??k = 1e-4#1e-4
    ??  = 0.9
    t  = 1e-4
    ??  = 1e-4
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
            dk, status = gmres(jk,-f(xk),xk,??k,krylovdim) # Inexact restarted

            ?? = 1.0
            m = 1
            while m <= mmax
                xt .= xk+??*dk
                box_projection!(xt,lb,ub)
                if norm(f(xt)) <= (1.0 - t*??*(1-??k))*norm(f(xk))
                    xn .= xt
                    ??k = min(1.0 - ??*(1.0 - ??k),1e-3)
                    flag_ng = false
                    break
                else
                    ?? = ??*??
                    m += 1
                    flag_ng = true
                end
            end
        else # Take a Newton step
            dk = -jk'f(xk)
            ?? = 1.0
            m = 1
            while m <= mmax
                xt .= xk+??*dk
                box_projection!(xt,lb,ub)
                if g(xt) <= g(xk) + ??*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    ?? = ??*??
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
    ??k = 1e-4#1e-4
    ??  = 0.9
    t  = 1e-4
    ??  = 1e-4
    mmax = 50

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false
            jk .= sparse(ForwardDiff.jacobian(f,ffk,xk))
            if !all(isfinite,nonzeros(jk))
                error("The jacobian has non-finite elements")
            end    
            dk, status = gmres(jk,-ffk,xk,??k,krylovdim) # Inexact restarted

            ?? = 1.0
            m = 1
            while m <= mmax
                xt .= xk+??*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if norm(ffn) <= (1.0 - t*??*(1-??k))*norm(ffk)
                    xn .= xt
                    ??k = min(1.0 - ??*(1.0 - ??k),1e-3)
                    flag_ng = false
                    break
                else
                    ?? = ??*??
                    m += 1
                    flag_ng = true
                end
            end
        else # Take a Newton step
            dk = -jk'ffk
            ?? = 1.0
            m = 1
            while m <= mmax
                xt .= xk+??*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if 0.5*norm(ffn)^2 <=  0.5*norm(ffk)^2 + ??*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    ?? = ??*??
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

### Constrained Jacobian-free Newton-Krylov

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
    ??k = 1e-4
    ??  = 0.9
    t  = 1e-4
    ??  = 1e-4
    mmax  = 50

    g(x) = 0.5*norm(f(x))^2

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false
            dk, status = jacobian_free_gmres(f,xk,??k,krylovdim) # Using inexact restarted Arnoldi (could use inexact)

            ?? = 1.0
            m = 1
            while m <= mmax
                if norm(f(box_projection(xk+??*dk,lb,ub))) <= (1.0 - t*??*(1-??k))*norm(f(xk))
                    xn = box_projection(xk+??*dk,lb,ub)
                    ??k = min(1.0 - ??*(1.0 - ??k),1e-3)
                    flag_ng = false
                    break
                else
                    ?? = ??*??
                    m += 1
                    flag_ng = true
                end
            end
        else # Take a Newton step
            dk = - ForwardDiff.gradient(g,xk)
            ?? = 1.0
            m = 1
            while m <= mmax
                if g(box_projection(xk+??*dk,lb,ub)) <= g(xk) + ??*dk'*(box_projection(xk+??*dk,lb,ub) .- xk)
                    xn = box_projection(xk+??*dk,lb,ub)
                    flag_ng = false
                    break
                else
                    ?? = ??*??
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
    ??k = 1e-4
    ??  = 0.9
    t  = 1e-4
    ??  = 1e-4
    mmax  = 50

    function g(x)
        ftemp = Array{Number,1}(undef,length(x))
        f(ftemp,x)
        return 0.5*norm(ftemp)^2
    end

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false
            dk, status = jacobian_free_gmres_inplace(f,xk,??k,krylovdim) # Using inexact restarted Arnoldi (could use inexact)

            ?? = 1.0
            m = 1
            while m <= mmax
                f(ffk,xk)
                f(ffn,box_projection(xk+??*dk,lb,ub))
                if norm(ffn) <= (1.0 - t*??*(1-??k))*norm(ffk)
                    xn = box_projection(xk+??*dk,lb,ub)
                    ??k = min(1.0 - ??*(1.0 - ??k),1e-3)
                    flag_ng = false
                    break
                else
                    ?? = ??*??
                    m += 1
                    flag_ng = true
                end
            end
        else # Take a Newton step
            dk = -ForwardDiff.gradient(g,xk)
            ?? = 1.0
            m = 1
            while m <= mmax
                f(ffn,box_projection(xk+??*dk,lb,ub))
                if 0.5*norm(ffn)^2 <= 0.5*norm(ffk) + ??*dk'*(box_projection(xk+??*dk,lb,ub) .- xk)
                    xn = box_projection(xk+??*dk,lb,ub)
                    flag_ng = false
                    break
                else
                    ?? = ??*??
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
    elseif method == :dogleg_bmp
        if sparsejac == :no
            return constrained_dogleg_bmp_solver(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
        elseif sparsejac == :yes
            return constrained_dogleg_bmp_solver_sparse(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
        end
    elseif method == :nk
        if sparsejac == :no
            return constrained_newton_krylov(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
        elseif sparsejac == :yes
            return constrained_newton_krylov_sparse(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
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
    elseif method == :lm
        return constrained_levenberg_marquardt(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    elseif method == :lm_kyf
        return constrained_levenberg_marquardt_kyf(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    elseif method == :lm_ar
        return constrained_levenberg_marquardt_ar(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    elseif method == :dogleg
        return constrained_dogleg_solver(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    elseif method == :dogleg_bmp
        return constrained_dogleg_bmp_solver(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)
    elseif method == :nk
        return constrained_newton_krylov(f,j,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters,krylovdim=krylovdim)
    else
        error("Your chosen solution method is not supported")
    end

end
