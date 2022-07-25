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

function box_trust_projection(d,lb,ub,Δ)

    y = copy(x)

    for i in eachindex(x)
        y[i] = min(max(x[i],lb[i],-Δ),ub[i],Δ)
    end

    return y

end

function box_trust_projection!(x,lb,ub,Δ)

    for i in eachindex(x)
        x[i] = min(max(x[i],lb[i],-Δ),ub[i],Δ)
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

function constrained_newton(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_newton_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_newton_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_newton_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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

        if rank(jk) == n # Compute a Newton step
            xn .= xk - jk\f(xk)
        else # Compute a Levenberg-Marquardt step
            λ = 0.001*norm(f(xk))^2
            xn .= xk - (jk'jk + λ*I)\(jk'f(xk))
        end

        box_projection!(xn,lb,ub)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:nr,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
        if rank(jk) == n # Compute a Newton step
            xn .= xk - jk\ffk
        else # Compute a Levenberg-Marquardt step
            λ = 0.001*norm(ffk)^2
            xn .= xk - (jk'jk + λ*I)\(jk'ffk)
        end
  
        box_projection!(xn,lb,ub)
       
        f(ffk,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffk)
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end

    f(ffk,xn)
    results = SolverResults(:nr,x,xn,ffk,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_newton_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_newton_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_newton_outplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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

        if rank(jk) == n # Compute a Newton step
            xn .= xk - jk\f(xk)
        else # Compute a Levenberg-Marquardt step
            λ = 0.001*norm(f(xk))^2
            xn .= xk - (jk'jk + λ*I)\(jk'f(xk))
        end
  
        box_projection!(xn,lb,ub)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:nr,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton_inplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
        if rank(jk) == n # Compute a Newton step
            xn .= xk - jk\ffk
        else # Compute a Levenberg-Marquardt step
            λ = 0.001*norm(ffk)^2
            xn .= xk - (jk'jk + λ*I)\(jk'ffk)
        end
  
        box_projection!(xn,lb,ub)

        f(ffk,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffk)
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    f(ffk,xn)
    results = SolverResults(:nr,x,xn,ffk,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton_sparse(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_newton_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_newton_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_newton_sparse_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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

        if rank(jk) == n # Compute a Newton step
            xn .= xk - jk\f(xk)
        else # Compute a Levenberg-Marquardt step
            λ = 0.001*norm(f(xk))^2
            xn .= xk - (jk'jk + λ*I)\(jk'f(xk))
        end
  
        box_projection!(xn,lb,ub)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:nr,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton_sparse_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
        if rank(jk) == n # Compute a Newton step
            xn .= xk - jk\ffk
        else # Compute a Levenberg-Marquardt step
            λ = 0.001*norm(ffk)^2
            xn .= xk - (jk'jk + λ*I)\(jk'ffk)
        end
  
        box_projection!(xn,lb,ub)

        f(ffk,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffk)
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    f(ffk,xn)
    results = SolverResults(:nr,x,xn,ffk,lenx,lenf,iter,solution_trace)

    return results
  
end

### Constrained Levenberg-Marquardt-kyf

function constrained_levenberg_marquardt_kyf(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_levenberg_marquardt_kyf_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_levenberg_marquardt_kyf_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_levenberg_marquardt_kyf_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    β = 0.9
    γ = 0.99995
    σ = 10^(-4)
    ρ = 10^(-8)
    p = 2.1

    μk = max(0.5*10^(-8)*norm(f(xk))^2,1e-4)

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + μk*I)\(jk'f(xk))
        xn .= xk + dk
        box_projection!(xn,lb,ub)

        if norm(f(xn)) > γ*norm(f(xk))
            g .= jk'f(xk)
            if g'dk <= -ρ*norm(dk)^p
                α = 1.0
                while norm(f(xk+α*dk))^2 > norm(f(xk))^2 + 2*α*β*g'dk
                    α = β*α
                end
                xn .= xk + α*dk
            else
                α = 1.0
                while true
                    xt .= xk-α*g
                    box_projection!(xt,lb,ub)
                    if norm(f(xt))^2 <= norm(f(xk))^2 + 2*σ*g'*(xt-xk)
                        xn .=  xt
                        break
                    else
                        α = β*α
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

        μk = min(μk,norm(f(xk))^2)

    end
  
    results = SolverResults(:lm_kyf,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_kyf_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    β = 0.9
    γ = 0.99995
    σ = 10^(-4)
    ρ = 10^(-8)
    p = 2.1

    μk = max(0.5*10^(-8)*norm(ffk)^2,1e-4)

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,ffk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + μk*I)\(jk'ffk)
        xn .= xk + dk
        box_projection!(xn,lb,ub)

        f(ffn,xn)
        if norm(ffn) > γ*norm(ffk)
            g .= jk'ffk
            if g'dk <= -ρ*norm(dk)^p
                α = 1.0
                while true
                    f(ffn,xk+α*dk)
                    if norm(ffn)^2 > norm(ffk)^2 + 2*α*β*g'dk
                        α = β*α
                    else
                        break
                    end
                end
                xn .= xk + α*dk
            else
                α = 1.0
                while true
                    xt .= xk-α*g
                    box_projection!(xt,lb,ub)
                    f(ffn,xt)
                    if norm(ffn)^2 <= norm(ffk)^2 + 2*σ*g'*(xt-xk)
                        xn .=  xt
                        break
                    else
                        α = β*α
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

        μk = min(μk,norm(ffn)^2)

    end
  
    f(ffn,xn)
    results = SolverResults(:lm_kyf,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_kyf(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_levenberg_marquardt_kyf_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_levenberg_marquardt_kyf_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_levenberg_marquardt_kyf_outplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    β = 0.9
    γ = 0.99995
    σ = 10^(-4)
    ρ = 10^(-8)
    p = 2.1

    μk = max(0.5*10^(-8)*norm(f(xk))^2,1e-4)

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

        dk .= -(jk'jk + μk*I)\(jk'f(xk))
        xn .= xk + dk
        box_projection!(xn,lb,ub)

        if norm(f(xn)) > γ*norm(f(xk))
            g .= jk'f(xk)
            if g'dk <= -ρ*norm(dk)^p
                α = 1.0
                while norm(f(xk+α*dk))^2 > norm(f(xk))^2 + 2*α*β*g'dk
                    α = β*α
                end
                xn .= xk + α*dk
            else
                α = 1.0
                while true
                    xt .= xk-α*g
                    box_projection!(xt,lb,ub)
                    if norm(f(xt))^2 <= norm(f(xk))^2 + 2*σ*g'*(xt-xk)
                        xn .=  xt
                        break
                    else
                        α = β*α
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

        μk = min(μk,norm(f(xk))^2)

    end
  
    results = SolverResults(:lm_kyf,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_kyf_inplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    β = 0.9
    γ = 0.99995
    σ = 10^(-4)
    ρ = 10^(-8)
    p = 2.1

    μk = max(0.5*10^(-8)*norm(ffk)^2,1e-4)

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
        dk .= -(jk'jk + μk*I)\(jk'ffk)
        xn .= xk + dk
        box_projection!(xn,lb,ub)

        f(ffn,xn)
        if norm(ffn) > γ*norm(ffk)
            g .= jk'ffk
            if g'dk <= -ρ*norm(dk)^p
                α = 1.0
                while true
                    f(ffn,xk+α*dk)
                    if norm(ffn)^2 > norm(ffk)^2 + 2*α*β*g'dk
                        α = β*α
                    else
                        break
                    end
                end
                xn .= xk + α*dk
            else
                α = 1.0
                while true
                    xt .= xk-α*g
                    box_projection!(xt,lb,ub)
                    f(ffn,xt)
                    if norm(ffn)^2 <= norm(ffk)^2 + 2*σ*g'*(xt-xk)
                        xn .=  xt
                        break
                    else
                        α = β*α
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

        μk = min(μk,norm(ffn)^2)

    end
  
    f(ffn,xn)
    results = SolverResults(:lm_kyf,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_kyf_sparse(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_levenberg_marquardt_kyf_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_levenberg_marquardt_kyf_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_levenberg_marquardt_kyf_sparse_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    β = 0.9
    γ = 0.99995
    σ = 10^(-4)
    ρ = 10^(-8)
    p = 2.1

    μk = max(0.5*10^(-8)*norm(f(xk))^2,1e-4)

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + μk*I)\(jk'f(xk))
        xn .= xk + dk
        box_projection!(xn,lb,ub)

        if norm(f(xn)) > γ*norm(f(xk))
            g .= jk'f(xk)
            if g'dk <= -ρ*norm(dk)^p
                α = 1.0
                while norm(f(xk+α*dk))^2 > norm(f(xk))^2 + 2*α*β*g'dk
                    α = β*α
                end
                xn .= xk + α*dk
            else
                α = 1.0
                while true
                    xt .= xk-α*g
                    box_projection!(xt,lb,ub)
                    if norm(f(xt))^2 <= norm(f(xk))^2 + 2*σ*g'*(xt-xk)
                        xn .=  xt
                        break
                    else
                        α = β*α
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

        μk = min(μk,norm(f(xk))^2)

    end
  
    results = SolverResults(:lm_kyf,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_kyf_sparse_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    β = 0.9
    γ = 0.99995
    σ = 10^(-4)
    ρ = 10^(-8)
    p = 2.1

    μk = max(0.5*10^(-8)*norm(ffk)^2,1e-4)

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,ffk,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + μk*I)\(jk'ffk)
        xn .= xk + dk
        box_projection!(xn,lb,ub)

        f(ffn,xn)
        if norm(ffn) > γ*norm(ffk)
            g .= jk'ffk
            if g'dk <= -ρ*norm(dk)^p
                α = 1.0
                while true
                    f(ffn,xk+α*dk)
                    if norm(ffn)^2 > norm(ffk)^2 + 2*α*β*g'dk
                        α = β*α
                    else
                        break
                    end
                end
                xn .= xk + α*dk
            else
                α = 1.0
                while true
                    xt .= xk-α*g
                    box_projection!(xt,lb,ub)
                    f(ffn,xt)
                    if norm(ffn)^2 <= norm(ffk)^2 + 2*σ*g'*(xt-xk)
                        xn .=  xt
                        break
                    else
                        α = β*α
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

        μk = min(μk,norm(ffn)^2)

    end
  
    f(ffn,xn)
    results = SolverResults(:lm_kyf,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
   
end

### Constrained Levenberg-Marquardt-ar

function constrained_levenberg_marquardt_ar(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_levenberg_marquardt_ar_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_levenberg_marquardt_ar_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_levenberg_marquardt_ar_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    σ1 = 0.005
    σ2 = 0.005
    ρ  = 0.8
    r  = 0.5
    γ  = 10^(-16)
    μ  = 10^(-4)

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        λk = μ*norm(f(xk))^2

        d1k = -(jk'jk + λk*I)\(jk'f(xk))
        d2k = -(jk'jk + λk*I)\(jk'f(xk+d1k))
        d3k = -(jk'jk + λk*I)\(jk'f(xk+d1k+d2k))

        dk = d1k+d2k+d3k

        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk

        if norm(f(z)) <= ρ*norm(f(xk))
            α = 1.0
        else
            if f(xk)'jk*dk > -γ
                dk = d1k
                z .= xk+dk
                box_projection!(z,lb,ub)
                s .= z-xk
            end
            α = 1.0
            epsilon = 1/10
            while true
                if norm(f(xk+α*s))^2 > (1+epsilon)*norm(f(xk))^2 - σ1*α^2*norm(s)^2 - σ2*α^2*norm(f(xk))^2
                    α = r*α
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + α*s

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:lm_ar,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_levenberg_marquardt_ar_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    σ1 = 0.005
    σ2 = 0.005
    ρ  = 0.8
    r  = 0.5
    γ  = 10^(-16)
    μ  = 10^(-4)

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,ffk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        λk = μ*norm(ffk)^2

        d1k = -(jk'jk + λk*I)\(jk'ffk)
        f(ffn,xk+d1k)
        d2k = -(jk'jk + λk*I)\(jk'ffn)
        f(ffn,xk+d1k+d2k)
        d3k = -(jk'jk + λk*I)\(jk'ffn)

        dk = d1k+d2k+d3k

        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk

        f(ffn,z)
        if norm(ffn) <= ρ*norm(ffk)
            α = 1.0
        else
            if ffk'jk*dk > -γ
                dk = d1k
                z .= xk+dk
                box_projection!(z,lb,ub)
                s .= z-xk
            end
            α = 1.0
            epsilon = 1/10
            while true
                f(ffn,xk+α*s)
                if norm(ffn)^2 > (1+epsilon)*norm(ffk)^2 - σ1*α^2*norm(s)^2 - σ2*α^2*norm(ffk)^2
                    α = r*α
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + α*s

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:lm_ar,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_levenberg_marquardt_ar(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_levenberg_marquardt_ar_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_levenberg_marquardt_ar_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_levenberg_marquardt_ar_outplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    σ1 = 0.005
    σ2 = 0.005
    ρ  = 0.8
    r  = 0.5
    γ  = 10^(-16)
    μ  = 10^(-4)

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

        λk = μ*norm(f(xk))^2

        d1k = -(jk'jk + λk*I)\(jk'f(xk))
        d2k = -(jk'jk + λk*I)\(jk'f(xk+d1k))
        d3k = -(jk'jk + λk*I)\(jk'f(xk+d1k+d2k))

        dk = d1k+d2k+d3k

        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk

        if norm(f(z)) <= ρ*norm(f(xk))
            α = 1.0
        else
            if f(xk)'jk*dk > -γ
                dk = d1k
                z .= xk+dk
                box_projection!(z,lb,ub)
                s .= z-xk
            end
            α = 1.0
            epsilon = 1/10
            while true
                if norm(f(xk+α*s))^2 > (1+epsilon)*norm(f(xk))^2 - σ1*α^2*norm(s)^2 - σ2*α^2*norm(f(xk))^2
                    α = r*α
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + α*s

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:lm_ar,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_levenberg_marquardt_ar_inplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    σ1 = 0.005
    σ2 = 0.005
    ρ  = 0.8
    r  = 0.5
    γ  = 10^(-16)
    μ  = 10^(-4)

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
        λk = μ*norm(ffk)^2

        d1k = -(jk'jk + λk*I)\(jk'ffk)
        f(ffn,xk+d1k)
        d2k = -(jk'jk + λk*I)\(jk'ffn)
        f(ffn,xk+d1k+d2k)
        d3k = -(jk'jk + λk*I)\(jk'ffn)

        dk = d1k+d2k+d3k

        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk

        f(ffn,z)
        if norm(ffn) <= ρ*norm(ffk)
            α = 1.0
        else
            if ffk'jk*dk > -γ
                dk = d1k
                z .= xk+dk
                box_projection!(z,lb,ub)
                s .= z-xk
            end
            α = 1.0
            epsilon = 1/10
            while true
                f(ffn,xk+α*s)
                if norm(ffn)^2 > (1+epsilon)*norm(ffk)^2 - σ1*α^2*norm(s)^2 - σ2*α^2*norm(ffk)^2
                    α = r*α
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + α*s

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:lm_ar,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_levenberg_marquardt_ar_sparse(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_levenberg_marquardt_ar_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_levenberg_marquardt_ar_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_levenberg_marquardt_ar_sparse_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    σ1 = 0.005
    σ2 = 0.005
    ρ  = 0.8
    r  = 0.5
    γ  = 10^(-16)
    μ  = 10^(-4)

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        λk = μ*norm(f(xk))^2

        d1k = -(jk'jk + λk*I)\(jk'f(xk))
        d2k = -(jk'jk + λk*I)\(jk'f(xk+d1k))
        d3k = -(jk'jk + λk*I)\(jk'f(xk+d1k+d2k))

        dk = d1k+d2k+d3k

        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk

        if norm(f(z)) <= ρ*norm(f(xk))
            α = 1.0
        else
            if f(xk)'jk*dk > -γ
                dk = d1k
                z .= xk+dk
                box_projection!(z,lb,ub)
                s .= z-xk
            end
            α = 1.0
            epsilon = 1/10
            while true
                if norm(f(xk+α*s))^2 > (1+epsilon)*norm(f(xk))^2 - σ1*α^2*norm(s)^2 - σ2*α^2*norm(f(xk))^2
                    α = r*α
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + α*s

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:lm_ar,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_levenberg_marquardt_ar_sparse_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    σ1 = 0.005
    σ2 = 0.005
    ρ  = 0.8
    r  = 0.5
    γ  = 10^(-16)
    μ  = 10^(-4)

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,ffk,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        λk = μ*norm(ffk)^2

        d1k = -(jk'jk + λk*I)\(jk'ffk)
        f(ffn,xk+d1k)
        d2k = -(jk'jk + λk*I)\(jk'ffn)
        f(ffn,xk+d1k+d2k)
        d3k = -(jk'jk + λk*I)\(jk'ffn)

        dk = d1k+d2k+d3k

        z .= xk+dk
        box_projection!(z,lb,ub)
        s .= z-xk

        f(ffn,z)
        if norm(ffn) <= ρ*norm(ffk)
            α = 1.0
        else
            if ffk'jk*dk > -γ
                dk = d1k
                z .= xk+dk
                box_projection!(z,lb,ub)
                s .= z-xk
            end
            α = 1.0
            epsilon = 1/10
            while true
                f(ffn,xk+α*s)
                if norm(ffn)^2 > (1+epsilon)*norm(ffk)^2 - σ1*α^2*norm(s)^2 - σ2*α^2*norm(ffk)^2
                    α = r*α
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + α*s

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:lm_ar,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
  
end

### Constrained trust region

function constrained_trust_region(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_trust_region_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_trust_region_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_trust_region_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)

    lbk = similar(lb)
    ubk = similar(ub)

    xk = copy(x)
    xn = similar(x)

    p  = Array{T,1}(undef,n)
    jk = Array{T,2}(undef,n,n)

    Ql  = 0.0
    ηl  = 0.0
    ηll = 0.0

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver parameters
    M = 10
    Δ = 1.0
    Δmin = 1e-5
    Δmax = 1e5
    ρ1 = 0.1
    ρ2 = 0.9
    σ1 = 0.5
    σ2 = 2.0
    λ = 1/M
    η = 0.85
    μ = 0.001
    
    g(x) = 0.5*norm(f(x))^2

    g_store = Array{T,1}(undef,M) # Stores the M most recent values for g(x)

    mk = 1
    iter = 0

    while true

        g_store[1] = g(xk)

        # Compute the Jacobian of f(x) and check if it has non-finite elements

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        # Compute the principal direction

        if rank(jk) == n # Compute a Newton step
            p .= -jk\f(xk)
        else # Compute a Levenberg-Marquardt step
            λ = μ*norm(f(xk))^2
            p .= -(jk'jk + λ*I)\(jk'f(xk))
        end

        lbk .= lb-xk
        ubk .= ub-xk
        box_trust_projection!(p,lbk,ubk,Δ) # Constrains xn to be in the box and in the trust region

        if iter == 0

            gλ = g(xk)
            Qk = 1.0
            Ck = g(xk)

            rk = - (Ck - g(xk+p))/(f(xk)'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(f(xk))
            if rk < ρ1
                Δ *= σ1
            elseif rk >= ρ1 && rk < ρ2
                Δ = min(max(C,Δmin),Δmax)
                xn .= xk + p
            else
                Δ = min(max(σ2*C,Δmin),Δmax)
                xn .= xk + p
            end

            # Update η and Ql

            ηl, η = η, 0.5*η
            Ql = copy(Qk)

            # Shift elements in g_store down one position

            g_store[2] = g_store[1]

        else

            mk = min(mk+1,M)

            gλ = λ*g_store[1]
            for i = 2:mk
                gλ += λ*g_store[i]
            end
            if mk < M
                (g_max,i_max) = findmax(g_store[1:mk])
                gλ += (1.0 - λ*mk)*g_max
            end
            gλ = max(g_store[1],gλ)

            Qk = η^(iter)*Ql + 1.0

            Ck = (η^(iter)*Ql*gλ + g(xk))/Qk

            rk = - (Ck - g(xk+p))/(f(xk)'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(f(xk))
            if rk < ρ1
                Δ *= σ1
            elseif rk >= ρ1 && rk < ρ2
                Δ = min(max(C,Δmin),Δmax)
                xn .= xk + p
            else
                Δ = min(max(σ2*C,Δmin),Δmax)
                xn .= xk + p
            end

            # Update η and Ql

            ηll, ηl = ηl, η
            η = 0.5*(ηl + ηll)
            Ql = copy(Qk)

            # Shift elements in g_store down one position

            for i = mk:-1:2
                g_store[i] = g_store[i-1]
            end

        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol && Δ == Δmin) || (lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:tr,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_trust_region_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)

    lbk = similar(lb)
    ubk = similar(ub)

    xk = copy(x)
    xn = similar(x)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    p  = Array{T,1}(undef,n)
    jk = Array{T,2}(undef,n,n)

    Ql  = 0.0
    ηl  = 0.0
    ηll = 0.0

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver parameters
    M = 10
    Δ = 1.0
    Δmin = 1e-5
    Δmax = 1e5
    ρ1 = 0.1
    ρ2 = 0.9
    σ1 = 0.5
    σ2 = 2.0
    λ = 1/M
    η = 0.85
    μ = 0.001
    
    g_store = Array{T,1}(undef,M) # Stores the M most recent values for g(x)

    mk = 1
    iter = 0

    while true

        # Compute the Jacobian of f(x) and check if it has non-finite elements

        jk .= ForwardDiff.jacobian(f,ffk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        # Compute the principal direction

        if rank(jk) == n # Compute a Newton step
            p .= -jk\ffk
        else # Compute a Levenberg-Marquardt step
            λ = μ*norm(ffk)^2
            p .= -(jk'jk + λ*I)\(jk'ffk)
        end

        lbk .= lb-xk
        ubk .= ub-xk
        box_trust_projection!(p,lbk,ubk,Δ) # Constrains xn to be in the box and in the trust region

        g_store[1] = 0.5*norm(ffk)^2

        if iter == 0

            gλ = g_store[1]
            Qk = 1.0
            Ck = g_store[1]

            f(ffn,xk+p)
            rk = - (Ck - 0.5*norm(ffn)^2)/(ffk'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(ffk)
            if rk < ρ1
                Δ *= σ1
            elseif rk >= ρ1 && rk < ρ2
                Δ = min(max(C,Δmin),Δmax)
                xn .= xk + p
            else
                Δ = min(max(σ2*C,Δmin),Δmax)
                xn .= xk + p
            end

            # Update η and Ql

            ηl, η = η, 0.5*η
            Ql = copy(Qk)

            # Shift elements in g_store down one position

            g_store[2] = g_store[1]

        else

            mk = min(mk+1,M)

            gλ = λ*g_store[1]
            for i = 2:mk
                gλ += λ*g_store[i]
            end
            if mk < M
                (g_max,i_max) = findmax(g_store[1:mk])
                gλ += (1.0 - λ*mk)*g_max
            end
            gλ = max(g_store[1],gλ)

            Qk = η^(iter)*Ql + 1.0

            Ck = (η^(iter)*Ql*gλ + g_store[1])/Qk

            f(ffn,xk+p)
            rk = - (Ck - 0.5*norm(ffn)^2)/(ffk'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(ffk)
            if rk < ρ1
                Δ *= σ1
            elseif rk >= ρ1 && rk < ρ2
                Δ = min(max(C,Δmin),Δmax)
                xn .= xk + p
            else
                Δ = min(max(σ2*C,Δmin),Δmax)
                xn .= xk + p
            end

            # Update η and Ql

            ηll, ηl = ηl, η
            η = 0.5*(ηl + ηll)
            Ql = copy(Qk)

            # Shift elements in g_store down one position

            for i = mk:-1:2
                g_store[i] = g_store[i-1]
            end

        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol && Δ == Δmin) || (lenf <= ftol)
            break
        end
  
    end
  
    f(ffn,xn)
    results = SolverResults(:tr,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_trust_region(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_trust_region_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_trust_region_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_trust_region_outplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    j_inplace = !applicable(j,x)

    n = length(x)

    lbk = similar(lb)
    ubk = similar(ub)

    xk = copy(x)
    xn = similar(x)

    p  = Array{T,1}(undef,n)
    jk = Array{T,2}(undef,n,n)

    Ql  = 0.0
    ηl  = 0.0
    ηll = 0.0

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver parameters
    M = 10
    Δ = 1.0
    Δmin = 1e-5
    Δmax = 1e5
    ρ1 = 0.1
    ρ2 = 0.9
    σ1 = 0.5
    σ2 = 2.0
    λ = 1/M
    η = 0.85
    μ = 0.001
    
    g(x) = 0.5*norm(f(x))^2

    g_store = Array{T,1}(undef,M) # Stores the M most recent values for g(x)

    mk = 1
    iter = 0

    while true

        g_store[1] = g(xk)

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        # Compute the principal direction

        if rank(jk) == n # Compute a Newton step
            p .= -jk\f(xk)
        else # Compute a Levenberg-Marquardt step
            λ = μ*norm(f(xk))^2
            p .= -(jk'jk + λ*I)\(jk'f(xk))
        end

        lbk .= lb-xk
        ubk .= ub-xk
        box_trust_projection!(p,lbk,ubk,Δ) # Constrains xn to be in the box and in the trust region

        if iter == 0

            gλ = g(xk)
            Qk = 1.0
            Ck = g(xk)

            rk = - (Ck - g(xk+p))/(f(xk)'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(f(xk))
            if rk < ρ1
                Δ *= σ1
            elseif rk >= ρ1 && rk < ρ2
                Δ = min(max(C,Δmin),Δmax)
                xn .= xk + p
            else
                Δ = min(max(σ2*C,Δmin),Δmax)
                xn .= xk + p
            end

            # Update η and Ql

            ηl, η = η, 0.5*η
            Ql = copy(Qk)

            # Shift elements in g_store down one position

            g_store[2] = g_store[1]

        else

            mk = min(mk+1,M)

            gλ = λ*g_store[1]
            for i = 2:mk
                gλ += λ*g_store[i]
            end
            if mk < M
                (g_max,i_max) = findmax(g_store[1:mk])
                gλ += (1.0 - λ*mk)*g_max
            end
            gλ = max(g_store[1],gλ)

            Qk = η^(iter)*Ql + 1.0

            Ck = (η^(iter)*Ql*gλ + g(xk))/Qk

            rk = - (Ck - g(xk+p))/(f(xk)'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(f(xk))
            if rk < ρ1
                Δ *= σ1
            elseif rk >= ρ1 && rk < ρ2
                Δ = min(max(C,Δmin),Δmax)
                xn .= xk + p
            else
                Δ = min(max(σ2*C,Δmin),Δmax)
                xn .= xk + p
            end

            # Update η and Ql

            ηll, ηl = ηl, η
            η = 0.5*(ηl + ηll)
            Ql = copy(Qk)

            # Shift elements in g_store down one position

            for i = mk:-1:2
                g_store[i] = g_store[i-1]
            end

        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol && Δ == Δmin) || (lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:tr,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_trust_region_inplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    j_inplace = !applicable(j,x)

    n = length(x)

    lbk = similar(lb)
    ubk = similar(ub)

    xk = copy(x)
    xn = similar(x)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    p  = Array{T,1}(undef,n)
    jk = Array{T,2}(undef,n,n)

    Ql  = 0.0
    ηl  = 0.0
    ηll = 0.0

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver parameters
    M = 10
    Δ = 1.0
    Δmin = 1e-5
    Δmax = 1e5
    ρ1 = 0.1
    ρ2 = 0.9
    σ1 = 0.5
    σ2 = 2.0
    λ = 1/M
    η = 0.85
    μ = 0.001
    
    g_store = Array{T,1}(undef,M) # Stores the M most recent values for g(x)

    mk = 1
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

        # Compute the principal direction

        f(ffk,xk)  
        if rank(jk) == n # Compute a Newton step
            p .= -jk\ffk
        else # Compute a Levenberg-Marquardt step
            λ = μ*norm(ffk)^2
            p .= -(jk'jk + λ*I)\(jk'ffk)
        end

        lbk .= lb-xk
        ubk .= ub-xk
        box_trust_projection!(p,lbk,ubk,Δ) # Constrains xn to be in the box and in the trust region

        g_store[1] = 0.5*norm(ffk)^2

        if iter == 0

            gλ = g_store[1]
            Qk = 1.0
            Ck = g_store[1]

            f(ffn,xk+p)
            rk = - (Ck - 0.5*norm(ffn)^2)/(ffk'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(ffk)
            if rk < ρ1
                Δ *= σ1
            elseif rk >= ρ1 && rk < ρ2
                Δ = min(max(C,Δmin),Δmax)
                xn .= xk + p
            else
                Δ = min(max(σ2*C,Δmin),Δmax)
                xn .= xk + p
            end

            # Update η and Ql

            ηl, η = η, 0.5*η
            Ql = copy(Qk)

            # Shift elements in g_store down one position

            g_store[2] = g_store[1]

        else

            mk = min(mk+1,M)

            gλ = λ*g_store[1]
            for i = 2:mk
                gλ += λ*g_store[i]
            end
            if mk < M
                (g_max,i_max) = findmax(g_store[1:mk])
                gλ += (1.0 - λ*mk)*g_max
            end
            gλ = max(g_store[1],gλ)

            Qk = η^(iter)*Ql + 1.0

            Ck = (η^(iter)*Ql*gλ + g_store[1])/Qk

            f(ffn,xk+p)
            rk = - (Ck - 0.5*norm(ffn)^2)/(ffk'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(ffk)
            if rk < ρ1
                Δ *= σ1
            elseif rk >= ρ1 && rk < ρ2
                Δ = min(max(C,Δmin),Δmax)
                xn .= xk + p
            else
                Δ = min(max(σ2*C,Δmin),Δmax)
                xn .= xk + p
            end

            # Update η and Ql

            ηll, ηl = ηl, η
            η = 0.5*(ηl + ηll)
            Ql = copy(Qk)

            # Shift elements in g_store down one position

            for i = mk:-1:2
                g_store[i] = g_store[i-1]
            end

        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol && Δ == Δmin) || (lenf <= ftol)
            break
        end
  
    end
  
    f(ffn,xn)
    results = SolverResults(:tr,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_trust_region_sparse(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_trust_region_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_trust_region_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_trust_region_sparse_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)

    lbk = similar(lb)
    ubk = similar(ub)

    xk = copy(x)
    xn = similar(x)

    p  = Array{T,1}(undef,n)
    jk = sparse(Array{T,2}(undef,n,n))

    Ql  = 0.0
    ηl  = 0.0
    ηll = 0.0

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver parameters
    M = 10
    Δ = 1.0
    Δmin = 1e-5
    Δmax = 1e5
    ρ1 = 0.1
    ρ2 = 0.9
    σ1 = 0.5
    σ2 = 2.0
    λ = 1/M
    η = 0.85
    μ = 0.001
    
    g(x) = 0.5*norm(f(x))^2

    g_store = Array{T,1}(undef,M) # Stores the M most recent values for g(x)

    mk = 1
    iter = 0
    
    while true

        g_store[1] = g(xk)

        # Compute the Jacobian of f(x) and check if it has non-finite elements

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        # Compute the principal direction

        if rank(jk) == n # Compute a Newton step
            p .= -jk\f(xk)
        else # Compute a Levenberg-Marquardt step
            λ = μ*norm(f(xk))^2
            p .= -(jk'jk + λ*I)\(jk'f(xk))
        end

        lbk .= lb-xk
        ubk .= ub-xk
        box_trust_projection!(p,lbk,ubk,Δ) # Constrains xn to be in the box and in the trust region

        if iter == 0

            gλ = g(xk)
            Qk = 1.0
            Ck = g(xk)

            rk = - (Ck - g(xk+p))/(f(xk)'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(f(xk))
            if rk < ρ1
                Δ *= σ1
            elseif rk >= ρ1 && rk < ρ2
                Δ = min(max(C,Δmin),Δmax)
                xn .= xk + p
            else
                Δ = min(max(σ2*C,Δmin),Δmax)
                xn .= xk + p
            end

            # Update η and Ql

            ηl, η = η, 0.5*η
            Ql = copy(Qk)

            # Shift elements in g_store down one position

            g_store[2] = g_store[1]

        else

            mk = min(mk+1,M)

            gλ = λ*g_store[1]
            for i = 2:mk
                gλ += λ*g_store[i]
            end
            if mk < M
                (g_max,i_max) = findmax(g_store[1:mk])
                gλ += (1.0 - λ*mk)*g_max
            end
            gλ = max(g_store[1],gλ)

            Qk = η^(iter)*Ql + 1.0

            Ck = (η^(iter)*Ql*gλ + g(xk))/Qk

            rk = - (Ck - g(xk+p))/(f(xk)'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(f(xk))
            if rk < ρ1
                Δ *= σ1
            elseif rk >= ρ1 && rk < ρ2
                Δ = min(max(C,Δmin),Δmax)
                xn .= xk + p
            else
                Δ = min(max(σ2*C,Δmin),Δmax)
                xn .= xk + p
            end

            # Update η and Ql

            ηll, ηl = ηl, η
            η = 0.5*(ηl + ηll)
            Ql = copy(Qk)

            # Shift elements in g_store down one position

            for i = mk:-1:2
                g_store[i] = g_store[i-1]
            end

        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol && Δ == Δmin) || (lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:tr,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_trust_region_sparse_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)

    lbk = similar(lb)
    ubk = similar(ub)

    xk = copy(x)
    xn = similar(x)

    ffk = Array{T,1}(undef,n)
    ffn = Array{T,1}(undef,n)

    p  = Array{T,1}(undef,n)
    jk = sparse(Array{T,2}(undef,n,n))

    Ql  = 0.0
    ηl  = 0.0
    ηll = 0.0

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    f(ffk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,ffk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver parameters
    M = 10
    Δ = 1.0
    Δmin = 1e-5
    Δmax = 1e5
    ρ1 = 0.1
    ρ2 = 0.9
    σ1 = 0.5
    σ2 = 2.0
    λ = 1/M
    η = 0.85
    μ = 0.001
    
    g_store = Array{T,1}(undef,M) # Stores the M most recent values for g(x)

    mk = 1
    iter = 0

    while true

        # Compute the Jacobian of f(x) and check if it has non-finite elements

        jk .= sparse(ForwardDiff.jacobian(f,ffk,xk))
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        # Compute the principal direction

        if rank(jk) == n # Compute a Newton step
            p .= -jk\ffk
        else # Compute a Levenberg-Marquardt step
            λ = μ*norm(ffk)^2
            p .= -(jk'jk + λ*I)\(jk'ffk)
        end

        lbk .= lb-xk
        ubk .= ub-xk
        box_trust_projection!(p,lbk,ubk,Δ) # Constrains xn to be in the box and in the trust region

        g_store[1] = 0.5*norm(ffk)^2

        if iter == 0

            gλ = g_store[1]
            Qk = 1.0
            Ck = g_store[1]

            f(ffn,xk+p)
            rk = - (Ck - 0.5*norm(ffn)^2)/(ffk'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(ffk)
            if rk < ρ1
                Δ *= σ1
            elseif rk >= ρ1 && rk < ρ2
                Δ = min(max(C,Δmin),Δmax)
                xn .= xk + p
            else
                Δ = min(max(σ2*C,Δmin),Δmax)
                xn .= xk + p
            end

            # Update η and Ql

            ηl, η = η, 0.5*η
            Ql = copy(Qk)

            # Shift elements in g_store down one position

            g_store[2] = g_store[1]

        else

            mk = min(mk+1,M)

            gλ = λ*g_store[1]
            for i = 2:mk
                gλ += λ*g_store[i]
            end
            if mk < M
                (g_max,i_max) = findmax(g_store[1:mk])
                gλ += (1.0 - λ*mk)*g_max
            end
            gλ = max(g_store[1],gλ)

            Qk = η^(iter)*Ql + 1.0

            Ck = (η^(iter)*Ql*gλ + g_store[1])/Qk

            f(ffn,xk+p)
            rk = - (Ck - 0.5*norm(ffn)^2)/(ffk'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(ffk)
            if rk < ρ1
                Δ *= σ1
            elseif rk >= ρ1 && rk < ρ2
                Δ = min(max(C,Δmin),Δmax)
                xn .= xk + p
            else
                Δ = min(max(σ2*C,Δmin),Δmax)
                xn .= xk + p
            end

            # Update η and Ql

            ηll, ηl = ηl, η
            η = 0.5*(ηl + ηll)
            Ql = copy(Qk)

            # Shift elements in g_store down one position

            for i = mk:-1:2
                g_store[i] = g_store[i-1]
            end

        end

        f(ffn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,ffn)
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol && Δ == Δmin) || (lenf <= ftol)
            break
        end
  
    end
  
    f(ffn,xn)
    results = SolverResults(:tr,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
  
end


### Extra functions for constrained dogleg solver

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

function step_selection_outplace(f::Function,x::Array{T,1},Gk::AbstractArray{T,2},jk::AbstractArray{T,2},gk::Array{T,1},pkn::Array{T,1},lb::Array{T,1},ub::Array{T,1},δ::T,θ::T) where {T <: AbstractFloat}

    λk = Inf
    for i in eachindex(gk)
        if gk[i] != 0.0
            λk = min(max(((lb[i] - x[i])/gk[i]),((ub[i]-x[i])/gk[i])),λk)
        else
            λk = minimum((Inf,λk))
        end
    end

    taukprime = min(-(f(x)'*jk*gk)/norm(jk*gk)^2,δ/norm(Gk*gk))
    tauk = taukprime
    if !isequal(x+taukprime*gk, box_projection(x+taukprime*gk,lb,ub))
        tauk = θ*λk
    end

    pc = tauk*gk

    pc_pkn = pc-pkn
    
    if norm(-pc_pkn) < 1e-13 # "division by zero errors can otherwise occur"
        return (pc+pkn)/2
    else
        γhat = -(f(x)+jk*pc)'*jk*(-pc_pkn)/norm(jk*(-pc_pkn))^2
        if (pc'Gk^2*(pc_pkn))^2 - norm(Gk*(pc_pkn))^2*(norm(Gk*pc)^2-δ^2) >= 0.0
            r = ((pc'Gk^2*(pc_pkn))^2 - norm(Gk*(pc_pkn))^2*(norm(Gk*pc)^2-δ^2))^0.5
            γplus  = (pc'*Gk^2*(pc_pkn) + r)/norm(Gk*(pc_pkn))^2
            γminus = (pc'*Gk^2*(pc_pkn) - r)/norm(Gk*(pc_pkn))^2
        else
            γplus  = pc'*Gk^2*(pc_pkn)/norm(Gk*(pc_pkn))^2
            γminus = pc'*Gk^2*(pc_pkn)/norm(Gk*(pc_pkn))^2
        end

        if γhat > 1.0
            γtildaplus = Inf
            for i in eachindex(pkn)
                if (pkn[i]-pc[i]) != 0.0
                    γtildaplus = min(max(((lb[i] - x[i]- pc[i])/(pkn[i]-pc[i])),((ub[i] - x[i] - pc[i])/(pkn[i]-pc[i]))),γtildaplus)
                else
                    γtildaplus = min(Inf,γtildaplus)
                end
            end
            γ = min(γhat,γplus,θ*γtildaplus)
        elseif γhat < 0.0
            γtildaminus = -Inf
            for i in eachindex(pkn)
                if (-pkn[i]+pc[i]) != 0.0
                    γtildaminus = min(max(-((lb[i] - x[i]- pc[i])/(-(pkn[i]-pc[i]))),-((ub[i] - x[i] - pc[i])/(-(pkn[i]-pc[i])))),γtildaminus)
                else
                    γtildaminus = min(Inf,γtildaminus)
                end
            end
            γ = max(γhat,γminus,θ*γtildaminus)
        else
            γ = γhat
        end

        p = (1.0-γ)*pc + γ*pkn

        return p

    end

end

function step_selection_inplace(f::Function,x::Array{T,1},Gk::AbstractArray{T,2},jk::AbstractArray{T,2},gk::Array{T,1},pkn::Array{T,1},lb::Array{T,1},ub::Array{T,1},δ::T,θ::T) where {T <: AbstractFloat}

    ff = Array{T,1}(undef,length(x))

    λk = Inf
    for i in eachindex(gk)
        if gk[i] != 0.0
            λk = min(max(((lb[i] - x[i])/gk[i]),((ub[i]-x[i])/gk[i])),λk)
        else
            λk = minimum((Inf,λk))
        end
    end

    f(ff,x)
    taukprime = min(-(ff'*jk*gk)/norm(jk*gk)^2,δ/norm(Gk*gk))

    tauk = taukprime
    if !isequal(x+taukprime*gk, box_projection(x+taukprime*gk,lb,ub))
        tauk = θ*λk
    end

    pc = tauk*gk
    
    pc_pkn = pc-pkn
    
    if norm(-pc_pkn) < 1e-13 # "division by zero errors can otherwise occur"
        return (pc+pkn)/2
    else
        γhat = -(ff+jk*pc)'*jk*(-pc_pkn)/norm(jk*(-pc_pkn))^2
        if (pc'Gk^2*(pc_pkn))^2 - norm(Gk*(pc_pkn))^2*(norm(Gk*pc)^2-δ^2) >= 0.0
            r = ((pc'Gk^2*(pc_pkn))^2 - norm(Gk*(pc_pkn))^2*(norm(Gk*pc)^2-δ^2))^0.5
            γplus  = (pc'*Gk^2*(pc_pkn) + r)/norm(Gk*(pc_pkn))^2
            γminus = (pc'*Gk^2*(pc_pkn) - r)/norm(Gk*(pc_pkn))^2
        else
            γplus  = pc'*Gk^2*(pc_pkn)/norm(Gk*(pc_pkn))^2
            γminus = pc'*Gk^2*(pc_pkn)/norm(Gk*(pc_pkn))^2
        end

        if γhat > 1.0
            γtildaplus = Inf
            for i in eachindex(pkn)
                if (pkn[i]-pc[i]) != 0.0
                    γtildaplus = min(max(((lb[i] - x[i]- pc[i])/(pkn[i]-pc[i])),((ub[i] - x[i] - pc[i])/(pkn[i]-pc[i]))),γtildaplus)
                else
                    γtildaplus = min(Inf,γtildaplus)
                end
            end
            γ = min(γhat,γplus,θ*γtildaplus)
        elseif γhat < 0.0
            γtildaminus = -Inf
            for i in eachindex(pkn)
                if (-pkn[i]+pc[i]) != 0.0
                    γtildaminus = min(max(-((lb[i] - x[i]- pc[i])/(-(pkn[i]-pc[i]))),-((ub[i] - x[i] - pc[i])/(-(pkn[i]-pc[i])))),γtildaminus)
                else
                    γtildaminus = min(Inf,γtildaminus)
                end
            end
            γ = max(γhat,γminus,θ*γtildaminus)
        else
            γ = γhat
        end

        p = (1.0-γ)*pc + γ*pkn

        return p

    end

end

### Constrained dogleg solvers

function constrained_dogleg_solver(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_dogleg_solver_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_dogleg_solver_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_dogleg_solver_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    δ  = 1.0 
    γ1 = 2.0
    γ2 = 0.25
    θ  = 0.99995 
    β1 = 0.5 
    β2 = 0.9 

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
   
        α = max(θ,1.0-norm(f(xk)))
        pkn .= xk-jk\f(xk)
        box_projection!(pkn,lb,ub)
        pkn .= α*(pkn-xk)

        p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,δ,θ)
    
        while true
            ρ = (norm(f(xk)) - norm(f(xk+p))) / (norm(f(xk)) - norm(f(xk) + jk'*p))
            if ρ < β1 # linear approximation is poor fit so reduce the trust region
                δ = min(γ2*δ,0.5*norm(p))
                p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,δ,θ)
            elseif ρ > β2 # linear approximation is good fit so expand the trust region
                δ = max(δ,γ1*norm(p))
                p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,δ,θ)
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:dogleg,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_solver_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    δ  = 1.0 
    γ1 = 2.0
    γ2 = 0.25
    θ  = 0.99995 
    β1 = 0.5 
    β2 = 0.9 

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
   
        α = max(θ,1.0-norm(ffk))
        pkn .= xk-jk\ffk
        box_projection!(pkn,lb,ub)
        pkn .= α*(pkn-xk)

        p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,δ,θ)
    
        while true
            f(ffn,xk+p)
            ρ = (norm(ffk) - norm(ffn)) / (norm(ffk) - norm(ffk + jk'p))
            if ρ < β1 # linear approximation is poor fit so reduce the trust region
                δ = min(γ2*δ,0.5*norm(p))
                p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,δ,θ)
            elseif ρ > β2 # linear approximation is good fit so expand the trust region
                δ = max(δ,γ1*norm(p))
                p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,δ,θ)
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:dogleg,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_solver(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_dogleg_solver_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_dogleg_solver_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_dogleg_solver_outplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    δ  = 1.0 
    γ1 = 2.0
    γ2 = 0.25
    θ  = 0.99995 
    β1 = 0.5 
    β2 = 0.9 

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
   
        α = max(θ,1.0-norm(f(xk)))
        pkn .= xk-jk\f(xk)
        box_projection!(pkn,lb,ub)
        pkn .= α*(pkn-xk)

        p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,δ,θ)
    
        while true
            ρ = (norm(f(xk)) - norm(f(xk+p))) / (norm(f(xk)) - norm(f(xk) + jk'*p))
            if ρ < β1 # linear approximation is poor fit so reduce the trust region
                δ = min(γ2*δ,0.5*norm(p))
                p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,δ,θ)
            elseif ρ > β2 # linear approximation is good fit so expand the trust region
                δ = max(δ,γ1*norm(p))
                p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,δ,θ)
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:dogleg,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_solver_inplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    δ  = 1.0 
    γ1 = 2.0
    γ2 = 0.25
    θ  = 0.99995 
    β1 = 0.5 
    β2 = 0.9 

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
   
        α = max(θ,1.0-norm(ffk))
        pkn .= xk-jk\ffk
        box_projection!(pkn,lb,ub)
        pkn .= α*(pkn-xk)

        p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,δ,θ)
    
        while true
            f(ffn,xk+p)
            ρ = (norm(ffk) - norm(ffn)) / (norm(ffk) - norm(ffk + jk'p))
            if ρ < β1 # linear approximation is poor fit so reduce the trust region
                δ = min(γ2*δ,0.5*norm(p))
                p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,δ,θ)
            elseif ρ > β2 # linear approximation is good fit so expand the trust region
                δ = max(δ,γ1*norm(p))
                p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,δ,θ)
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:dogleg,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_solver_sparse(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_dogleg_solver_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_dogleg_solver_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_dogleg_solver_sparse_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    δ  = 1.0 
    γ1 = 2.0
    γ2 = 0.25
    θ  = 0.99995 
    β1 = 0.5 
    β2 = 0.9 

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
   
        α = max(θ,1.0-norm(f(xk)))
        pkn .= xk-jk\f(xk)
        box_projection!(pkn,lb,ub)
        pkn .= α*(pkn-xk)

        p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,δ,θ)
    
        while true
            ρ = (norm(f(xk)) - norm(f(xk+p))) / (norm(f(xk)) - norm(f(xk) + jk'*p))
            if ρ < β1 # linear approximation is poor fit so reduce the trust region
                δ = min(γ2*δ,0.5*norm(p))
                p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,δ,θ)
            elseif ρ > β2 # linear approximation is good fit so expand the trust region
                δ = max(δ,γ1*norm(p))
                p .= step_selection_outplace(f,xk,Gk,jk,gk,pkn,lb,ub,δ,θ)
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:dogleg,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_solver_sparse_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    δ  = 1.0 
    γ1 = 2.0
    γ2 = 0.25
    θ  = 0.99995 
    β1 = 0.5 
    β2 = 0.9 

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
   
        α = max(θ,1.0-norm(ffk))
        pkn .= xk-jk\ffk
        box_projection!(pkn,lb,ub)
        pkn .= α*(pkn-xk)

        p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,δ,θ)
    
        while true
            f(ffn,xk+p)
            ρ = (norm(ffk) - norm(ffn)) / (norm(ffk) - norm(ffk + jk'p))
            if ρ < β1 # linear approximation is poor fit so reduce the trust region
                δ = min(γ2*δ,0.5*norm(p))
                p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,δ,θ)
            elseif ρ > β2 # linear approximation is good fit so expand the trust region
                δ = max(δ,γ1*norm(p))
                p .= step_selection_inplace(f,xk,Gk,jk,gk,pkn,lb,ub,δ,θ)
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:dogleg,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end


### Constrained Newton-Krylov

function constrained_newton_krylov(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_newton_krylov_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
    else
        return constrained_newton_krylov_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
    end

end

function constrained_newton_krylov_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

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
    ηk = 1e-4#1e-4
    β  = 0.9
    t  = 1e-4
    σ  = 1e-4
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
            dk, status = gmres(jk,-f(xk),xk,ηk,krylovdim) # Inexact restarted

            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                box_projection!(xt,lb,ub)
                if norm(f(xt)) <= (1.0 - t*α*(1-ηk))*norm(f(xk))
                    xn .= xt
                    ηk = min(1.0 - α*(1.0 - ηk),1e-3)
                    flag_ng = false
                    break
                else
                    α = β*α
                    m += 1
                    flag_ng = true
                end
            end
        else # Take a Newton step
            dk = -jk'f(xk)
            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                box_projection!(xt,lb,ub)
                if g(xt) <= g(xk) + σ*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    α = β*α
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:nk,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

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
    ηk = 1e-4#1e-4
    β  = 0.9
    t  = 1e-4
    σ  = 1e-4
    mmax = 50

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false
            jk .= ForwardDiff.jacobian(f,ffk,xk)
            if !all(isfinite,jk)
                error("The jacobian has non-finite elements")
            end    
            dk, status = gmres(jk,-ffk,xk,ηk,krylovdim) # Inexact restarted

            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if norm(ffn) <= (1.0 - t*α*(1-ηk))*norm(ffk)
                    xn .= xt
                    ηk = min(1.0 - α*(1.0 - ηk),1e-3)
                    flag_ng = false
                    break
                else
                    α = β*α
                    m += 1
                    flag_ng = true
                end
            end
        else # Take a Newton step
            dk = -jk'ffk
            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if 0.5*norm(ffn)^2 <=  0.5*norm(ffk)^2 + σ*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    α = β*α
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:nk,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_newton_krylov_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
    else
        return constrained_newton_krylov_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
    end

end

function constrained_newton_krylov_outplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

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
    ηk = 1e-4#1e-4
    β  = 0.9
    t  = 1e-4
    σ  = 1e-4
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
    
            dk, status = gmres(jk,-f(xk),xk,ηk,krylovdim) # Inexact restarted

            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                box_projection!(xt,lb,ub)
                if norm(f(xt)) <= (1.0 - t*α*(1-ηk))*norm(f(xk))
                    xn .= xt
                    ηk = min(1.0 - α*(1.0 - ηk),1e-3)
                    flag_ng = false
                    break
                else
                    α = β*α
                    m += 1
                    flag_ng = true
                end
            end
        else # Take a Newton step
            dk = -jk'f(xk)
            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                box_projection!(xt,lb,ub)
                if g(xt) <= g(xk) + σ*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    α = β*α
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:nk,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov_inplace(f::Function,j::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

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
    ηk = 1e-4#1e-4
    β  = 0.9
    t  = 1e-4
    σ  = 1e-4
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
            dk, status = gmres(jk,-ffk,xk,ηk,krylovdim) # Inexact restarted

            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if norm(ffn) <= (1.0 - t*α*(1-ηk))*norm(ffk)
                    xn .= xt
                    ηk = min(1.0 - α*(1.0 - ηk),1e-3)
                    flag_ng = false
                    break
                else
                    α = β*α
                    m += 1
                    flag_ng = true
                end
            end
        else # Take a Newton step
            dk = -jk'ffk
            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if 0.5*norm(ffn)^2 <=  0.5*norm(ffk)^2 + σ*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    α = β*α
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:nk,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov_sparse(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_newton_krylov_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
    else
        return constrained_newton_krylov_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
    end

end

function constrained_newton_krylov_sparse_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

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
    ηk = 1e-4#1e-4
    β  = 0.9
    t  = 1e-4
    σ  = 1e-4
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
            dk, status = gmres(jk,-f(xk),xk,ηk,krylovdim) # Inexact restarted

            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                box_projection!(xt,lb,ub)
                if norm(f(xt)) <= (1.0 - t*α*(1-ηk))*norm(f(xk))
                    xn .= xt
                    ηk = min(1.0 - α*(1.0 - ηk),1e-3)
                    flag_ng = false
                    break
                else
                    α = β*α
                    m += 1
                    flag_ng = true
                end
            end
        else # Take a Newton step
            dk = -jk'f(xk)
            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                box_projection!(xt,lb,ub)
                if g(xt) <= g(xk) + σ*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    α = β*α
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:nk,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov_sparse_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

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
    ηk = 1e-4#1e-4
    β  = 0.9
    t  = 1e-4
    σ  = 1e-4
    mmax = 50

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false
            jk .= sparse(ForwardDiff.jacobian(f,ffk,xk))
            if !all(isfinite,nonzeros(jk))
                error("The jacobian has non-finite elements")
            end    
            dk, status = gmres(jk,-ffk,xk,ηk,krylovdim) # Inexact restarted

            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if norm(ffn) <= (1.0 - t*α*(1-ηk))*norm(ffk)
                    xn .= xt
                    ηk = min(1.0 - α*(1.0 - ηk),1e-3)
                    flag_ng = false
                    break
                else
                    α = β*α
                    m += 1
                    flag_ng = true
                end
            end
        else # Take a Newton step
            dk = -jk'ffk
            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                box_projection!(xt,lb,ub)
                f(ffn,xt)
                if 0.5*norm(ffn)^2 <=  0.5*norm(ffk)^2 + σ*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    α = β*α
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:nk,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

### Constrained Jacobian-free Newton-Krylov

function constrained_jacobian_free_newton_krylov(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

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
        return constrained_jacobian_free_newton_krylov_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
    else
        return constrained_jacobian_free_newton_krylov_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
    end

end

function constrained_jacobian_free_newton_krylov_outplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

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
    ηk = 1e-4
    β  = 0.9
    t  = 1e-4
    σ  = 1e-4
    mmax  = 50

    g(x) = 0.5*norm(f(x))^2

    flag_ng = false
    iter = 0
    while true

        if flag_ng == false
            dk, status = jacobian_free_gmres(f,xk,ηk,krylovdim) # Using inexact restarted Arnoldi (could use inexact)

            α = 1.0
            m = 1
            while m <= mmax
                if norm(f(box_projection(xk+α*dk,lb,ub))) <= (1.0 - t*α*(1-ηk))*norm(f(xk))
                    xn = box_projection(xk+α*dk,lb,ub)
                    ηk = min(1.0 - α*(1.0 - ηk),1e-3)
                    flag_ng = false
                    break
                else
                    α = β*α
                    m += 1
                    flag_ng = true
                end
            end
        else # Take a Newton step
            dk = - ForwardDiff.gradient(g,xk)
            α = 1.0
            m = 1
            while m <= mmax
                if g(box_projection(xk+α*dk,lb,ub)) <= g(xk) + σ*dk'*(box_projection(xk+α*dk,lb,ub) .- xk)
                    xn = box_projection(xk+α*dk,lb,ub)
                    flag_ng = false
                    break
                else
                    α = β*α
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:jfnk,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_jacobian_free_newton_krylov_inplace(f::Function,x::Array{T,1},lb::Array{T,1},ub::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

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
    ηk = 1e-4
    β  = 0.9
    t  = 1e-4
    σ  = 1e-4
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
            dk, status = jacobian_free_gmres_inplace(f,xk,ηk,krylovdim) # Using inexact restarted Arnoldi (could use inexact)

            α = 1.0
            m = 1
            while m <= mmax
                f(ffk,xk)
                f(ffn,box_projection(xk+α*dk,lb,ub))
                if norm(ffn) <= (1.0 - t*α*(1-ηk))*norm(ffk)
                    xn = box_projection(xk+α*dk,lb,ub)
                    ηk = min(1.0 - α*(1.0 - ηk),1e-3)
                    flag_ng = false
                    break
                else
                    α = β*α
                    m += 1
                    flag_ng = true
                end
            end
        else # Take a Newton step
            dk = -ForwardDiff.gradient(g,xk)
            α = 1.0
            m = 1
            while m <= mmax
                f(ffn,box_projection(xk+α*dk,lb,ub))
                if 0.5*norm(ffn)^2 <= 0.5*norm(ffk) + σ*dk'*(box_projection(xk+α*dk,lb,ub) .- xk)
                    xn = box_projection(xk+α*dk,lb,ub)
                    flag_ng = false
                    break
                else
                    α = β*α
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

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(ffn,xn)
    results = SolverResults(:jfnk,x,xn,ffn,lenx,lenf,iter,solution_trace)

    return results
 
end

function nlboxsolve(f::Function,x::Array{T,1},lb::Array{T,1} = [-Inf for _ in eachindex(x)],ub::Array{T,1}= [Inf for _ in eachindex(x)];xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,method::Symbol=:lm_ar,krylovdim::S=30,sparsejac::Symbol=:no) where {T <: AbstractFloat, S <: Integer}

    if method == :nr
        if sparsejac == :no
            return constrained_newton(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
        elseif sparsejac == :yes
            return constrained_newton_sparse(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
        end
    elseif method == :lm_kyf
        if sparsejac == :no
            return constrained_levenberg_marquardt_kyf(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
        elseif sparsejac == :yes
            return constrained_levenberg_marquardt_kyf_sparse(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
        end
    elseif method == :lm_ar
        if sparsejac == :no
            return constrained_levenberg_marquardt_ar(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
        elseif sparsejav == :yes
            return constrained_levenberg_marquardt_ar_sparse(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
        end
    elseif method == :tr
        if sparsejac == :no
            return constrained_trust_region(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
        elseif sparsejac == :yes
            return constrained_trust_region_sparse(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
        end
    elseif method == :dogleg
        if sparsejac == :no
            return constrained_dogleg_solver(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
        elseif sparsejac == :yes
            return constrained_dogleg_solver_sparse(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
        end
    elseif method == :nk
        if sparsejac == :no
            return constrained_newton_krylov(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
        elseif sparsejac == :yes
            return constrained_newton_krylov_sparse(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
        end
    elseif method == :jfnk
        return constrained_jacobian_free_newton_krylov(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        error("Your chosen solution method is not supported")
    end

end

function nlboxsolve(f::Function,j::Function,x::Array{T,1},lb::Array{T,1} = [-Inf for _ in eachindex(x)],ub::Array{T,1}= [Inf for _ in eachindex(x)];xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,method::Symbol=:lm_ar,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    if method == :nr
        return constrained_newton(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    elseif method == :lm_kyf
        return constrained_levenberg_marquardt_kyf(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    elseif method == :lm_ar
        return constrained_levenberg_marquardt_ar(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    elseif method == :tr
        return constrained_trust_region(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    elseif method == :dogleg
        return constrained_dogleg_solver(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    elseif method == :nk
        return constrained_newton_krylov(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
    else
        error("Your chosen solution method is not supported")
    end

end