function box_trust_projection(lb,ub,Δ)

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

function box_check(lb::AbstractVector{T},ub::AbstractVector{T}) where {T<:AbstractFloat}

    if length(lb) != length(ub)
        error(" 'lb' and 'ub' must have the same length")
    end

    return all(lb[i] < ub[i] for i in eachindex(lb))

end

################ Solvers ###################

### Constrained Newton-Raphson

function constrained_newton(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_newton_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_newton_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_newton_outplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    jk = Matrix{T}(undef,n,n)
    Fk = Vector{T}(undef,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    iter = 0
    while true

        Fk .= f(xk)

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        if rank(jk) == n # Compute a Newton step
            xn .= xk - jk\Fk
        else # Compute a Levenberg-Marquardt step
            λ = 0.001*norm(Fk)^2
            xn .= xk - (jk'jk + λ*I)\(jk'Fk)
        end

        xn .= clamp.(xn,lb,ub)

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

    results = BoxSolverResults(:nr,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton_inplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    jk = Matrix{T}(undef,n,n)

    Fk = Vector{T}(undef,n)

    lenx = zero(T)
    lenf = zero(T)
    
    # Initialize solution trace
    f(Fk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,Fk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        if rank(jk) == n # Compute a Newton step
            xn .= xk - jk\Fk
        else # Compute a Levenberg-Marquardt step
            λ = 0.001*norm(Fk)^2
            xn .= xk - (jk'jk + λ*I)\(jk'Fk)
        end
  
        xn .= clamp.(xn,lb,ub)
       
        f(Fk,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fk)
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end

    f(Fk,xn)
    results = BoxSolverResults(:nr,x,xn,Fk,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_newton_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_newton_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_newton_outplace(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    j_inplace = !applicable(j,x)

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    jk = Matrix{T}(undef,n,n)
    Fk = Vector{T}(undef,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    iter = 0
    while true

        Fk .= f(xk)

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        if rank(jk) == n # Compute a Newton step
            xn .= xk - jk\Fk
        else # Compute a Levenberg-Marquardt step
            λ = 0.001*norm(Fk)^2
            xn .= xk - (jk'jk + λ*I)\(jk'Fk)
        end
  
        xn .= clamp.(xn,lb,ub)

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
  
    results = BoxSolverResults(:nr,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton_inplace(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    j_inplace = !applicable(j,x)

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    jk = Matrix{T}(undef,n,n)
    Fk = Vector{T}(undef,n)

    lenx = zero(T)
    lenf = zero(T)
    
    # Initialize solution trace
    f(Fk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
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

        f(Fk,xk)
        if rank(jk) == n # Compute a Newton step
            xn .= xk - jk\Fk
        else # Compute a Levenberg-Marquardt step
            λ = 0.001*norm(Fk)^2
            xn .= xk - (jk'jk + λ*I)\(jk'Fk)
        end
  
        xn .= clamp.(xn,lb,ub)

        f(Fk,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fk)
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    f(Fk,xn)
    results = BoxSolverResults(:nr,x,xn,Fk,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton_sparse(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_newton_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_newton_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_newton_sparse_outplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    jk = sparse(Matrix{T}(undef,n,n))
    Fk = Vector{T}(undef,n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state = SolverState(0,NaN,maximum(abs,f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    iter = 0
    while true

        Fk .= f(xk)

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end

        if rank(jk) == n # Compute a Newton step
            xn .= xk - jk\Fk
        else # Compute a Levenberg-Marquardt step
            λ = 0.001*norm(Fk)^2
            xn .= xk - (jk'jk + λ*I)\(jk'Fk)
        end
  
        xn .= clamp.(xn,lb,ub)

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
  
    results = BoxSolverResults(:nr,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_newton_sparse_inplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    jk = sparse(Matrix{T}(undef,n,n))
    Fk = Vector{T}(undef,n)

    lenx = zero(T)
    lenf = zero(T)
    
    # Initialize solution trace
    f(Fk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,Fk,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        if rank(jk) == n # Compute a Newton step
            xn .= xk - jk\Fk
        else # Compute a Levenberg-Marquardt step
            λ = 0.001*norm(Fk)^2
            xn .= xk - (jk'jk + λ*I)\(jk'Fk)
        end
  
        xn .= clamp.(xn,lb,ub)

        f(Fk,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fk)
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    f(Fk,xn)
    results = BoxSolverResults(:nr,x,xn,Fk,lenx,lenf,iter,solution_trace)

    return results
  
end

### Constrained Levenberg-Marquardt-kyf

function constrained_levenberg_marquardt_kyf(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_levenberg_marquardt_kyf_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_levenberg_marquardt_kyf_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_levenberg_marquardt_kyf_outplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 3.12 from Kanzow, Yamashita, and Fukushima (2004) "Levenberg-Marquardt methods
    # with strong local convergence properties for solving nonlinear equations with convex constraints", Journal of 
    # Computational and Applied Mathematics, 172, pp. 375--397.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)
    dk = similar(x)
    g =  similar(x)
    jk = Matrix{T}(undef,n,n)
    Fk = Vector{T}(undef,n)

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

        Fk .= f(xk)

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + μk*I)\(jk'Fk)
        xn .= xk + dk
        xn .= clamp.(xn,lb,ub)
        dk .= xn - xk

        if norm(f(xn)) > γ*norm(Fk)
            g .= jk'Fk
            if g'dk <= -ρ*norm(dk)^p
                α = 1.0
                while norm(f(xk+α*dk))^2 > norm(Fk)^2 + 2*α*β*g'dk
                    α = β*α
                end
                xn .= xk + α*dk
                xn .= clamp.(xn,lb,ub)
            else
                α = 1.0
                while true
                    xt .= xk-α*g
                    xt .= clamp.(xt,lb,ub)
                    if norm(f(xt))^2 <= norm(Fk)^2 + 2*σ*g'*(xt-xk)
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
  
    results = BoxSolverResults(:lm_kyf,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_kyf_inplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 3.12 from Kanzow, Yamashita, and Fukushima (2004) "Levenberg-Marquardt methods
    # with strong local convergence properties for solving nonlinear equations with convex constraints", Journal of 
    # Computational and Applied Mathematics, 172, pp. 375--397.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)
    dk = similar(x)
    g =  similar(x)
    jk = Matrix{T}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    Fk = Vector{T}(undef,n)
    Fn = Vector{T}(undef,n)
 
    # Initialize solution trace
    f(Fk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    β = 0.9
    γ = 0.99995
    σ = 10^(-4)
    ρ = 10^(-8)
    p = 2.1

    μk = max(0.5*10^(-8)*norm(Fk)^2,1e-4)

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,Fk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + μk*I)\(jk'Fk)
        xn .= xk + dk
        xn .= clamp.(xn,lb,ub)
        dk .= xn - xk

        f(Fn,xn)
        if norm(Fn) > γ*norm(Fk)
            g .= jk'Fk
            if g'dk <= -ρ*norm(dk)^p
                α = 1.0
                while true
                    f(Fn,xk+α*dk)
                    if norm(Fn)^2 > norm(Fk)^2 + 2*α*β*g'dk
                        α = β*α
                    else
                        break
                    end
                end
                xn .= xk + α*dk
                xn .= clamp.(xn,lb,ub)
            else
                α = 1.0
                while true
                    xt .= xk-α*g
                    xt .= clamp.(xt,lb,ub)
                    f(Fn,xt)
                    if norm(Fn)^2 <= norm(Fk)^2 + 2*σ*g'*(xt-xk)
                        xn .=  xt
                        break
                    else
                        α = β*α
                    end
                end
            end
        end

        f(Fn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

        μk = min(μk,norm(Fn)^2)

    end
  
    f(Fn,xn)
    results = BoxSolverResults(:lm_kyf,x,xn,Fn,lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_kyf(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_levenberg_marquardt_kyf_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_levenberg_marquardt_kyf_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_levenberg_marquardt_kyf_outplace(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    jk = Matrix{T}(undef,n,n)
    Fk = Vector{T}(undef,n)

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

        Fk .= f(xk)

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        dk .= -(jk'jk + μk*I)\(jk'Fk)
        xn .= xk + dk
        xn .= clamp.(xn,lb,ub)
        dk .= xn - xk

        if norm(f(xn)) > γ*norm(Fk)
            g .= jk'Fk
            if g'dk <= -ρ*norm(dk)^p
                α = 1.0
                while norm(f(xk+α*dk))^2 > norm(Fk)^2 + 2*α*β*g'dk
                    α = β*α
                end
                xn .= xk + α*dk
                xn .= clamp.(xn,lb,ub)
            else
                α = 1.0
                while true
                    xt .= xk-α*g
                    xt .= clamp.(xt,lb,ub)
                    if norm(f(xt))^2 <= norm(Fk)^2 + 2*σ*g'*(xt-xk)
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
  
    results = BoxSolverResults(:lm_kyf,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_kyf_inplace(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    jk = Matrix{T}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    Fk = Vector{T}(undef,n)
    Fn = Vector{T}(undef,n)
 
    # Initialize solution trace
    f(Fk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    β = 0.9
    γ = 0.99995
    σ = 10^(-4)
    ρ = 10^(-8)
    p = 2.1

    μk = max(0.5*10^(-8)*norm(Fk)^2,1e-4)

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

        f(Fk,xk)
        dk .= -(jk'jk + μk*I)\(jk'Fk)
        xn .= xk + dk
        xn .= clamp.(xn,lb,ub)
        dk .= xn - xk

        f(Fn,xn)
        if norm(Fn) > γ*norm(Fk)
            g .= jk'Fk
            if g'dk <= -ρ*norm(dk)^p
                α = 1.0
                while true
                    f(Fn,xk+α*dk)
                    if norm(Fn)^2 > norm(Fk)^2 + 2*α*β*g'dk
                        α = β*α
                    else
                        break
                    end
                end
                xn .= xk + α*dk
                xn .= clamp.(xn,lb,ub)
            else
                α = 1.0
                while true
                    xt .= xk-α*g
                    xt .= clamp.(xt,lb,ub)
                    f(Fn,xt)
                    if norm(Fn)^2 <= norm(Fk)^2 + 2*σ*g'*(xt-xk)
                        xn .=  xt
                        break
                    else
                        α = β*α
                    end
                end
            end
        end

        f(Fn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

        μk = min(μk,norm(Fn)^2)

    end
  
    f(Fn,xn)
    results = BoxSolverResults(:lm_kyf,x,xn,Fn,lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_kyf_sparse(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_levenberg_marquardt_kyf_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_levenberg_marquardt_kyf_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_levenberg_marquardt_kyf_sparse_outplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 3.12 from Kanzow, Yamashita, and Fukushima (2004) "Levenberg-Marquardt methods
    # with strong local convergence properties for solving nonlinear equations with convex constraints", Journal of 
    # Computational and Applied Mathematics, 172, pp. 375--397.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)
    dk = similar(x)
    g =  similar(x)
    jk = sparse(Matrix{T}(undef,n,n))
    Fk = Vector{T}(undef,n)

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

        Fk .= f(xk)

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + μk*I)\(jk'Fk)
        xn .= xk + dk
        xn .= clamp.(xn,lb,ub)
        dk .= xn - xk

        if norm(f(xn)) > γ*norm(Fk)
            g .= jk'Fk
            if g'dk <= -ρ*norm(dk)^p
                α = 1.0
                while norm(f(xk+α*dk))^2 > norm(Fk)^2 + 2*α*β*g'dk
                    α = β*α
                end
                xn .= xk + α*dk
                xn .= clamp.(xn,lb,ub)
            else
                α = 1.0
                while true
                    xt .= xk-α*g
                    xt .= clamp.(xt,lb,ub)
                    if norm(f(xt))^2 <= norm(Fk)^2 + 2*σ*g'*(xt-xk)
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
  
    results = BoxSolverResults(:lm_kyf,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
   
end

function constrained_levenberg_marquardt_kyf_sparse_inplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 3.12 from Kanzow, Yamashita, and Fukushima (2004) "Levenberg-Marquardt methods
    # with strong local convergence properties for solving nonlinear equations with convex constraints", Journal of 
    # Computational and Applied Mathematics, 172, pp. 375--397.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)
    dk = similar(x)
    g =  similar(x)
    jk = sparse(Matrix{T}(undef,n,n))

    lenx = zero(T)
    lenf = zero(T)

    Fk = Vector{T}(undef,n)
    Fn = Vector{T}(undef,n)

    # Initialize solution trace
    f(Fk,x)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    β = 0.9
    γ = 0.99995
    σ = 10^(-4)
    ρ = 10^(-8)
    p = 2.1

    μk = max(0.5*10^(-8)*norm(Fk)^2,1e-4)

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,Fk,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        dk .= -(jk'jk + μk*I)\(jk'Fk)
        xn .= xk + dk
        xn .= clamp.(xn,lb,ub)
        dk .= xn - xk

        f(Fn,xn)
        if norm(Fn) > γ*norm(Fk)
            g .= jk'Fk
            if g'dk <= -ρ*norm(dk)^p
                α = 1.0
                while true
                    f(Fn,xk+α*dk)
                    if norm(Fn)^2 > norm(Fk)^2 + 2*α*β*g'dk
                        α = β*α
                    else
                        break
                    end
                end
                xn .= xk + α*dk
                xn .= clamp.(xn,lb,ub)
            else
                α = 1.0
                while true
                    xt .= xk-α*g
                    xt .= clamp.(xt,lb,ub)
                    f(Fn,xt)
                    if norm(Fn)^2 <= norm(Fk)^2 + 2*σ*g'*(xt-xk)
                        xn .=  xt
                        break
                    else
                        α = β*α
                    end
                end
            end
        end

        f(Fn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

        μk = min(μk,norm(Fn)^2)

    end
  
    f(Fn,xn)
    results = BoxSolverResults(:lm_kyf,x,xn,Fn,lenx,lenf,iter,solution_trace)

    return results
   
end

### Constrained Levenberg-Marquardt-ar

function constrained_levenberg_marquardt_ar(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_levenberg_marquardt_ar_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_levenberg_marquardt_ar_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_levenberg_marquardt_ar_outplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 2.1 from Amini and Rostami (2016), "Three-steps modified Levenberg-Marquardt 
    # method with a new line search for systems of nonlinear equations", Journal of Computational and Applied Mathematics, 
    # 300, pp. 30--42.

    # Modified to allow for box-constraints by Richard Dennis.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    z  = similar(x)
    s  = similar(x)
    jk = Matrix{T}(undef,n,n)
    Fk = Vector{T}(undef,n)

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

        Fk .= f(xk)

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        λk = μ*norm(Fk)^2

        d1k = -(jk'jk + λk*I)\(jk'Fk)
        d2k = -(jk'jk + λk*I)\(jk'f(xk+d1k))
        d3k = -(jk'jk + λk*I)\(jk'f(xk+d1k+d2k))

        dk = d1k+d2k+d3k

        z .= xk+dk
        z .= clamp.(z,lb,ub)
        s .= z-xk

        if norm(f(z)) <= ρ*norm(Fk)
            α = 1.0
        else
            if Fk'jk*dk > -γ
                dk = d1k
                z .= xk+dk
                z .= clamp.(z,lb,ub)
                s .= z-xk
            end
            α = 1.0
            epsilon = 1/10
            while true
                if norm(f(xk+α*s))^2 > (1+epsilon)*norm(Fk)^2 - σ1*α^2*norm(s)^2 - σ2*α^2*norm(Fk)^2
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
  
    results = BoxSolverResults(:lm_ar,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_levenberg_marquardt_ar_inplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 2.1 from Amini and Rostami (2016), "Three-steps modified Levenberg-Marquardt 
    # method with a new line search for systems of nonlinear equations", Journal of Computational and Applied Mathematics, 
    # 300, pp. 30--42.

    # Modified to allow for box-constraints by Richard Dennis.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    z  = similar(x)
    s  = similar(x)
    jk = Matrix{T}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    Fk = Vector{T}(undef,n)
    Fn = Vector{T}(undef,n)

    # Initialize solution trace
    f(Fk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
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

        jk .= ForwardDiff.jacobian(f,Fk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end
        λk = μ*norm(Fk)^2

        d1k = -(jk'jk + λk*I)\(jk'Fk)
        f(Fn,xk+d1k)
        d2k = -(jk'jk + λk*I)\(jk'Fn)
        f(Fn,xk+d1k+d2k)
        d3k = -(jk'jk + λk*I)\(jk'Fn)

        dk = d1k+d2k+d3k

        z .= xk+dk
        z .= clamp.(z,lb,ub)
        s .= z-xk

        f(Fn,z)
        if norm(Fn) <= ρ*norm(Fk)
            α = 1.0
        else
            if Fk'jk*dk > -γ
                dk = d1k
                z .= xk+dk
                z .= clamp.(z,lb,ub)
                s .= z-xk
            end
            α = 1.0
            epsilon = 1/10
            while true
                f(Fn,xk+α*s)
                if norm(Fn)^2 > (1+epsilon)*norm(Fk)^2 - σ1*α^2*norm(s)^2 - σ2*α^2*norm(Fk)^2
                    α = r*α
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + α*s

        f(Fn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(Fn,xn)
    results = BoxSolverResults(:lm_ar,x,xn,Fn,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_levenberg_marquardt_ar(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_levenberg_marquardt_ar_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_levenberg_marquardt_ar_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_levenberg_marquardt_ar_outplace(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    jk = Matrix{T}(undef,n,n)
    Fk = Vector{T}(undef,n)

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

        Fk .= f(xk)

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        λk = μ*norm(Fk)^2

        d1k = -(jk'jk + λk*I)\(jk'Fk)
        d2k = -(jk'jk + λk*I)\(jk'f(xk+d1k))
        d3k = -(jk'jk + λk*I)\(jk'f(xk+d1k+d2k))

        dk = d1k+d2k+d3k

        z .= xk+dk
        z .= clamp.(z,lb,ub)
        s .= z-xk

        if norm(f(z)) <= ρ*norm(Fk)
            α = 1.0
        else
            if Fk'jk*dk > -γ
                dk = d1k
                z .= xk+dk
                z .= clamp.(z,lb,ub)
                s .= z-xk
            end
            α = 1.0
            epsilon = 1/10
            while true
                if norm(f(xk+α*s))^2 > (1+epsilon)*norm(Fk)^2 - σ1*α^2*norm(s)^2 - σ2*α^2*norm(Fk)^2
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
  
    results = BoxSolverResults(:lm_ar,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_levenberg_marquardt_ar_inplace(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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
    jk = Matrix{T}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    Fk = Vector{T}(undef,n)
    Fn = Vector{T}(undef,n)

    # Initialize solution trace
    f(Fk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
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

        f(Fk,xk)
        λk = μ*norm(Fk)^2

        d1k = -(jk'jk + λk*I)\(jk'Fk)
        f(Fn,xk+d1k)
        d2k = -(jk'jk + λk*I)\(jk'Fn)
        f(Fn,xk+d1k+d2k)
        d3k = -(jk'jk + λk*I)\(jk'Fn)

        dk = d1k+d2k+d3k

        z .= xk+dk
        z .= clamp.(z,lb,ub)
        s .= z-xk

        f(Fn,z)
        if norm(Fn) <= ρ*norm(Fk)
            α = 1.0
        else
            if Fk'jk*dk > -γ
                dk = d1k
                z .= xk+dk
                z .= clamp.(z,lb,ub)
                s .= z-xk
            end
            α = 1.0
            epsilon = 1/10
            while true
                f(Fn,xk+α*s)
                if norm(Fn)^2 > (1+epsilon)*norm(Fk)^2 - σ1*α^2*norm(s)^2 - σ2*α^2*norm(Fk)^2
                    α = r*α
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + α*s

        f(Fn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(Fn,xn)
    results = BoxSolverResults(:lm_ar,x,xn,Fn,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_levenberg_marquardt_ar_sparse(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_levenberg_marquardt_ar_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_levenberg_marquardt_ar_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_levenberg_marquardt_ar_sparse_outplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 2.1 from Amini and Rostami (2016), "Three-steps modified Levenberg-Marquardt 
    # method with a new line search for systems of nonlinear equations", Journal of Computational and Applied Mathematics, 
    # 300, pp. 30--42.

    # Modified to allow for box-constraints by Richard Dennis.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    z  = similar(x)
    s  = similar(x)
    jk = sparse(Matrix{T}(undef,n,n))
    Fk = Vector{T}(undef,n)

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

        Fk .= f(xk)

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        λk = μ*norm(Fk)^2

        d1k = -(jk'jk + λk*I)\(jk'Fk)
        d2k = -(jk'jk + λk*I)\(jk'f(xk+d1k))
        d3k = -(jk'jk + λk*I)\(jk'f(xk+d1k+d2k))

        dk = d1k+d2k+d3k

        z .= xk+dk
        z .= clamp.(z,lb,ub)
        s .= z-xk

        if norm(f(z)) <= ρ*norm(Fk)
            α = 1.0
        else
            if Fk'jk*dk > -γ
                dk = d1k
                z .= xk+dk
                z .= clamp.(z,lb,ub)
                s .= z-xk
            end
            α = 1.0
            epsilon = 1/10
            while true
                if norm(f(xk+α*s))^2 > (1+epsilon)*norm(Fk)^2 - σ1*α^2*norm(s)^2 - σ2*α^2*norm(Fk)^2
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
  
    results = BoxSolverResults(:lm_ar,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_levenberg_marquardt_ar_sparse_inplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 2.1 from Amini and Rostami (2016), "Three-steps modified Levenberg-Marquardt 
    # method with a new line search for systems of nonlinear equations", Journal of Computational and Applied Mathematics, 
    # 300, pp. 30--42.

    # Modified to allow for box-constraints by Richard Dennis.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    z  = similar(x)
    s  = similar(x)
    jk = sparse(Matrix{T}(undef,n,n))

    lenx = zero(T)
    lenf = zero(T)

    Fk = Vector{T}(undef,n)
    Fn = Vector{T}(undef,n)

    # Initialize solution trace
    f(Fk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
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

        jk .= sparse(ForwardDiff.jacobian(f,Fk,xk))
        if !all(isfinite,nonzeros(jk))
            error("The jacobian has non-finite elements")
        end
        λk = μ*norm(Fk)^2

        d1k = -(jk'jk + λk*I)\(jk'Fk)
        f(Fn,xk+d1k)
        d2k = -(jk'jk + λk*I)\(jk'Fn)
        f(Fn,xk+d1k+d2k)
        d3k = -(jk'jk + λk*I)\(jk'Fn)

        dk = d1k+d2k+d3k

        z .= xk+dk
        z .= clamp.(z,lb,ub)
        s .= z-xk

        f(Fn,z)
        if norm(Fn) <= ρ*norm(Fk)
            α = 1.0
        else
            if Fk'jk*dk > -γ
                dk = d1k
                z .= xk+dk
                z .= clamp.(z,lb,ub)
                s .= z-xk
            end
            α = 1.0
            epsilon = 1/10
            while true
                f(Fn,xk+α*s)
                if norm(Fn)^2 > (1+epsilon)*norm(Fk)^2 - σ1*α^2*norm(s)^2 - σ2*α^2*norm(Fk)^2
                    α = r*α
                    epsilon = r*epsilon
                else
                    break
                end
            end
        end

        xn .= xk + α*s

        f(Fn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fn)
  
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(Fn,xn)
    results = BoxSolverResults(:lm_ar,x,xn,Fn,lenx,lenf,iter,solution_trace)

    return results
  
end

### Constrained trust region

function constrained_trust_region(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_trust_region_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_trust_region_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_trust_region_outplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)

    lbk = similar(lb)
    ubk = similar(ub)

    xk = copy(x)
    xn = similar(x)

    p  = Vector{T}(undef,n)
    jk = Matrix{T}(undef,n,n)
    Fk = Vector{T}(undef,n)

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

    g_store = Vector{T}(undef,M) # Stores the M most recent values for g(x)

    mk = 1
    iter = 0

    while true

        Fk .= f(xk)

        g_store[1] = g(xk)

        # Compute the Jacobian of f(x) and check if it has non-finite elements

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        # Compute the principal direction

        if rank(jk) == n # Compute a Newton step
            p .= -jk\Fk
        else # Compute a Levenberg-Marquardt step
            λ = μ*norm(Fk)^2
            p .= -(jk'jk + λ*I)\(jk'Fk)
        end

        lbk .= lb-xk
        ubk .= ub-xk
        box_trust_projection!(p,lbk,ubk,Δ) # Constrains xn to be in the box and in the trust region

        if iter == 0

            gλ = g(xk)
            Qk = 1.0
            Ck = g(xk)

            rk = - (Ck - g(xk+p))/(Fk'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(Fk)
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

            rk = - (Ck - g(xk+p))/(Fk'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(Fk)
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
  
    results = BoxSolverResults(:tr,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_trust_region_inplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)

    lbk = similar(lb)
    ubk = similar(ub)

    xk = copy(x)
    xn = similar(x)

    Fk = Vector{T}(undef,n)
    Fn = Vector{T}(undef,n)

    p  = Vector{T}(undef,n)
    jk = Matrix{T}(undef,n,n)

    Ql  = 0.0
    ηl  = 0.0
    ηll = 0.0

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    f(Fk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
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
    
    g_store = Vector{T}(undef,M) # Stores the M most recent values for g(x)

    mk = 1
    iter = 0

    while true

        # Compute the Jacobian of f(x) and check if it has non-finite elements

        jk .= ForwardDiff.jacobian(f,Fk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        # Compute the principal direction

        if rank(jk) == n # Compute a Newton step
            p .= -jk\Fk
        else # Compute a Levenberg-Marquardt step
            λ = μ*norm(Fk)^2
            p .= -(jk'jk + λ*I)\(jk'Fk)
        end

        lbk .= lb-xk
        ubk .= ub-xk
        box_trust_projection!(p,lbk,ubk,Δ) # Constrains xn to be in the box and in the trust region

        g_store[1] = 0.5*norm(Fk)^2

        if iter == 0

            gλ = g_store[1]
            Qk = 1.0
            Ck = g_store[1]

            f(Fn,xk+p)
            rk = - (Ck - 0.5*norm(Fn)^2)/(Fk'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(Fk)
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

            f(Fn,xk+p)
            rk = - (Ck - 0.5*norm(Fn)^2)/(Fk'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(Fk)
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

        f(Fn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fn)
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol && Δ == Δmin) || (lenf <= ftol)
            break
        end
  
    end
  
    f(Fn,xn)
    results = BoxSolverResults(:tr,x,xn,Fn,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_trust_region(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_trust_region_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_trust_region_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_trust_region_outplace(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    j_inplace = !applicable(j,x)

    n = length(x)

    lbk = similar(lb)
    ubk = similar(ub)

    xk = copy(x)
    xn = similar(x)

    p  = Vector{T}(undef,n)
    jk = Matrix{T}(undef,n,n)
    Fk = Vector{T}(undef,n)

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

    g_store = Vector{T}(undef,M) # Stores the M most recent values for g(x)

    mk = 1
    iter = 0

    while true

        Fk .= f(xk)

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
            p .= -jk\Fk
        else # Compute a Levenberg-Marquardt step
            λ = μ*norm(Fk)^2
            p .= -(jk'jk + λ*I)\(jk'Fk)
        end

        lbk .= lb-xk
        ubk .= ub-xk
        box_trust_projection!(p,lbk,ubk,Δ) # Constrains xn to be in the box and in the trust region

        if iter == 0

            gλ = g(xk)
            Qk = 1.0
            Ck = g(xk)

            rk = - (Ck - g(xk+p))/(Fk'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(Fk)
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

            rk = - (Ck - g(xk+p))/(Fk'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(Fk)
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
  
    results = BoxSolverResults(:tr,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_trust_region_inplace(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    j_inplace = !applicable(j,x)

    n = length(x)

    lbk = similar(lb)
    ubk = similar(ub)

    xk = copy(x)
    xn = similar(x)

    Fk = Vector{T}(undef,n)
    Fn = Vector{T}(undef,n)

    p  = Vector{T}(undef,n)
    jk = Matrix{T}(undef,n,n)

    Ql  = 0.0
    ηl  = 0.0
    ηll = 0.0

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    f(Fk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
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
    
    g_store = Vector{T}(undef,M) # Stores the M most recent values for g(x)

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

        f(Fk,xk)  
        if rank(jk) == n # Compute a Newton step
            p .= -jk\Fk
        else # Compute a Levenberg-Marquardt step
            λ = μ*norm(Fk)^2
            p .= -(jk'jk + λ*I)\(jk'Fk)
        end

        lbk .= lb-xk
        ubk .= ub-xk
        box_trust_projection!(p,lbk,ubk,Δ) # Constrains xn to be in the box and in the trust region

        g_store[1] = 0.5*norm(Fk)^2

        if iter == 0

            gλ = g_store[1]
            Qk = 1.0
            Ck = g_store[1]

            f(Fn,xk+p)
            rk = - (Ck - 0.5*norm(Fn)^2)/(Fk'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(Fk)
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

            f(Fn,xk+p)
            rk = - (Ck - 0.5*norm(Fn)^2)/(Fk'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(Fk)
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

        f(Fn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fn)
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol && Δ == Δmin) || (lenf <= ftol)
            break
        end
  
    end
  
    f(Fn,xn)
    results = BoxSolverResults(:tr,x,xn,Fn,lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_trust_region_sparse(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_trust_region_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_trust_region_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_trust_region_sparse_outplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)

    lbk = similar(lb)
    ubk = similar(ub)

    xk = copy(x)
    xn = similar(x)

    p  = Vector{T}(undef,n)
    jk = sparse(Matrix{T}(undef,n,n))
    Fk = Vector{T}(undef,n)

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

    g_store = Vector{T}(undef,M) # Stores the M most recent values for g(x)

    mk = 1
    iter = 0
    
    while true

        Fk .= f(xk)

        g_store[1] = g(xk)

        # Compute the Jacobian of f(x) and check if it has non-finite elements

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        # Compute the principal direction

        if rank(jk) == n # Compute a Newton step
            p .= -jk\Fk
        else # Compute a Levenberg-Marquardt step
            λ = μ*norm(Fk)^2
            p .= -(jk'jk + λ*I)\(jk'Fk)
        end

        lbk .= lb-xk
        ubk .= ub-xk
        box_trust_projection!(p,lbk,ubk,Δ) # Constrains xn to be in the box and in the trust region

        if iter == 0

            gλ = g(xk)
            Qk = 1.0
            Ck = g(xk)

            rk = - (Ck - g(xk+p))/(Fk'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(Fk)
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

            rk = - (Ck - g(xk+p))/(Fk'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(Fk)
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
  
    results = BoxSolverResults(:tr,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
  
end

function constrained_trust_region_sparse_inplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    n = length(x)

    lbk = similar(lb)
    ubk = similar(ub)

    xk = copy(x)
    xn = similar(x)

    Fk = Vector{T}(undef,n)
    Fn = Vector{T}(undef,n)

    p  = Vector{T}(undef,n)
    jk = sparse(Matrix{T}(undef,n,n))

    Ql  = 0.0
    ηl  = 0.0
    ηll = 0.0

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    f(Fk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
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
    
    g_store = Vector{T}(undef,M) # Stores the M most recent values for g(x)

    mk = 1
    iter = 0

    while true

        # Compute the Jacobian of f(x) and check if it has non-finite elements

        jk .= sparse(ForwardDiff.jacobian(f,Fk,xk))
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        # Compute the principal direction

        if rank(jk) == n # Compute a Newton step
            p .= -jk\Fk
        else # Compute a Levenberg-Marquardt step
            λ = μ*norm(Fk)^2
            p .= -(jk'jk + λ*I)\(jk'Fk)
        end

        lbk .= lb-xk
        ubk .= ub-xk
        box_trust_projection!(p,lbk,ubk,Δ) # Constrains xn to be in the box and in the trust region

        g_store[1] = 0.5*norm(Fk)^2

        if iter == 0

            gλ = g_store[1]
            Qk = 1.0
            Ck = g_store[1]

            f(Fn,xk+p)
            rk = - (Ck - 0.5*norm(Fn)^2)/(Fk'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(Fk)
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

            f(Fn,xk+p)
            rk = - (Ck - 0.5*norm(Fn)^2)/(Fk'jk*p + 0.5*p'jk'jk*p)

            # Update the trust region

            C = norm(Fk)
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

        f(Fn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fn)
    
        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol && Δ == Δmin) || (lenf <= ftol)
            break
        end
  
    end
  
    f(Fn,xn)
    results = BoxSolverResults(:tr,x,xn,Fn,lenx,lenf,iter,solution_trace)

    return results
  
end

function coleman_li_scale(gradf::AbstractVector{T},x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T}) where {T <: AbstractFloat}

    df = similar(gradf)

    @inbounds for i in eachindex(df)
        gi = gradf[i]
        if gi < 0.0 && ub[i] < Inf
            df[i] = max(ub[i] - x[i],eps(T))
        elseif gi > 0.0 && lb[i] > -Inf
            df[i] = max(x[i] - lb[i],eps(T))
        elseif gi == 0.0
            dl = lb[i] > -Inf ? x[i] - lb[i] : Inf
            du = ub[i] < Inf  ? ub[i] - x[i] : Inf
            df[i] = max(isinf(min(dl, du)) ? 1.0 : min(dl, du),eps(T))
        else
            df[i] = 1.0
        end
    end

    return df
    
end

function compute_gammatildaplus(pc_pkn::AbstractVector{T},x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T},pc::AbstractVector{T}) where {T <: AbstractFloat}

    γtildaplus = Inf
    for i in eachindex(pc_pkn)
        g = pc_pkn[i]
        if g != 0.0
            tl = lb[i] > -Inf ? -(lb[i] - x[i] - pc[i])/pc_pkn[i] : -Inf
            tu = ub[i] < Inf  ? -(ub[i] - x[i] - pc[i])/pc_pkn[i] : -Inf
            Λi = max(tl, tu)
            if 0.0 < Λi < γtildaplus
                γtildaplus = Λi
            end
        end
    end

    return γtildaplus

end

function compute_gammatildaminus(pc_pkn::AbstractVector{T},x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T},pc::AbstractVector{T}) where {T <: AbstractFloat}

    γtildaminus = Inf
    for i in eachindex(pc_pkn)
        g = pc_pkn[i]
        if g != 0.0
            tl = lb[i] > -Inf ? (lb[i] - x[i] - pc[i])/pc_pkn[i] : -Inf
            tu = ub[i] < Inf  ? (ub[i] - x[i] - pc[i])/pc_pkn[i] : -Inf
            Λi = max(tl, tu)
            if 0.0 < Λi < γtildaminus
                γtildaminus = Λi
            end
        end
    end

    return -γtildaminus

end

function step_selection(Fk::AbstractVector{T},x::AbstractVector{T},Gk::AbstractMatrix{T},jk::AbstractMatrix{T},gk::AbstractVector{T},pkn::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T},δk::T,θ::T) where {T <: AbstractFloat}

    # Compute the Cauchy step

    λk = Inf
    for i in eachindex(gk)
        g = gk[i]
        if g != 0.0
            tl = lb[i] > -Inf ? (lb[i] - x[i]) / g : -Inf
            tu = ub[i] < Inf  ? (ub[i] - x[i]) / g : -Inf
            Λi = max(tl, tu)
            if 0.0 < Λi < λk
                λk = Λi
            end
        end
    end

    t1 = -(Fk'*jk*gk)/norm(jk*gk)^2
    t2 = δk/norm(Gk*gk)

    taukprime = min(t1,t2)
    tauk = taukprime
    if !isequal(x+taukprime*gk, clamp.(x+taukprime*gk,lb,ub))
        tauk = θ*λk
    end

    pc = tauk*gk # Cauchy step computed

    # Step selection step

    pc_pkn = pc - pkn
    
    if norm(pc_pkn) < eps(T) # If the norm is small the step are basically the same
        return pc
    end
        
    γhat   = ((Fk+jk*pc)'*jk*pc_pkn)/norm(jk*pc_pkn)^2

    r      = max(0.0,((pc'Gk.^2*pc_pkn)^2 - norm(Gk*pc_pkn)^2*(norm(Gk*pc)^2 - δk^2)))^0.5
    γplus  = (pc'*Gk.^2*pc_pkn + r)/norm(Gk*pc_pkn)^2
    γminus = (pc'*Gk.^2*pc_pkn - r)/norm(Gk*pc_pkn)^2

    if γhat >= 0.0
        γtildaplus = compute_gammatildaplus(pc_pkn,x,lb,ub,pc)
        γ = min(γhat,γplus,θ*γtildaplus)
    else
        γtildaminus = compute_gammatildaminus(pc_pkn,x,lb,ub,pc)
        γ = max(γhat,γminus,θ*γtildaminus)
    end

    p = (1-γ)*pc + γ*pkn

    return p

end

function constrained_dogleg_solver(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_dogleg_solver_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_dogleg_solver_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_dogleg_solver_outplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of the algorithm from Bellavia, Macconi, and Pieraccini (2012), "Constrained
    # dogleg methods for nonlinear systems with simple bounds", Computational Optimization and Applications,
    # 53, pp. 771--794

    n = length(x)
    xk  = copy(x)
    xn  = similar(x)
    pkn = similar(x)
    p   = similar(x)
    df  = similar(x)

    jk = Matrix{T}(undef,n,n)
    Gk = Matrix{T}(undef,n,n)
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

    iter = 0
    while true

        Fk = f(xk)

        jk .= ForwardDiff.jacobian(f,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        gradf = jk'*Fk

        df .= coleman_li_scale(gradf,xk,lb,ub)
        Gk .= Diagonal(df.^(-1/2))

        gk .= -df.*gradf

        singular_jk = false
        pN = try
            -jk\Fk # unrestricted Newton step
        catch
            singular_jk = true
            zeros(T,n)
        end

        α = max(θ, 1.0 - norm(Fk))
        pkn = singular_jk ? zeros(T,n) : α*(clamp.(xk .+ pN, lb, ub) .- xk) # restricted Newton step

        δk = δ
        while true

            p .= step_selection(Fk,xk,Gk,jk,gk,pkn,lb,ub,δk,θ)

            # Ensure strict feasibility of trial point
            x_trial = clamp.(xk .+ p, lb, ub)
            @inbounds for i in 1:n
                if x_trial[i] <= lb[i]
                    x_trial[i] = lb[i] + eps(T) * (1.0 + abs(lb[i]))
                end
                if x_trial[i] >= ub[i]
                    x_trial[i] = ub[i] - eps(T) * (1.0 + abs(ub[i]))
                end
            end
            p .= x_trial .- xk

            norm_Fk  = norm(Fk)
            Fk_trial = f(xk .+ p)
            denom_ρ  = norm_Fk - norm(Fk .+ jk*p)
            ρ = abs(denom_ρ) > eps(T) ? (norm_Fk - norm(Fk_trial)) / denom_ρ : 0.0

            if ρ >= β1
                if ρ >= β2
                    δk = max(δk, γ1*norm(p))
                end
                xn .= xk .+ p
                break
            else
                δk = min(γ2*δk, 0.5*norm(p))
                if δk < sqrt(eps(T))
                    return BoxSolverResults(:dogleg, x, xk, Fk, lenx, lenf, iter, solution_trace)
                end
            end
        end

        lenx = maximum(abs,xn.-xk)
        lenf = maximum(abs,f(xn))

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end

    results = BoxSolverResults(:dogleg,x,xk,f(xk),lenx,lenf,iter,solution_trace)

    return results

end

function constrained_dogleg_solver_inplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of the algorithm from Bellavia, Macconi, and Pieraccini (2012), "Constrained 
    # dogleg methods for nonlinear systems with simple bounds", Computational Optimization and Applications, 
    # 53, pp. 771--794 

    n = length(x)
    xk  = copy(x)
    xn  = similar(x)
    pkn = similar(x)
    p   = similar(x)
    df  = similar(x)

    jk = Matrix{T}(undef,n,n)
    Gk = Matrix{T}(undef,n,n)
    gk = similar(x)

    lenx = zero(T)
    lenf = zero(T)
    
    Fk       = Vector{T}(undef,n)
    Fk_trial = Vector{T}(undef,n)
    Fn       = Vector{T}(undef,n)

    # Initialize solution trace
    f(Fk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    δ  = 1.0 
    γ1 = 2.0
    γ2 = 0.25
    θ  = 0.99995 
    β1 = 0.5 
    β2 = 0.9 

    iter = 0
    while true

        jk .= ForwardDiff.jacobian(f,Fk,xk)
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        gradf = jk'*Fk

        df .= coleman_li_scale(gradf,xk,lb,ub)
        Gk .= Diagonal(df.^(-1/2))

        gk .= -df.*gradf

        singular_jk = false
        pN = try
            -jk\Fk # unrestricted Newton step
        catch
            singular_jk = true
            zeros(T,n)
        end

        α = max(θ, 1.0 - norm(Fk))
        pkn = singular_jk ? zeros(T,n) : α*(clamp.(xk .+ pN, lb, ub) .- xk) # restricted Newton step

        δk = δ
        while true

            p .= step_selection(Fk,xk,Gk,jk,gk,pkn,lb,ub,δk,θ)

            # Ensure strict feasibility of trial point
            x_trial = clamp.(xk .+ p, lb, ub)
            @inbounds for i in 1:n
                if x_trial[i] <= lb[i]
                    x_trial[i] = lb[i] + eps(T) * (1.0 + abs(lb[i]))
                end
                if x_trial[i] >= ub[i]
                    x_trial[i] = ub[i] - eps(T) * (1.0 + abs(ub[i]))
                end
            end
            p .= x_trial .- xk

            norm_Fk  = norm(Fk)
            f(Fk_trial,xk .+ p)
            denom_ρ  = norm_Fk - norm(Fk .+ jk*p)
            ρ = abs(denom_ρ) > eps(T) ? (norm_Fk - norm(Fk_trial)) / denom_ρ : 0.0

            if ρ >= β1
                if ρ >= β2
                    δk = max(δk, γ1*norm(p))
                end
                xn .= xk .+ p
                break
            else
                δk = min(γ2*δk, 0.5*norm(p))
                if δk < sqrt(eps(T))
                    return BoxSolverResults(:dogleg, x, xk, Fk, lenx, lenf, iter, solution_trace)
                end
            end
        end
    
        f(Fn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fn)

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(Fn,xn)
    results = BoxSolverResults(:dogleg,x,xn,Fn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_solver(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_dogleg_solver_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_dogleg_solver_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_dogleg_solver_outplace(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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

    jk = Matrix{T}(undef,n,n)
    Gk = Matrix{T}(undef,n,n)
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

    iter = 0
    while true

        Fk = f(xk)

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        gradf = jk'*Fk

        df .= coleman_li_scale(gradf,xk,lb,ub)
        Gk .= Diagonal(df.^(-1/2))

        gk .= -df.*gradf

        singular_jk = false
        pN = try
            -jk\Fk # unrestricted Newton step
        catch
            singular_jk = true
            zeros(T,n)
        end

        α = max(θ, 1.0 - norm(Fk))
        pkn = singular_jk ? zeros(T,n) : α*(clamp.(xk .+ pN, lb, ub) .- xk) # restricted Newton step

        δk = δ
        while true

            p .= step_selection(Fk,xk,Gk,jk,gk,pkn,lb,ub,δk,θ)

            # Ensure strict feasibility of trial point
            x_trial = clamp.(xk .+ p, lb, ub)
            @inbounds for i in 1:n
                if x_trial[i] <= lb[i]
                    x_trial[i] = lb[i] + eps(T) * (1.0 + abs(lb[i]))
                end
                if x_trial[i] >= ub[i]
                    x_trial[i] = ub[i] - eps(T) * (1.0 + abs(ub[i]))
                end
            end
            p .= x_trial .- xk

            norm_Fk  = norm(Fk)
            Fk_trial = f(xk .+ p)
            denom_ρ  = norm_Fk - norm(Fk .+ jk*p)
            ρ = abs(denom_ρ) > eps(T) ? (norm_Fk - norm(Fk_trial)) / denom_ρ : 0.0

            if ρ >= β1
                if ρ >= β2
                    δk = max(δk, γ1*norm(p))
                end
                xn .= xk .+ p
                break
            else
                δk = min(γ2*δk, 0.5*norm(p))
                if δk < sqrt(eps(T))
                    return BoxSolverResults(:dogleg, x, xk, Fk, lenx, lenf, iter, solution_trace)
                end
            end
        end

        lenx = maximum(abs,xn.-xk)
        lenf = maximum(abs,f(xn))

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end

    results = BoxSolverResults(:dogleg,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_solver_inplace(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

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

    jk = Matrix{T}(undef,n,n)
    Gk = Matrix{T}(undef,n,n)
    gk = similar(x)

    lenx = zero(T)
    lenf = zero(T)
    
    Fk       = Vector{T}(undef,n)
    Fk_trial = Vector{T}(undef,n)
    Fn       = Vector{T}(undef,n)

    # Initialize solution trace
    f(Fk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    δ  = 1.0 
    γ1 = 2.0
    γ2 = 0.25
    θ  = 0.99995 
    β1 = 0.5 
    β2 = 0.9 

    iter = 0
    while true

        f(Fk,xk)

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        gradf = jk'*Fk

        df .= coleman_li_scale(gradf,xk,lb,ub)
        Gk .= Diagonal(df.^(-1/2))

        gk .= -df.*gradf

        singular_jk = false
        pN = try
            -jk\Fk # unrestricted Newton step
        catch
            singular_jk = true
            zeros(T,n)
        end

        α = max(θ, 1.0 - norm(Fk))
        pkn = singular_jk ? zeros(T,n) : α*(clamp.(xk .+ pN, lb, ub) .- xk) # restricted Newton step

        δk = δ
        while true

            p .= step_selection(Fk,xk,Gk,jk,gk,pkn,lb,ub,δk,θ)

            # Ensure strict feasibility of trial point
            x_trial = clamp.(xk .+ p, lb, ub)
            @inbounds for i in 1:n
                if x_trial[i] <= lb[i]
                    x_trial[i] = lb[i] + eps(T) * (1.0 + abs(lb[i]))
                end
                if x_trial[i] >= ub[i]
                    x_trial[i] = ub[i] - eps(T) * (1.0 + abs(ub[i]))
                end
            end
            p .= x_trial .- xk

            norm_Fk  = norm(Fk)
            f(Fk_trial,xk .+ p)
            denom_ρ  = norm_Fk - norm(Fk .+ jk*p)
            ρ = abs(denom_ρ) > eps(T) ? (norm_Fk - norm(Fk_trial)) / denom_ρ : 0.0

            if ρ >= β1
                if ρ >= β2
                    δk = max(δk, γ1*norm(p))
                end
                xn .= xk .+ p
                break
            else
                δk = min(γ2*δk, 0.5*norm(p))
                if δk < sqrt(eps(T))
                    return BoxSolverResults(:dogleg, x, xk, Fk, lenx, lenf, iter, solution_trace)
                end
            end
        end
    
        f(Fn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fn)

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(Fn,xn)
    results = BoxSolverResults(:dogleg,x,xn,Fn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_solver_sparse(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_dogleg_solver_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_dogleg_solver_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_dogleg_solver_sparse_outplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of the algorithm from Bellavia, Macconi, and Pieraccini (2012), "Constrained 
    # dogleg methods for nonlinear systems with simple bounds", Computational Optimization and Applications, 
    # 53, pp. 771--794 

    n = length(x)
    xk  = copy(x)
    xn  = similar(x)
    pkn = similar(x)
    p   = similar(x)
    df  = similar(x)

    jk = sparse(Matrix{T}(undef,n,n))
    Gk = Matrix{T}(undef,n,n)
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

    iter = 0
    while true

        Fk = f(xk)

        jk .= sparse(ForwardDiff.jacobian(f,xk))
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        gradf = jk'*Fk

        df .= coleman_li_scale(gradf,xk,lb,ub)
        Gk .= sparse(Diagonal(df.^(-1/2)))

        gk .= -df.*gradf

        singular_jk = false
        pN = try
            -jk\Fk # unrestricted Newton step
        catch
            singular_jk = true
            zeros(T,n)
        end

        α = max(θ, 1.0 - norm(Fk))
        pkn = singular_jk ? zeros(T,n) : α*(clamp.(xk .+ pN, lb, ub) .- xk) # restricted Newton step

        δk = δ
        while true

            p .= step_selection(Fk,xk,Gk,jk,gk,pkn,lb,ub,δk,θ)

            # Ensure strict feasibility of trial point
            x_trial = clamp.(xk .+ p, lb, ub)
            @inbounds for i in 1:n
                if x_trial[i] <= lb[i]
                    x_trial[i] = lb[i] + eps(T) * (1.0 + abs(lb[i]))
                end
                if x_trial[i] >= ub[i]
                    x_trial[i] = ub[i] - eps(T) * (1.0 + abs(ub[i]))
                end
            end
            p .= x_trial .- xk

            norm_Fk  = norm(Fk)
            Fk_trial = f(xk .+ p)
            denom_ρ  = norm_Fk - norm(Fk .+ jk*p)
            ρ = abs(denom_ρ) > eps(T) ? (norm_Fk - norm(Fk_trial)) / denom_ρ : 0.0

            if ρ >= β1
                if ρ >= β2
                    δk = max(δk, γ1*norm(p))
                end
                xn .= xk .+ p
                break
            else
                δk = min(γ2*δk, 0.5*norm(p))
                if δk < sqrt(eps(T))
                    return BoxSolverResults(:dogleg, x, xk, Fk, lenx, lenf, iter, solution_trace)
                end
            end
        end

        lenx = maximum(abs,xn.-xk)
        lenf = maximum(abs,f(xn))

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = BoxSolverResults(:dogleg,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_dogleg_solver_sparse_inplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of the algorithm from Bellavia, Macconi, and Pieraccini (2012), "Constrained 
    # dogleg methods for nonlinear systems with simple bounds", Computational Optimization and Applications, 
    # 53, pp. 771--794 

    n = length(x)
    xk  = copy(x)
    xn  = similar(x)
    pkn = similar(x)
    p   = similar(x)
    df  = similar(x)

    jk = sparse(Matrix{T}(undef,n,n))
    Gk = Matrix{T}(undef,n,n)
    gk = similar(x)

    lenx = zero(T)
    lenf = zero(T)
    
    Fk       = Vector{T}(undef,n)
    Fk_trial = Vector{T}(undef,n)
    Fn       = Vector{T}(undef,n)

    # Initialize solution trace
    f(Fk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
    solution_trace = SolverTrace(Array{SolverState}(undef,0))
    push!(solution_trace.trace,solver_state)

    # Initialize solver-parameters
    δ  = 1.0 
    γ1 = 2.0
    γ2 = 0.25
    θ  = 0.99995 
    β1 = 0.5 
    β2 = 0.9 

    iter = 0
    while true

        jk .= sparse(ForwardDiff.jacobian(f,Fk,xk))
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        gradf = jk'*Fk

        df .= coleman_li_scale(gradf,xk,lb,ub)
        Gk .= sparse(Diagonal(df.^(-1/2)))

        gk .= -df.*gradf

        singular_jk = false
        pN = try
            -jk\Fk # unrestricted Newton step
        catch
            singular_jk = true
            zeros(T,n)
        end

        α = max(θ, 1.0 - norm(Fk))
        pkn = singular_jk ? zeros(T,n) : α*(clamp.(xk .+ pN, lb, ub) .- xk) # restricted Newton step

        δk = δ
        while true

            p .= step_selection(Fk,xk,Gk,jk,gk,pkn,lb,ub,δk,θ)

            # Ensure strict feasibility of trial point
            x_trial = clamp.(xk .+ p, lb, ub)
            @inbounds for i in 1:n
                if x_trial[i] <= lb[i]
                    x_trial[i] = lb[i] + eps(T) * (1.0 + abs(lb[i]))
                end
                if x_trial[i] >= ub[i]
                    x_trial[i] = ub[i] - eps(T) * (1.0 + abs(ub[i]))
                end
            end
            p .= x_trial .- xk

            norm_Fk  = norm(Fk)
            f(Fk_trial,xk .+ p)
            denom_ρ  = norm_Fk - norm(Fk .+ jk*p)
            ρ = abs(denom_ρ) > eps(T) ? (norm_Fk - norm(Fk_trial)) / denom_ρ : 0.0

            if ρ >= β1
                if ρ >= β2
                    δk = max(δk, γ1*norm(p))
                end
                xn .= xk .+ p
                break
            else
                δk = min(γ2*δk, 0.5*norm(p))
                if δk < sqrt(eps(T))
                    return BoxSolverResults(:dogleg, x, xk, Fk, lenx, lenf, iter, solution_trace)
                end
            end
        end
    
        f(Fn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fn)

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(Fn,xn)
    results = BoxSolverResults(:dogleg,x,xn,Fn,lenx,lenf,iter,solution_trace)

    return results
 
end

### Constrained Newton-Krylov

function constrained_newton_krylov(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_newton_krylov_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
    else
        return constrained_newton_krylov_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
    end

end

function constrained_newton_krylov_outplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 1 from Chen and Vuik (2016) "Globalization Technique for 
    # Projected Newton-Krylov Methods", International Journal for Numerical Methods in Engineering, 
    # 110, pp. 661--674.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)

    jk = Matrix{T}(undef,n,n)
    Fk = Vector{T}(undef,n)

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

        Fk .= f(xk)

        if flag_ng == false
            jk .= ForwardDiff.jacobian(f,xk)
            if !all(isfinite,jk)
                error("The jacobian has non-finite elements")
            end    
            dk, status = gmres(jk,-Fk,xk,ηk,krylovdim) # Inexact restarted

            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                xt .= clamp.(xt,lb,ub)
                if norm(f(xt)) <= (1.0 - t*α*(1-ηk))*norm(Fk)
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
            dk = -jk'Fk
            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                xt .= clamp.(xt,lb,ub)
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
  
    results = BoxSolverResults(:nk,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov_inplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 1 from Chen and Vuik (2016) "Globalization Technique for 
    # Projected Newton-Krylov Methods", International Journal for Numerical Methods in Engineering, 
    # 110, pp. 661--674.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)

    jk = Matrix{T}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    Fk = Vector{T}(undef,n)
    Fn = Vector{T}(undef,n)

    # Initialize solution trace
    f(Fk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
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
            jk .= ForwardDiff.jacobian(f,Fk,xk)
            if !all(isfinite,jk)
                error("The jacobian has non-finite elements")
            end    
            dk, status = gmres(jk,-Fk,xk,ηk,krylovdim) # Inexact restarted

            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                xt .= clamp.(xt,lb,ub)
                f(Fn,xt)
                if norm(Fn) <= (1.0 - t*α*(1-ηk))*norm(Fk)
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
            dk = -jk'Fk
            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                xt .= clamp.(xt,lb,ub)
                f(Fn,xt)
                if 0.5*norm(Fn)^2 <=  0.5*norm(Fk)^2 + σ*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    α = β*α
                    m += 1
                end
            end
        end

        f(Fn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fn)

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(Fn,xn)
    results = BoxSolverResults(:nk,x,xn,Fn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_newton_krylov_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
    else
        return constrained_newton_krylov_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
    end

end

function constrained_newton_krylov_outplace(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 1 from Chen and Vuik (2016) "Globalization Technique for 
    # Projected Newton-Krylov Methods", International Journal for Numerical Methods in Engineering, 
    # 110, pp. 661--674.

    j_inplace = !applicable(j,x)
    
    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)

    jk = Matrix{T}(undef,n,n)
    Fk = Vector{T}(undef,n)

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
    mmax = 50

    g(x) = 0.5*norm(f(x))^2

    flag_ng = false
    iter = 0
    while true

        Fk .= f(xk)

        if flag_ng == false
            if j_inplace == false
                jk .= j(xk)
            else
                j(jk,xk)
            end
            if !all(isfinite,jk)
                error("The jacobian has non-finite elements")
            end
    
            dk, status = gmres(jk,-Fk,xk,ηk,krylovdim) # Inexact restarted

            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                xt .= clamp.(xt,lb,ub)
                if norm(f(xt)) <= (1.0 - t*α*(1-ηk))*norm(Fk)
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
            dk = -jk'Fk
            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                xt .= clamp.(xt,lb,ub)
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
  
    results = BoxSolverResults(:nk,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov_inplace(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 1 from Chen and Vuik (2016) "Globalization Technique for 
    # Projected Newton-Krylov Methods", International Journal for Numerical Methods in Engineering, 
    # 110, pp. 661--674.

    j_inplace = !applicable(j,x)

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)

    jk = Matrix{T}(undef,n,n)

    lenx = zero(T)
    lenf = zero(T)

    Fk = Vector{T}(undef,n)
    Fn = Vector{T}(undef,n)

    # Initialize solution trace
    f(Fk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
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
    
            f(Fk,xk)
            dk, status = gmres(jk,-Fk,xk,ηk,krylovdim) # Inexact restarted

            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                xt .= clamp.(xt,lb,ub)
                f(Fn,xt)
                if norm(Fn) <= (1.0 - t*α*(1-ηk))*norm(Fk)
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
            dk = -jk'Fk
            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                xt .= clamp.(xt,lb,ub)
                f(Fn,xt)
                if 0.5*norm(Fn)^2 <=  0.5*norm(Fk)^2 + σ*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    α = β*α
                    m += 1
                end
            end
        end

        f(Fn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fn)

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(Fn,xn)
    results = BoxSolverResults(:nk,x,xn,Fn,lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov_sparse(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_newton_krylov_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
    else
        return constrained_newton_krylov_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
    end

end

function constrained_newton_krylov_sparse_outplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 1 from Chen and Vuik (2016) "Globalization Technique for 
    # Projected Newton-Krylov Methods", International Journal  for Numerical Methods in Engineering, 
    # 110, pp. 661--674.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)

    jk = sparse(Matrix{T}(undef,n,n))
    Fk = Vector{T}(undef,n)

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

        Fk .= f(xk)

        if flag_ng == false
            jk .= sparse(ForwardDiff.jacobian(f,xk))
            if !all(isfinite,nonzeros(jk))
                error("The jacobian has non-finite elements")
            end    
            dk, status = gmres(jk,-Fk,xk,ηk,krylovdim) # Inexact restarted

            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                xt .= clamp.(xt,lb,ub)
                if norm(f(xt)) <= (1.0 - t*α*(1-ηk))*norm(Fk)
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
            dk = -jk'Fk
            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                xt .= clamp.(xt,lb,ub)
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
  
    results = BoxSolverResults(:nk,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_newton_krylov_sparse_inplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 1 from Chen and Vuik (2016) "Globalization Technique for 
    # Projected Newton-Krylov Methods", International Journal  for Numerical Methods in Engineering, 
    # 110, pp. 661--674.

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    xt = similar(x)

    jk = sparse(Matrix{T}(undef,n,n))

    lenx = zero(T)
    lenf = zero(T)

    Fk = Vector{T}(undef,n)
    Fn = Vector{T}(undef,n)

    # Initialize solution trace
    f(Fk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
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
            jk .= sparse(ForwardDiff.jacobian(f,Fk,xk))
            if !all(isfinite,nonzeros(jk))
                error("The jacobian has non-finite elements")
            end    
            dk, status = gmres(jk,-Fk,xk,ηk,krylovdim) # Inexact restarted

            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                xt .= clamp.(xt,lb,ub)
                f(Fn,xt)
                if norm(Fn) <= (1.0 - t*α*(1-ηk))*norm(Fk)
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
            dk = -jk'Fk
            α = 1.0
            m = 1
            while m <= mmax
                xt .= xk+α*dk
                xt .= clamp.(xt,lb,ub)
                f(Fn,xt)
                if 0.5*norm(Fn)^2 <=  0.5*norm(Fk)^2 + σ*dk'*(xt - xk)
                    xn .= xt
                    flag_ng = false
                    break
                else
                    α = β*α
                    m += 1
                end
            end
        end

        f(Fn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fn)

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(Fn,xn)
    results = BoxSolverResults(:nk,x,xn,Fn,lenx,lenf,iter,solution_trace)

    return results
 
end

### Constrained Jacobian-free Newton-Krylov

function constrained_jacobian_free_newton_krylov(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_jacobian_free_newton_krylov_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
    else
        return constrained_jacobian_free_newton_krylov_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
    end

end

function constrained_jacobian_free_newton_krylov_outplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This function is my implementation of a constrained Jacobian-free Newton-Krylov solver

    n = length(x)

    xk = copy(x)
    xn = similar(x)
    dk = similar(x)

    Fk = Vector{T}(undef,n)

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

        Fk .= f(xk)

        if flag_ng == false
            dk, status = jacobian_free_gmres(f,xk,ηk,krylovdim) # Using inexact restarted Arnoldi (could use inexact)

            α = 1.0
            m = 1
            while m <= mmax
                if norm(f(clamp(xk+α*dk,lb,ub))) <= (1.0 - t*α*(1-ηk))*norm(Fk)
                    xn = clamp(xk+α*dk,lb,ub)
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
                if g(clamp(xk+α*dk,lb,ub)) <= g(xk) + σ*dk'*(clamp(xk+α*dk,lb,ub) .- xk)
                    xn = clamp(xk+α*dk,lb,ub)
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
  
    results = BoxSolverResults(:jfnk,x,xn,f(xn),lenx,lenf,iter,solution_trace)

    return results
 
end

function constrained_jacobian_free_newton_krylov_inplace(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

    # This function is my implementation of a constrained Jacobian-free Newton-Krylov solver

    n = length(x)
    xk = copy(x)
    xn = similar(x)
    dk = similar(x)

    lenx = zero(T)
    lenf = zero(T)

    Fk = Vector{T}(undef,n)
    Fn = Vector{T}(undef,n)

    # Initialize solution trace
    f(Fk,xk)
    solver_state = SolverState(0,NaN,maximum(abs,Fk))
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
                f(Fk,xk)
                f(Fn,clamp(xk+α*dk,lb,ub))
                if norm(Fn) <= (1.0 - t*α*(1-ηk))*norm(Fk)
                    xn = clamp(xk+α*dk,lb,ub)
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
                f(Fn,clamp(xk+α*dk,lb,ub))
                if 0.5*norm(Fn)^2 <= 0.5*norm(Fk) + σ*dk'*(clamp(xk+α*dk,lb,ub) .- xk)
                    xn = clamp(xk+α*dk,lb,ub)
                    flag_ng = false
                    break
                else
                    α = β*α
                    m += 1
                end
            end
        end

        f(Fn,xn)
        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,Fn)

        xk .= xn

        iter += 1

        solver_state = SolverState(iter,lenx,lenf)
        push!(solution_trace.trace,solver_state)

        if iter >= iterations || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    f(Fn,xn)
    results = BoxSolverResults(:jfnk,x,xn,Fn,lenx,lenf,iter,solution_trace)

    return results
 
end

###########################

# Constrained Coleman Li (1996)

function _steihaug_cg(g::AbstractVector{T}, B, Δ::T, iterations::S) where {T <: AbstractFloat, S <: Integer}

    n  = length(g)
    q  = zeros(T, n)
    r  = copy(g)      
    p  = -copy(r)     
    rr = dot(r,r)
    ε  = 1e-10*rr     

    rr == 0 && return q    

    for _ in 1:iterations

        Bp = B * p
        κ  = dot(p,Bp)    

        κ ≤ 0 && return _tr_boundary_point(q, p, Δ)

        α = rr / κ

        norm(q .+ α .* p) ≥ Δ && return _tr_boundary_point(q, p, Δ)

        q   = q .+ α .* p
        r   = r .+ α .* Bp
        rr_new = dot(r,r)

        rr_new ≤ ε && break   

        β  = rr_new / rr
        p  = -r .+ β .* p    
        rr = rr_new

    end

    return q

end

function _tr_boundary_point(q::AbstractVector{T}, p::AbstractVector{T}, Δ::T) where {T <: AbstractFloat}

    qq   = dot(q,q)
    qp   = dot(q,p)
    pp   = dot(p,p)
    disc = qp^2 - pp * (qq - Δ^2) 
    τ    = (-qp + sqrt(max(disc, zero(T)))) / pp

    return q .+ τ .* p

end

function constrained_coleman_li(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_coleman_li_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_coleman_li_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_coleman_li_outplace(f::Function, x::AbstractVector{T}, lb::AbstractVector{T}, ub::AbstractVector{T}; xtol::T=1e-8, ftol::T=1e-8, iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    n  = length(x)
    xk = copy(x)
    xn = similar(x)
    df = similar(x)
    gk = similar(x)

    jk = Matrix{T}(undef, n, n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state   = SolverState(0, NaN, maximum(abs, f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef, 0))
    push!(solution_trace.trace, solver_state)

    # Initialize solver parameters
    δ     = 1.0
    δ_max = 1_000
    η1    = 0.1
    η2    = 0.9
    γ1    = 0.25
    γ2    = 2.5

    xk .= clamp.(x, lb, ub)

    iter = 0
    while true

        Fk = f(xk)

        jk .= ForwardDiff.jacobian(f, xk)
        if !all(isfinite, jk)
            error("The jacobian has non-finite elements")
        end

        gradf = jk' * Fk

        df .= coleman_li_scale(gradf, xk, lb, ub).^(1/2)
        gk .= df .* gradf

        D = Diagonal(df)
        B = Symmetric(D * (jk' * jk) * D)

        q = _steihaug_cg(gk, B, δ, n)

        xn .= clamp.(xk .+ df .* q, lb, ub)

        fval  = 0.5 * dot(Fk, Fk)
        pred  = -(dot(gk, q) + 0.5 * dot(q, B * q))
        Fn    = f(xn)
        fnew  = 0.5 * dot(Fn, Fn)
        ared  = fval - fnew
        ρ     = pred > eps(T) ? ared / pred : T(ared ≥ 0 ? 1 : -1)

        lenx = maximum(abs,xn .- xk)
        lenf = maximum(abs,f(xn))

        if ρ ≥ η1
            xk .= xn
        end

        if ρ < η1 # Shrink the radius
            δ = γ1 * δ
        elseif ρ ≥ η2 && norm(q) ≥ 0.9 * δ # Expand the radius
            δ = min(γ2 * δ, δ_max)
        end

        iter += 1

        solver_state = SolverState(iter, lenx, lenf)
        push!(solution_trace.trace, solver_state)

        δ ≤ eps(T) * (1 + norm(xk)) && break

        if iter >= iterations || (lenx <= xtol && δ <= xtol) || (lenf <= ftol)
            break
        end

    end

    results = BoxSolverResults(:coleman_li, x, xk, f(xk), lenx, lenf, iter, solution_trace)

    return results

end

function constrained_coleman_li_inplace(f::Function, x::AbstractVector{T}, lb::AbstractVector{T}, ub::AbstractVector{T}; xtol::T=1e-8, ftol::T=1e-8, iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    n  = length(x)
    xk = copy(x)
    xn = similar(x)
    df = similar(x)
    gk = similar(x)

    jk = Matrix{T}(undef, n, n)

    lenx = zero(T)
    lenf = zero(T)

    Fk = Vector{T}(undef, n)
    Fn = Vector{T}(undef, n)

    # Initialize solution trace
    f(Fk, xk)
    solver_state   = SolverState(0, NaN, maximum(abs, Fk))
    solution_trace = SolverTrace(Array{SolverState}(undef, 0))
    push!(solution_trace.trace, solver_state)

    # Initialize solver parameters
    δ     = 1.0
    δ_max = 1_000
    η1    = 0.1
    η2    = 0.9
    γ1    = 0.25
    γ2    = 2.5

    xk .= clamp.(x, lb, ub)

    iter = 0
    while true

        f(Fk, xk)

        jk .= ForwardDiff.jacobian(f, Fk, xk)
        if !all(isfinite, jk)
            error("The jacobian has non-finite elements")
        end

        gradf = jk' * Fk

        df .= coleman_li_scale(gradf, xk, lb, ub).^(1/2)
        gk .= df .* gradf

        D = Diagonal(df)
        B = Symmetric(D * (jk' * jk) * D)

        q = _steihaug_cg(gk, B, δ, n)

        xn .= clamp.(xk .+ df .* q, lb, ub)

        fval  = 0.5 * dot(Fk, Fk)
        pred  = -(dot(gk, q) + 0.5 * dot(q, B * q))
        f(Fn, xn)
        fnew  = 0.5 * dot(Fn, Fn)
        ared  = fval - fnew
        ρ     = pred > eps(T) ? ared / pred : T(ared ≥ 0 ? 1 : -1)

        lenx = maximum(abs, xn .- xk)
        lenf = maximum(abs, Fn)

        if ρ ≥ η1
            xk .= xn
        end

        if ρ < η1 # Shrink the radius
            δ = γ1 * δ
        elseif ρ ≥ η2 && norm(q) ≥ 0.9 * δ # Expand the radius
            δ = min(γ2 * δ, δ_max)
        end

        iter += 1

        solver_state = SolverState(iter, lenx, lenf)
        push!(solution_trace.trace, solver_state)

        δ ≤ eps(T) * (1 + norm(xk)) && break

        if iter >= iterations || (lenx <= xtol && δ <= xtol) || (lenf <= ftol)
            break
        end

    end

    f(Fn, xk)
    results = BoxSolverResults(:coleman_li, x, xk, Fn, lenx, lenf, iter, solution_trace)

    return results

end

function constrained_coleman_li(f::Function,j::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_coleman_li_outplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_coleman_li_inplace(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_coleman_li_outplace(f::Function, j::Function, x::AbstractVector{T}, lb::AbstractVector{T}, ub::AbstractVector{T}; xtol::T=1e-8, ftol::T=1e-8, iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    j_inplace = !applicable(j,x)

    n  = length(x)
    xk = copy(x)
    xn = similar(x)
    df = similar(x)
    gk = similar(x)

    jk = Matrix{T}(undef, n, n)

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state   = SolverState(0, NaN, maximum(abs, f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef, 0))
    push!(solution_trace.trace, solver_state)

    # Initialize solver parameters
    δ     = 1.0
    δ_max = 1_000
    η1    = 0.1
    η2    = 0.9
    γ1    = 0.25
    γ2    = 2.5

    xk .= clamp.(x, lb, ub)

    iter = 0
    while true

        Fk = f(xk)

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        gradf = jk' * Fk

        df .= coleman_li_scale(gradf, xk, lb, ub).^(1/2)
        gk .= df .* gradf

        D = Diagonal(df)
        B = Symmetric(D * (jk' * jk) * D)

        q = _steihaug_cg(gk, B, δ, n)

        xn .= clamp.(xk .+ df .* q, lb, ub)

        fval  = 0.5 * dot(Fk, Fk)
        pred  = -(dot(gk, q) + 0.5 * dot(q, B * q))
        Fn    = f(xn)
        fnew  = 0.5 * dot(Fn, Fn)
        ared  = fval - fnew
        ρ     = pred > eps(T) ? ared / pred : T(ared ≥ 0 ? 1 : -1)

        lenx = maximum(abs,xn .- xk)
        lenf = maximum(abs,f(xn))

        if ρ ≥ η1
            xk .= xn
        end

        if ρ < η1 # Shrink the radius
            δ = γ1 * δ
        elseif ρ ≥ η2 && norm(q) ≥ 0.9 * δ # Expand the radius
            δ = min(γ2 * δ, δ_max)
        end

        iter += 1

        solver_state = SolverState(iter, lenx, lenf)
        push!(solution_trace.trace, solver_state)

        δ ≤ eps(T) * (1 + norm(xk)) && break

        if iter >= iterations || (lenx <= xtol && δ <= xtol) || (lenf <= ftol)
            break
        end

    end

    results = BoxSolverResults(:coleman_li, x, xk, f(xk), lenx, lenf, iter, solution_trace)

    return results

end

function constrained_coleman_li_inplace(f::Function, j::Function, x::AbstractVector{T}, lb::AbstractVector{T}, ub::AbstractVector{T}; xtol::T=1e-8, ftol::T=1e-8, iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    j_inplace = !applicable(j,x)

    n  = length(x)
    xk = copy(x)
    xn = similar(x)
    df = similar(x)
    gk = similar(x)

    jk = Matrix{T}(undef, n, n)

    lenx = zero(T)
    lenf = zero(T)

    Fk = Vector{T}(undef, n)
    Fn = Vector{T}(undef, n)

    # Initialize solution trace
    f(Fk, xk)
    solver_state   = SolverState(0, NaN, maximum(abs, Fk))
    solution_trace = SolverTrace(Array{SolverState}(undef, 0))
    push!(solution_trace.trace, solver_state)

    # Initialize solver parameters
    δ     = 1.0
    δ_max = 1_000
    η1    = 0.1
    η2    = 0.9
    γ1    = 0.25
    γ2    = 2.5

    xk .= clamp.(x, lb, ub)

    iter = 0
    while true

        f(Fk, xk)

        if j_inplace == false
            jk .= j(xk)
        else
            j(jk,xk)
        end
        if !all(isfinite,jk)
            error("The jacobian has non-finite elements")
        end

        gradf = jk' * Fk

        df .= coleman_li_scale(gradf, xk, lb, ub).^(1/2)
        gk .= df .* gradf

        D = Diagonal(df)
        B = Symmetric(D * (jk' * jk) * D)

        q = _steihaug_cg(gk, B, δ, n)

        xn .= clamp.(xk .+ df .* q, lb, ub)

        fval  = 0.5 * dot(Fk, Fk)
        pred  = -(dot(gk, q) + 0.5 * dot(q, B * q))
        f(Fn, xn)
        fnew  = 0.5 * dot(Fn, Fn)
        ared  = fval - fnew
        ρ     = pred > eps(T) ? ared / pred : T(ared ≥ 0 ? 1 : -1)

        lenx = maximum(abs, xn .- xk)
        lenf = maximum(abs, Fn)

        if ρ ≥ η1
            xk .= xn
        end

        if ρ < η1 # Shrink the radius
            δ = γ1 * δ
        elseif ρ ≥ η2 && norm(q) ≥ 0.9 * δ # Expand the radius
            δ = min(γ2 * δ, δ_max)
        end

        iter += 1

        solver_state = SolverState(iter, lenx, lenf)
        push!(solution_trace.trace, solver_state)

        δ ≤ eps(T) * (1 + norm(xk)) && break

        if iter >= iterations || (lenx <= xtol && δ <= xtol) || (lenf <= ftol)
            break
        end

    end

    f(Fn, xk)
    results = BoxSolverResults(:coleman_li, x, xk, Fn, lenx, lenf, iter, solution_trace)

    return results

end

function constrained_coleman_li_sparse(f::Function,x::AbstractVector{T},lb::AbstractVector{T},ub::AbstractVector{T};xtol::T=1e-8,ftol::T=1e-8,iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    if box_check(lb,ub) !== true # Check that box is formed correctly
        error("Problem with box constaint.  Check that lb and ub are formed correctly and are entered in the correct order")
    end

    x .= clamp.(x,lb,ub) # Put the initial guess inside the box

    f_inplace = !applicable(f,x) # Check if function is inplace

    if f_inplace == false
        return constrained_coleman_li_sparse_outplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    else
        return constrained_coleman_li_sparse_inplace(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    end

end

function constrained_coleman_li_sparse_outplace(f::Function, x::AbstractVector{T}, lb::AbstractVector{T}, ub::AbstractVector{T}; xtol::T=1e-8, ftol::T=1e-8, iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    n  = length(x)
    xk = copy(x)
    xn = similar(x)
    df = similar(x)
    gk = similar(x)

    jk = sparse(Matrix{T}(undef, n, n))

    lenx = zero(T)
    lenf = zero(T)

    # Initialize solution trace
    solver_state   = SolverState(0, NaN, maximum(abs, f(x)))
    solution_trace = SolverTrace(Array{SolverState}(undef, 0))
    push!(solution_trace.trace, solver_state)

    # Initialize solver parameters
    δ     = 1.0
    δ_max = 1_000
    η1    = 0.1
    η2    = 0.9
    γ1    = 0.25
    γ2    = 2.5

    xk .= clamp.(x, lb, ub)

    iter = 0
    while true

        Fk = f(xk)

        jk .= sparse(ForwardDiff.jacobian(f, xk))
        if !all(isfinite, jk)
            error("The jacobian has non-finite elements")
        end

        gradf = jk' * Fk

        df .= coleman_li_scale(gradf, xk, lb, ub).^(1/2)
        gk .= df .* gradf

        D = sparse(Diagonal(df))
        B = Symmetric(D * (jk' * jk) * D)

        q = _steihaug_cg(gk, B, δ, n)

        xn .= clamp.(xk .+ df .* q, lb, ub)

        fval  = 0.5 * dot(Fk, Fk)
        pred  = -(dot(gk, q) + 0.5 * dot(q, B * q))
        Fn    = f(xn)
        fnew  = 0.5 * dot(Fn, Fn)
        ared  = fval - fnew
        ρ     = pred > eps(T) ? ared / pred : T(ared ≥ 0 ? 1 : -1)

        lenx = maximum(abs,xn .- xk)
        lenf = maximum(abs,f(xn))

        if ρ ≥ η1
            xk .= xn
        end

        if ρ < η1 # Shrink the radius
            δ = γ1 * δ
        elseif ρ ≥ η2 && norm(q) ≥ 0.9 * δ # Expand the radius
            δ = min(γ2 * δ, δ_max)
        end

        iter += 1

        solver_state = SolverState(iter, lenx, lenf)
        push!(solution_trace.trace, solver_state)

        δ ≤ eps(T) * (1 + norm(xk)) && break

        if iter >= iterations || (lenx <= xtol && δ <= xtol) || (lenf <= ftol)
            break
        end

    end

    results = BoxSolverResults(:coleman_li, x, xk, f(xk), lenx, lenf, iter, solution_trace)

    return results

end

function constrained_coleman_li_sparse_inplace(f::Function, x::AbstractVector{T}, lb::AbstractVector{T}, ub::AbstractVector{T}; xtol::T=1e-8, ftol::T=1e-8, iterations::S=100) where {T <: AbstractFloat, S <: Integer}

    n  = length(x)
    xk = copy(x)
    xn = similar(x)
    df = similar(x)
    gk = similar(x)

    jk = sparse(Matrix{T}(undef, n, n))

    lenx = zero(T)
    lenf = zero(T)

    Fk = Vector{T}(undef, n)
    Fn = Vector{T}(undef, n)

    # Initialize solution trace
    f(Fk, xk)
    solver_state   = SolverState(0, NaN, maximum(abs, Fk))
    solution_trace = SolverTrace(Array{SolverState}(undef, 0))
    push!(solution_trace.trace, solver_state)

    # Initialize solver parameters
    δ     = 1.0
    δ_max = 1_000
    η1    = 0.1
    η2    = 0.9
    γ1    = 0.25
    γ2    = 2.5

    xk .= clamp.(x, lb, ub)

    iter = 0
    while true

        f(Fk, xk)

        jk .= sparse(ForwardDiff.jacobian(f, Fk, xk))
        if !all(isfinite, jk)
            error("The jacobian has non-finite elements")
        end

        gradf = jk' * Fk

        df .= coleman_li_scale(gradf, xk, lb, ub).^(1/2)
        gk .= df .* gradf

        D = sparse(Diagonal(df))
        B = Symmetric(D * (jk' * jk) * D)

        q = _steihaug_cg(gk, B, δ, n)

        xn .= clamp.(xk .+ df .* q, lb, ub)

        fval  = 0.5 * dot(Fk, Fk)
        pred  = -(dot(gk, q) + 0.5 * dot(q, B * q))
        f(Fn, xn)
        fnew  = 0.5 * dot(Fn, Fn)
        ared  = fval - fnew
        ρ     = pred > eps(T) ? ared / pred : T(ared ≥ 0 ? 1 : -1)

        lenx = maximum(abs, xn .- xk)
        lenf = maximum(abs, Fn)

        if ρ ≥ η1
            xk .= xn
        end

        if ρ < η1 # Shrink the radius
            δ = γ1 * δ
        elseif ρ ≥ η2 && norm(q) ≥ 0.9 * δ # Expand the radius
            δ = min(γ2 * δ, δ_max)
        end

        iter += 1

        solver_state = SolverState(iter, lenx, lenf)
        push!(solution_trace.trace, solver_state)

        δ ≤ eps(T) * (1 + norm(xk)) && break

        if iter >= iterations || (lenx <= xtol && δ <= xtol) || (lenf <= ftol)
            break
        end

    end

    f(Fn, xk)
    results = BoxSolverResults(:coleman_li, x, xk, Fn, lenx, lenf, iter, solution_trace)

    return results

end

function nlboxsolve(f::Function,x::AbstractVector{T},lb::Vector{T} = [-T(Inf) for _ in eachindex(x)],ub::Vector{T}= [T(Inf) for _ in eachindex(x)];xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,method::Symbol=:lm_ar,krylovdim::S=30,sparsejac::Symbol=:no) where {T <: AbstractFloat, S <: Integer}

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
        elseif sparsejac == :yes
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
    elseif method == :cl
        if sparsejac == :no
            return constrained_coleman_li(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
        elseif sparsejac == :yes
            return constrained_coleman_li_sparse(f,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
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

function nlboxsolve(f::Function,j::Function,x::AbstractVector{T},lb = [-T(Inf) for _ in eachindex(x)],ub = [T(Inf) for _ in eachindex(x)];xtol::T=1e-8,ftol::T=1e-8,iterations::S=100,method::Symbol=:lm_ar,krylovdim::S=30) where {T <: AbstractFloat, S <: Integer}

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
    elseif method == :cl
        return constrained_coleman_li(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations)
    elseif method == :nk
        return constrained_newton_krylov(f,j,x,lb,ub,xtol=xtol,ftol=ftol,iterations=iterations,krylovdim=krylovdim)
    else
        error("Your chosen solution method is not supported")
    end

end