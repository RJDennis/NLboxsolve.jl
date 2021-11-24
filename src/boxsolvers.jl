struct SolverResults

    solution_method::Symbol
    initial::Array{Float64,1}
    zero::Array{Float64,1}
    fzero::Array{Float64,1}
    xdist::Float64
    fdist::Float64
    iters::Integer

end

function line_search(f::Function,x::Array{Float64,1},d::Array{Float64,1})

    alpha = 1.0
    beta  = 0.0001
    tau   = 0.5
  
    decent = (ForwardDiff.gradient(f,x)'*d)[1]
    if decent > 0.0
        error("d is not a decent direction")
    end
  
    while f(x+alpha*d) > f(x) + alpha*beta*decent
        alpha = tau*alpha
    end
  
    return alpha
  
end
  
function box_projection(x::Array{T,1},l::Array{T,1},u::Array{T,1}) where {T <: AbstractFloat}

    y = copy(x)

    for i in eachindex(x)
        if y[i] < l[i]
            y[i] = l[i]
        elseif y[i] > u[i]
            y[i] = u[i]
        end
    end

    return y

end

function constrained_newton(f::Function,x::Array{T,1},l::Array{T,1},u::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    xk = copy(x)
    xn = similar(x)

    lenx = zero(T)
    lenf = zero(T)

    iter = 1
    while true

        j = ForwardDiff.jacobian(f,xk)
        xn .= xk .- j\f(xk)
  
        xn = box_projection(xn,l,u)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
    
        xk .= xn

        iter += 1

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end
  
    end
  
    results = SolverResults(:newton,x,xn,f(xn),lenx,lenf,iter)

    return results
  
end
  
function constrained_levenberg_marquardt(f::Function,x::Array{T,1},l::Array{T,1},u::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    xk = copy(x)
    xn = similar(x)

    lenx = zero(T)
    lenf = zero(T)

    muk = (1/2)*10^(-8)*norm(f(xk))^2

    iter = 0
    while true

        j = ForwardDiff.jacobian(f,xk)

        du = -(j'j + muk*I)\(j'f(xk))

        xn .= xk .+ du

        xn = box_projection(xn,l,u)

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
  
        xk .= xn

        iter += 1

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

        muk = min(muk,norm(f(xk))^2)

    end
  
    results = SolverResults(:lm,x,xn,f(xn),lenx,lenf,iter)

    return results
   
end

function constrained_levenberg_marquardt_kyf(f::Function,x::Array{T,1},l::Array{T,1},u::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 3.12 from Kanzow, Yamashita, and Fukushima(2004) "Levenberg-Marquardt methods
    # with strong local convergence properties for solving nonlinear equations with convex constraints", Journal of 
    # Computational and Applied Mathematics, 172, pp375--397.

    xk = copy(x)
    xn = similar(x)

    lenx = zero(T)
    lenf = zero(T)

    m(x) = (1/2)*norm(f(x))^2

    beta  = 0.9
    gamma = 0.99995
    sigma = 10^(-4)
    rho   = 10^(-8)
    p     = 2.1

    muk = (1/2)*10^(-8)*norm(f(xk))^2

    iter = 0
    while true

        j = ForwardDiff.jacobian(f,xk)

        du = -(j'j + muk*I)\(j'f(xk))

        z = box_projection(xk+du,l,u)
        s = z-xk

        if norm(f(xk+du)) <= gamma*norm(f(xk))
            xn .= xk .+ s
        elseif (j's)[1] <= -rho*norm(s)^p
            alpha = line_search(m,xk,s)
            xn .= xk .+ alpha*s
        else
            t = 1.0
            while true
                xt = box_projection(xk-t*j'*f(xk),l,u)
                if f(xt) <= f(xk) .+ sigma*(f(xt)'*j*(xt-xk))[1]
                    xn .=  xt
                    break
                else
                    t = beta*t
                end
            end
        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
  
        xk .= xn

        iter += 1

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

        muk = min(muk,norm(f(xk))^2)

    end
  
    results = SolverResults(:lm_kyf,x,xn,f(xn),lenx,lenf,iter)

    return results
   
end

function constrained_levenberg_marquardt_fan(f::Function,x::Array{T,1},l::Array{T,1},u::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 2.1 from Fan (2013) "On the Levenberg-Marquardt methods for convex constrained
    # nonlinear equations", Journal of Industrial and Management Optimization, 9, 1, pp227--241.

    xk = copy(x)
    xn = similar(x)

    lenx = zero(T)
    lenf = zero(T)

    delta = 1.0
    mu    = 10^(-4)
    gamma = 0.99995
    beta  = 0.9
    sigma = 10^(-4)

    iter = 0
    while true

        muk = mu*norm(f(xk))^delta
        j = ForwardDiff.jacobian(f,xk)

        du = -(j'j + muk*I)\(j'f(xk))

        z = box_projection(xk+du,l,u)
        s = z-xk

        if norm(f(xk+du)) <= gamma*norm(f(xk))
            xn .= xk .+ s
        else
            alpha = 1.0
            while true
                xt = box_projection(xk-alpha*j'f(xk),l,u)
                if norm(f(xt))^2 <= norm(f(xk))^2 + sigma*(f(xk)'*j*(xt-xk))[1]
                    xn .=  xt
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

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:lm_fan,x,xn,f(xn),lenx,lenf,iter)

    return results
 
end

function constrained_levenberg_marquardt_ar(f::Function,x::Array{T,1},l::Array{T,1},u::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of Algorithm 2.1 from Amini and Rostami (2016), "Three-steps modified Levenberg-Marquardt 
    # method with a new line search for systems of nonlinear equations", Journal of Computational and Applied Mathematics, 
    # 300, pp30--42.

    # Modified to allow for box-constraints by Richard Dennis.

    xk = copy(x)
    xn = similar(x)

    lenx = zero(T)
    lenf = zero(T)

    sigma1 = 0.005
    sigma2 = 0.005
    rho    = 0.8
    r      = 0.5
    gamma  = 10^(-16)
    mu     = 10^(-4)

    iter = 0
    while true

        jk = ForwardDiff.jacobian(f,xk)
        lambdak = mu*norm(f(xk))^2

        d1k = -(jk'jk + lambdak*I)\(jk'f(xk))
        d2k = -(jk'jk + lambdak*I)\(jk'f(xk+d1k))
        d3k = -(jk'jk + lambdak*I)\(jk'f(xk+d1k+d2k))

        dk = d1k+d2k+d3k

        z = box_projection(xk+dk,l,u)
        s = z-xk

        if norm(f(z)) <= rho*norm(f(xk))
            alpha = 1.0
        else
            if f(xk)'*jk*dk > -gamma
                dk = d1k
                z = box_projection(xk+dk,l,u)
                s = z-xk
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

        xn .= xk .+ alpha*s

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))
  
        xk .= xn

        iter += 1

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:lm_ar,x,xn,f(xn),lenx,lenf,iter)

    return results
  
end

function coleman_li(f::Function,x::Array{T,1},l::Array{T,1},u::Array{T,1}) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    D = zeros(n,n)

    m(x) = (1/2)*norm(f(x))^2
    df = ForwardDiff.gradient(m,x)

    for i = 1:n
        if df[i] < 0.0 && u[i] < Inf
            D[i,i] = u[i] - x[i]
        elseif df[i] > 0.0 && l[i] > -Inf
            D[i,i] = x[i] - l[i]
        elseif df[i] == 0.0 && (l[i] > -Inf || u[i] < Inf)
            D[i,i] = min(x[i]-l[i],u[i]-x[i])
        else
            D[i,i] = 1.0
        end
    end

    return D

end

function step_selection(f::Function,x::Array{T,1},Dk::Array{T,2},Gk::Array{T,2},jk::Array{T,2},gk::Array{T,1},pkn::Array{T,1},l::Array{T,1},u::Array{T,1},deltak::T,theta::T) where {T <: AbstractFloat}

    n = length(x)

    lambda = zeros(n)
    for i = 1:n
        if gk[i] != 0.0
            lambda[i] = max(((l[i] - x[i])/gk[i]),((u[i]-x[i])/gk[i]))
        else
            lambda[i] = Inf
        end
    end

    lambdak = minimum(lambda)

    taukprime = min(-(f(x)'*jk*gk)/norm(jk*gk)^2,deltak/norm(Gk*gk))
    
    tauk = taukprime
    if !isequal(x+taukprime*gk, box_projection(x+taukprime*gk,l,u))
        tauk = theta*lambdak
    end

    pc = tauk*gk

    if norm(pkn-pc) < 1e-13 # "division by zero errors can otherwise occur"
        return (pc+pkn)/2
    else
        gammahat   = -(f(x)+jk*pc)'*jk*(pkn-pc)/norm(jk*(pkn-pc))^2
        if (pc'Gk^2*(pc-pkn))^2 - norm(Gk*(pc-pkn))^2*(norm(Gk*pc)^2-deltak^2) >= 0.0
            gammaplus  = (pc'*Gk^2*(pc-pkn) + ((pc'Gk^2*(pc-pkn))^2 - norm(Gk*(pc-pkn))^2*(norm(Gk*pc)^2-deltak^2))^(1/2))/norm(Gk*(pc-pkn))^2
            gammaminus = (pc'*Gk^2*(pc-pkn) - ((pc'Gk^2*(pc-pkn))^2 - norm(Gk*(pc-pkn))^2*(norm(Gk*pc)^2-deltak^2))^(1/2))/norm(Gk*(pc-pkn))^2
        else
            gammaplus  = pc'*Gk^2*(pc-pkn)/norm(Gk*(pc-pkn))^2
            gammaminus = pc'*Gk^2*(pc-pkn)/norm(Gk*(pc-pkn))^2
        end

        if gammahat > 1.0
            lambda = zeros(n)
            for i = 1:n
                if (pkn[i]-pc[i]) != 0.0
                    lambda[i] = max(((l[i] - x[i]- pc[i])/(pkn[i]-pc[i])),((u[i] - x[i] - pc[i])/(pkn[i]-pc[i])))
                else
                    lambda[i] = Inf
                end
            end
            gammatildaplus = minimum(lambda)
            gamma = min(gammahat,gammaplus,theta*gammatildaplus)
        elseif gammahat < 0.0
            lambda = zeros(n)
            for i = 1:n
                if (-pkn[i]+pc[i]) != 0.0
                    lambda[i] = max(((l[i] - x[i]- pc[i])/(-(pkn[i]-pc[i]))),((u[i] - x[i] - pc[i])/(-(pkn[i]-pc[i]))))
                else
                    lambda[i] = Inf
                end
            end
            gammatildaminus = -minimum(lambda)
            gamma = max(gammahat,gammaminus,theta*gammatildaminus)
        else
            gamma = gammahat
        end

        p = (1.0-gamma)*pc + gamma*pkn

        return p

    end

end

function constrained_dogleg_solver(f::Function,x::Array{T,1},l::Array{T,1},u::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100) where {T <: AbstractFloat, S <: Integer}

    # This is an implementation of the algorithm from Bellavia, Macconi, and Pieraccini (2012), "Constrained 
    # dogleg methods for nonlinear systems with simple bounds", Computational Optimization and Applications, 
    # 53, pp.771–794 

    xk = copy(x)
    xn = similar(x)

    lenx = zero(T)
    lenf = zero(T)

    # Replace infinities with largest possible Float64
    
    for i in eachindex(x)
        if l[i] == -Inf
            l[i] = -1/eps()
        elseif l[i] == Inf
            l[i] = 1/eps()
        end
        if u[i] == -Inf
            u[i] = -1/eps()
        elseif u[i] == Inf
            u[i] = 1/eps()
        end
    end
    
    deltak = 1.0 
    theta  = 0.99995 
    beta1  = 0.25 
    beta2  = 0.80 

    iter = 0
    while true
        jk = ForwardDiff.jacobian(f,xk)
        Dk = coleman_li(f,xk,l,u)
        Gk = Dk^(-1/2)
        gk = -Dk*jk'f(xk)
   
        alphak = max(0.99995,1.0-norm(f(xk)))
        pkn = -jk\f(xk)
        pkn = alphak*(box_projection(xk+pkn,l,u)-xk)

        p = step_selection(f,xk,Dk,Gk,jk,gk,pkn,l,u,deltak,theta)
    
        while true
            rhof = (norm(f(xk)) - norm(f(xk+p))) / (norm(f(xk)) - norm(f(xk) + jk'*p))
            if rhof < beta1 # linear approximation is poor fit so reduce the trust region
                deltak = min(0.25*deltak,0.5*norm(p))
                p = step_selection(f,xk,Dk,Gk,jk,gk,pkn,l,u,deltak,theta)
            elseif rhof > beta2 # linear approximation is good fit so expand the trust region
                deltak = max(deltak,2*norm(p))
                p = step_selection(f,xk,Dk,Gk,jk,gk,pkn,l,u,deltak,theta)
            end
            xn .= xk .+ p
            break
        end

        lenx = maximum(abs,xn-xk)
        lenf = maximum(abs,f(xn))

        xk .= xn

        iter += 1

        if iter >= maxiters || (lenx <= xtol || lenf <= ftol)
            break
        end

    end
  
    results = SolverResults(:dogleg,x,xn,f(xn),lenx,lenf,iter)

    return results
 
end

function nlboxsolve(f::Function,x::Array{T,1},l::Array{T,1},u::Array{T,1};xtol::T=1e-8,ftol::T=1e-8,maxiters::S=100,method::Symbol = :lm_ar) where {T <: AbstractFloat, S <: Integer}

    if method == :newton
        return constrained_newton(f,x,l,u,xtol=xtol,ftol=ftol,maxiters=maxiters)
    elseif method == :lm
        return constrained_levenberg_marquardt(f,x,l,u,xtol=xtol,ftol=ftol,maxiters=maxiters)
    elseif method == :lm_kyf
        return constrained_levenberg_marquardt_kyf(f,x,l,u,xtol=xtol,ftol=ftol,maxiters=maxiters)
    elseif method == :lm_fan
        return constrained_levenberg_marquardt_fan(f,x,l,u,xtol=xtol,ftol=ftol,maxiters=maxiters)
    elseif method == :lm_ar
        return constrained_levenberg_marquardt_ar(f,x,l,u,xtol=xtol,ftol=ftol,maxiters=maxiters)
    elseif method == :dogleg
        return constrained_dogleg_solver(f,x,l,u,xtol=xtol,ftol=ftol,maxiters=maxiters)
    end

end