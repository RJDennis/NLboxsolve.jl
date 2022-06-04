function givens(x::Array{N,1}) where {N <: Number}

    if length(x) != 2
        error("'x' should have length 2")
    end

    z = x[1]
    y = x[2]
  
    if y == zero(N)
        c = one(N)
        s = zero(N)
    elseif abs(y'y) >= abs(z'z)
        tau = -z/y
        s = one(N)/(sqrt(one(N)+tau'tau))
        c = s*tau
    else
        tau = -y/z
        c = one(N)/(sqrt(one(N)+tau'tau))
        s = c*tau
    end
  
    g = [c s'; -s c']
  
    return g

end

function givens!(g::Array{N,2},x::AbstractArray{N,1}) where {N <: Number}

    if length(x) != 2
        error("'x' should have length 2")
    end

    z = x[1]
    y = x[2]
  
    if y == zero(N)
        c = one(N)
        s = zero(N)
    elseif abs(y'y) >= abs(z'z)
        tau = -z/y
        s = one(N)/(sqrt(one(N)+tau'tau))
        c = s*tau
    else
        tau = -y/z
        c = one(N)/(sqrt(one(N)+tau'tau))
        s = c*tau
    end
  
    g[1] = c
    g[2] = -s
    g[3] = s'
    g[4] = c'

end