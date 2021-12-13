function arnoldi(a::AbstractArray{T,2},b::AbstractArray{T,1},m::S = length(b)) where {T <: AbstractFloat, S <: Integer} # Exact

    n = length(b)
    if m > n
        error("'m' must be no larger than the length of 'b' ")
    end

    q = zeros(n,m)
    h = zeros(m+1,m)

    β = norm(b)
    qk = copy(b)

    @views for k = 1:m
        q[:,k] .= qk/β
        qk .= a*q[:,k]
        for j = 1:k
            h[j,k] = qk'q[:,j]
            qk .-= h[j,k]*q[:,j]
        end
        β = norm(qk)
        h[k+1,k] = β
        if β < eps(T)
            return q[:,1:k], h[1:k,1:k], k
        end
    end

    return q, h[1:m,1:m], m

end

function arnoldi(f::Function,x::Array{T,1},r::Array{T,1},m::S=length(x)) where {T <: AbstractFloat, S <: Integer} #Exact

    n = length(x)
    if m > n
        error("'m' must be no larger than the length of 'b' ")
    end

    q = zeros(n,m)
    h = zeros(m+1,m)

    β = norm(r)
    qk = copy(r)

    @views for k = 1:m
        q[:,k] = qk/β
        jacvec!(f,x,q[:,k],qk)
        for j = 1:k
            h[j,k] = qk'q[:,j]
            qk -= h[j,k]*q[:,j]
        end
        β = norm(qk)
        h[k+1,k] = β
        if β < eps(T)
            return q[:,1:k], h[1:k,1:k], k
        end
    end

    return q, h[1:m,:], m

end

function arnoldi(a::AbstractArray{T,2},b::AbstractArray{T,1},tol::T,m::S=length(b)) where {T <: AbstractFloat, S <: Integer} #Inexact

    n = length(b)
    if m > n
        error("'m' must be no larger than the length of 'b' ")
    end

    q = zeros(n,m)
    h = zeros(m+1,m)

    β = norm(b)
    qk = copy(b)

    @views for k = 1:m
        q[:,k] .= qk/β
        qk .= a*q[:,k]
        for j = 1:k
            h[j,k] = qk'q[:,j]
            qk .-= h[j,k]*q[:,j]
        end
        β = norm(qk)
        h[k+1,k] = β
        if β < tol
            return q[1:n,1:k], h[1:k,1:k], k
        end
    end

    return q[1:n,1:m], h[1:m,1:m], m

end

function arnoldi(f::Function,x::Array{T,1},r::Array{T,1},tol::T,m::S=length(x)) where {T <: AbstractFloat, S <: Integer} # Inexact

    n = length(x)
    if m > n
        error("'m' must be no larger than the length of 'b' ")
    end

    q = zeros(n,m)
    h = zeros(m+1,m)

    β = norm(r)
    qk = copy(r)

    @views for k = 1:m
        q[:,k] = qk/β
        jacvec!(f,x,q[:,k],qk)
        for j = 1:k
            h[j,k] = qk'q[:,j]
            qk -= h[j,k]*q[:,j]
        end
        β = norm(qk)
        h[k+1,k] = β
        if β < tol
            return q[1:n,1:k], h[1:k,1:k], k
        end
    end

    return q[1:n,1:m], h[1:m,1:m], m

end
