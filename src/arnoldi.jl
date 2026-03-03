function arnoldi(a::AbstractMatrix{T},b::AbstractVector{T},m::S = length(b)) where {T <: AbstractFloat, S <: Integer} # Exact

    n = length(b)
    if m > n
        error("'m' must be no larger than the length of 'b' ")
    end

    q = zeros(T,n,m)
    h = zeros(T,m+1,m)

    β  = norm(b)
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

function arnoldi(f::Function,x::AbstractVector{T},r::AbstractVector{T},m::S=length(x)) where {T <: AbstractFloat, S <: Integer} #Exact

    n = length(x)
    if m > n
        error("'m' must be no larger than the length of 'b' ")
    end

    q = zeros(T,n,m)
    h = zeros(T,m+1,m)

    β  = norm(r)
    qk = copy(r)

    @views for k = 1:m
        q[:,k] = qk/β
        jacvec!(qk,f,x,q[:,k])
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

function arnoldi(a::AbstractMatrix{T},b::AbstractVector{T},tol::T,m::S=length(b)) where {T <: AbstractFloat, S <: Integer} #Inexact

    n = length(b)
    if m > n
        error("'m' must be no larger than the length of 'b' ")
    end

    q = zeros(T,n,m)
    h = zeros(T,m+1,m)

    β  = norm(b)
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

function arnoldi(f::Function,x::AbstractVector{T},r::AbstractVector{T},tol::T,m::S=length(x)) where {T <: AbstractFloat, S <: Integer} # Inexact

    n = length(x)
    if m > n
        error("'m' must be no larger than the length of 'b' ")
    end

    q = zeros(T,n,m)
    h = zeros(T,m+1,m)

    β  = norm(r)
    qk = copy(r)

    @views for k = 1:m
        q[:,k] = qk/β
        jacvec!(qk,f,x,q[:,k])
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

function arnoldi_inplace(f::Function,x::AbstractVector{T},r::AbstractVector{T},tol::T,m::S=length(x)) where {T <: AbstractFloat, S <: Integer} # Inexact

    n = length(x)
    if m > n
        error("'m' must be no larger than the length of 'b' ")
    end

    q = zeros(T,n,m)
    h = zeros(T,m+1,m)

    β  = norm(r)
    qk = copy(r)

    @views for k = 1:m
        q[:,k] = qk/β
        jacvec_inplace!(qk,f,x,q[:,k])
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