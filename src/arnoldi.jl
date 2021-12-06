function arnoldi(a::AbstractArray{T,2},b::AbstractArray{T,1},m::S = length(b)) where {T <: AbstractFloat, S <: Integer}

    n = length(b)
    if m > n
        error("'m' must be no larger than the length of 'b' ")
    end

    q = zeros(n,m+1)
    h = zeros(m+1,m)

    q[:,1] = b/norm(b)

    c = 0
    @views for k = 1:m
        c += 1
        qk = a*q[:,k]
        for j = 1:k
            h[j,k] = qk'q[:,j]
            qk -= h[j,k]*q[:,j]
        end
        h[k+1,k] = norm(qk)
        if abs(h[k+1,k]) < eps(T)
            return q[1:n,1:c], h[1:c,1:c]
        else
            q[:,k+1] = qk/h[k+1,k]
        end
    end

    return q[1:n,1:m], h[1:m,1:m]

end

function arnoldi(f::Function,x::Array{T,1},r::Array{T,1},m::S=length(x)) where {T <: AbstractFloat, S <: Integer}

    n = length(x)
    if m > n
        error("'m' must be no larger than the length of 'b' ")
    end

    q = zeros(n,m+1)
    h = zeros(m+1,m)
    jvec = zeros(n)

    q[:,1] = r/norm(r)

    c = 0
    @views for k = 1:m
        c += 1
        jacvec!(f,x,q[:,k],jvec)
        qk = jvec
        for j = 1:k
            h[j,k] = qk'q[:,j]
            qk -= h[j,k]*q[:,j]
        end
        h[k+1,k] = norm(qk)
        if abs(h[k+1,k]) < eps(T)
            return q[1:n,1:c], h[1:c,1:c]
        else
            q[:,k+1] = qk/h[k+1,k]
        end
    end

    return q[1:n,1:m], h[1:m,1:m]

end

function arnoldi(a::AbstractArray{T,2},b::AbstractArray{T,1},tol::T,m::S=length(b)) where {T <: AbstractFloat, S <: Integer}

    n = length(b)

    q = zeros(n,m+1)
    h = zeros(m+1,m)

    q[:,1] = b/norm(b)

    c = 0
    @views for k = 1:m
        c += 1
        qk = a*q[:,k]
        for j = 1:k
            h[j,k] = qk'q[:,j]
            qk -= h[j,k]*q[:,j]
        end
        h[k+1,k] = norm(qk)
        if abs(h[k+1,k]) < tol
            return q[:,1:c], h[1:c,1:c]
        else
            q[:,k+1] = qk/h[k+1,k]
        end
    end

    return q[:,1:c], h[1:c,1:c]

end

function arnoldi(f::Function,x::Array{T,1},r::Array{T,1},tol::T,m::S=length(x)) where {T <: AbstractFloat, S <: Integer}

    n = length(x)

    q = zeros(n,m+1)
    h = zeros(m+1,m)
    jvec = zeros(n)

    q[:,1] = r/norm(r)

    c = 0
    @views for k = 1:m
        c += 1
        jacvec!(f,x,q[:,k],jvec)
        qk = jvec
        for j = 1:k
            h[j,k] = qk'q[:,j]
            qk -= h[j,k]*q[:,j]
        end
        h[k+1,k] = norm(qk)
        if abs(h[k+1,k]) < tol
            return q[:,1:c], h[1:c,1:c]
        else
            q[:,k+1] = qk/h[k+1,k]
        end
    end

    return q[:,1:c], h[1:c,1:c]

end
