function gmres(a::AbstractArray{T,2},b::AbstractArray{T,1},x::AbstractArray{T,1}) where {T <: AbstractFloat} # Exact GMRES

    # Create Hessenberg form using Arnoldi iteration
    q, h = arnoldi(a,b)
    
    c = size(h,1)

    e1    = zeros(c)
    e1[1] = norm(b)

    # Use Givens rotations to transform to upper-triangular form
    for i = 2:c
        g = givens(h[i-1:i,i-1])
        h[i-1:i,:] = g'h[i-1:i,:]
        e1[i-1:i] = g'e1[i-1:i]
    end

    step = q*(h\e1)

    return step

end

function gmres(a::AbstractArray{T,2},b::AbstractArray{T,1},x::AbstractArray{T,1},forcing_term::T) where {T <: AbstractFloat} # Inexact GMRES

    # Create Hessenberg form using Arnoldi iteration
    q, h = arnoldi(a,b,forcing_term*norm(b))
    
    c = size(h,1)

    e1    = zeros(c)
    e1[1] = norm(b)

    # Use Givens rotations to transform to upper-triangular form
    for i = 2:c
        g = givens(h[i-1:i,i-1])
        h[i-1:i,:] = g'h[i-1:i,:]
        e1[i-1:i] = g'e1[i-1:i]
    end

    step = q*(h\e1)

    return step

end

function gmres(a::AbstractArray{T,2},b::AbstractArray{T,1},x::AbstractArray{T,1},forcing_term::T,m_max::S) where {T <: AbstractFloat, S <: Integer} # Restarted inexact GMRES

    x = copy(x)
    n = length(x)
    
    m = min(n,m_max)

    while true

        # Create Hessenberg form using Arnoldi iteration
        q, h = arnoldi(a,b,forcing_term*norm(b),m)
    
        c = size(h,1)

        e1    = zeros(c)
        e1[1] = norm(b)

        # Use Givens rotations to transform to upper-triangular form
        for i = 2:c
            g = givens(h[i-1:i,i-1])
            h[i-1:i,:] = g'h[i-1:i,:]
            e1[i-1:i] = g'e1[i-1:i]
        end

        step = q*(h\e1)

        if maximum(abs,a*step-b) < forcing_term*norm(b)
            return step
        else
            x .= x + step
        end
    end

end

function gmres(f::Function,x::Array{T,1}) where {T <: AbstractFloat} # Exact GMRES

    n = length(x)
    a = ForwardDiff.jacobian(f,x)
    b = -f(x)

    # Create Hessenberg form using Arnoldi iteration
    q, h = arnoldi(a,b)
    
    c = size(h,1)

    e1    = zeros(c)
    e1[1] = norm(b)

    # Use Givens rotations to transform to upper-triangular form
    for i = 2:c
        g = givens(h[i-1:i,i-1])
        h[i-1:i,:] = g'h[i-1:i,:]
        e1[i-1:i] = g'e1[i-1:i]
    end

    step = q*(h\e1)

    return step

end

function gmres(f::Function,x::Array{T,1},forcing_term::T) where {T <: AbstractFloat} # Inexact GMRES

    n = length(x)
    a = ForwardDiff.jacobian(f,x)
    b = -f(x)

    # Create Hessenberg form using Arnoldi iteration
    q, h = arnoldi(a,b,forcing_term*norm(b))
    
    c = size(h,1)

    e1    = zeros(c)
    e1[1] = norm(b)

    # Use Givens rotations to transform to upper-triangular form
    for i = 2:c
        g = givens(h[i-1:i,i-1])
        h[i-1:i,:] = g'h[i-1:i,:]
        e1[i-1:i] = g'e1[i-1:i]
    end

    step = q*(h\e1)

    return step

end

function gmres(f::Function,x::Array{T,1},forcing_term::T,m_max::S) where {T <: AbstractFloat, S <: Integer} # Restarted inexact GMRES

    x = copy(x)
    n = length(x)
    
    m = min(n,m_max)

    while true

        a = ForwardDiff.jacobian(f,x)
        b = -f(x)

        # Create Hessenberg form using Arnoldi iteration
        q, h = arnoldi(a,b,forcing_term*norm(b),m)
    
        c = size(h,1)

        e1    = zeros(c)
        e1[1] = norm(b)

        # Use Givens rotations to transform to upper-triangular form
        for i = 2:c
            g = givens(h[i-1:i,i-1])
            h[i-1:i,:] = g'h[i-1:i,:]
            e1[i-1:i] = g'e1[i-1:i]
        end

        step = q*(h\e1)

        if maximum(abs,a*step-b) < forcing_term*norm(b)
            return step
        else
            x .= x + step
        end
    end

end

function jacobian_free_gmres(f::Function,x::Array{T,1}) where {T <: AbstractFloat} # Jacobian-free exact GMRES

    b = -f(x)

    # Create Hessenberg form using Arnoldi iteration
    q, h = arnoldi(f,x,b)
    
    c = size(h,1)

    e1    = zeros(c)
    e1[1] = norm(b)

    # Use Givens rotations to transform to upper-triangular form
    for i = 2:c
        g = givens(h[i-1:i,i-1])
        h[i-1:i,:] = g'h[i-1:i,:]
        e1[i-1:i] = g'e1[i-1:i]
    end

    step = q*(h\e1)

    return step

end

function jacobian_free_gmres(f::Function,x::Array{T,1},forcing_term::T) where {T <: AbstractFloat} # Jacobian-free inexact GMRES

    b = -f(x)

    # Create Hessenberg form using Arnoldi iteration
    q, h = arnoldi(f,x,b,forcing_term*norm(b))
    
    c = size(h,1)

    e1    = zeros(c)
    e1[1] = norm(b)

    # Use Givens rotations to transform to upper-triangular form
    for i = 2:c
        g = givens(h[i-1:i,i-1])
        h[i-1:i,:] = g'h[i-1:i,:]
        e1[i-1:i] = g'e1[i-1:i]
    end

    step = q*(h\e1)

    return step

end

function jacobian_free_gmres(f::Function,x::Array{T,1},forcing_term::T,m_max::S) where {T <: AbstractFloat, S <: Integer} # Jacobian-free restarted inexact GMRES

    x = copy(x)
    n = length(x)
    jvec = zeros(n)

    m = min(n,m_max)

    while true

        b = -f(x)

        # Create Hessenberg form using Arnoldi iteration
        q, h = arnoldi(f,x,b,forcing_term*norm(b),m)
    
        c = size(h,1)

        e1    = zeros(c)
        e1[1] = norm(b)

        # Use Givens rotations to transform to upper-triangular form
        for i = 2:c
            g = givens(h[i-1:i,i-1])
            h[i-1:i,:] = g'h[i-1:i,:]
            e1[i-1:i] = g'*e1[i-1:i]
        end

        step = q*(h\e1)

        jacvec!(f,x,step,jvec)

        if maximum(abs,jvec-b) < forcing_term*norm(b)
            return step
        else
            x .= x + step
        end
    end

end
