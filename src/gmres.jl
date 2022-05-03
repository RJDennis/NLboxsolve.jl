function gmres(a::AbstractArray{T,2},b::AbstractArray{T,1},x::AbstractArray{T,1}) where {T <: AbstractFloat} # Exact GMRES

    # Create Hessenberg form using Arnoldi iteration
    q, h, c = arnoldi(a,b)
    
    e1    = zeros(c)
    e1[1] = norm(b)

    g = zeros(T,2,2)
    # Use Givens rotations to transform to upper-triangular form
    @views for i = 2:c
        givens!(g,h[i-1:i,i-1])
        h[i-1:i,:] = g'h[i-1:i,:]
        e1[i-1:i] = g'e1[i-1:i]
    end

    step = q*(h\e1)

    return step

end

function gmres(a::AbstractArray{T,2},b::AbstractArray{T,1},x::AbstractArray{T,1},forcing_term::T) where {T <: AbstractFloat} # Inexact GMRES

    # Create Hessenberg form using Arnoldi iteration
    q, h, c = arnoldi(a,b,forcing_term*norm(b))
    
    e1    = zeros(c)
    e1[1] = norm(b)

    g = Array{T,2}(undef,2,2)
    # Use Givens rotations to transform to upper-triangular form
    @views for i = 2:c
        givens!(g,h[i-1:i,i-1])
        h[i-1:i,:] = g'h[i-1:i,:]
        e1[i-1:i] = g'e1[i-1:i]
    end

    step = q*(h\e1)

    return step

end

function gmres(a::AbstractArray{T,2},b::AbstractArray{T,1},x::AbstractArray{T,1},forcing_term::T,m::S) where {T <: AbstractFloat, S <: Integer} # Restarted inexact GMRES

    x = copy(x)
    n = length(x)

    m = min(m,n)
    
    step = Array{T,1}(undef,n)

    g = Array{T,2}(undef,2,2)

    loop_max = ceil(Int,n/m)
    loop_count = 1
    while loop_count <= loop_max 

        # Create Hessenberg form using Arnoldi iteration
        q, h, c = arnoldi(a,b,forcing_term*norm(b),m)
    
        e1    = zeros(c)
        e1[1] = norm(b)

        # Use Givens rotations to transform to upper-triangular form
        @views for i = 2:c
            givens!(g,h[i-1:i,i-1])
            h[i-1:i,:] = g'h[i-1:i,:]
            e1[i-1:i] = g'e1[i-1:i]
        end

        step .= q*(h\e1)

        if maximum(abs,a*step-b) < forcing_term*norm(b)
            return step, true
        else
            x .= x + step
        end
        loop_count += 1
    end

    return step, false

end

function gmres(f::Function,x::Array{T,1}) where {T <: AbstractFloat} # Exact GMRES

    n = length(x)
    a = ForwardDiff.jacobian(f,x)
    b = -f(x)

    # Create Hessenberg form using Arnoldi iteration
    q, h, c = arnoldi(a,b)
    
    e1    = zeros(c)
    e1[1] = norm(b)

    g = Array{T,2}(undef,2,2)
    # Use Givens rotations to transform to upper-triangular form
    @views for i = 2:c
        givens!(g,h[i-1:i,i-1])
        h[i-1:i,:] = g'h[i-1:i,:]
        e1[i-1:i] = g'e1[i-1:i]
    end

    step = q*(h\e1)

    return step

end

function gmres_sparse(f::Function,x::Array{T,1}) where {T <: AbstractFloat} # Exact GMRES

    n = length(x)
    a = sparse(ForwardDiff.jacobian(f,x))
    b = -f(x)

    # Create Hessenberg form using Arnoldi iteration
    q, h, c = arnoldi(a,b)
    
    e1    = zeros(c)
    e1[1] = norm(b)

    g = Array{T,2}(undef,2,2)
    # Use Givens rotations to transform to upper-triangular form
    @views for i = 2:c
        givens!(g,h[i-1:i,i-1])
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
    q, h, c = arnoldi(a,b,forcing_term*norm(b))
    
    e1    = zeros(c)
    e1[1] = norm(b)

    g = Array{T,2}(undef,2,2)
    # Use Givens rotations to transform to upper-triangular form
    @views for i = 2:c
        givens!(g,h[i-1:i,i-1])
        h[i-1:i,:] = g'h[i-1:i,:]
        e1[i-1:i] = g'e1[i-1:i]
    end

    step = q*(h\e1)

    return step

end

function gmres_sparse(f::Function,x::Array{T,1},forcing_term::T) where {T <: AbstractFloat} # Inexact GMRES

    n = length(x)
    a = sparse(ForwardDiff.jacobian(f,x))
    b = -f(x)

    # Create Hessenberg form using Arnoldi iteration
    q, h, c = arnoldi(a,b,forcing_term*norm(b))
    
    e1    = zeros(c)
    e1[1] = norm(b)

    g = Array{T,2}(undef,2,2)
    # Use Givens rotations to transform to upper-triangular form
    @views for i = 2:c
        givens!(g,h[i-1:i,i-1])
        h[i-1:i,:] = g'h[i-1:i,:]
        e1[i-1:i] = g'e1[i-1:i]
    end

    step = q*(h\e1)

    return step

end

function gmres(f::Function,x::Array{T,1},forcing_term::T,m::S) where {T <: AbstractFloat, S <: Integer} # Restarted inexact GMRES

    x = copy(x)
    n = length(x)

    step = similar(x)

    m = min(m,n)

    j = Array{T,2}(undef,n,n)
    b = Array{T,1}(undef,n)

    g = Array{T,2}(undef,2,2)

    loop_max = ceil(Int,n/m)
    loop_count = 1
    while loop_count <= loop_max 

        j .= ForwardDiff.jacobian(f,x)
        b .= -f(x)

        # Create Hessenberg form using Arnoldi iteration
        q, h, c = arnoldi(j,b,forcing_term*norm(b),m)
    
        e1    = zeros(c)
        e1[1] = norm(b)

        # Use Givens rotations to transform to upper-triangular form
        @views for i = 2:c
            givens!(g,h[i-1:i,i-1])
            h[i-1:i,:] = g'h[i-1:i,:]
            e1[i-1:i] = g'e1[i-1:i]
        end

        step .= q*(h\e1)

        if maximum(abs,j*step-b) < forcing_term*norm(b)
            return step, true
        else
            x .= x + step
        end
        loop_count +=1
    end

    return step, false

end

function gmres_sparse(f::Function,x::Array{T,1},forcing_term::T,m::S) where {T <: AbstractFloat, S <: Integer} # Restarted inexact GMRES

    x = copy(x)
    n = length(x)

    step = similar(x)

    m = min(m,n)

    j = sparse(Array{T,2}(undef,n,n))
    b = Array{T,1}(undef,n)

    g = Array{T,2}(undef,2,2)

    loop_max = ceil(Int,n/m)
    loop_count = 1
    while loop_count <= loop_max 

        j .= sparse(ForwardDiff.jacobian(f,x))
        b .= -f(x)

        # Create Hessenberg form using Arnoldi iteration
        q, h, c = arnoldi(j,b,forcing_term*norm(b),m)
    
        e1    = zeros(c)
        e1[1] = norm(b)

        # Use Givens rotations to transform to upper-triangular form
        @views for i = 2:c
            givens!(g,h[i-1:i,i-1])
            h[i-1:i,:] = g'h[i-1:i,:]
            e1[i-1:i] = g'e1[i-1:i]
        end

        step .= q*(h\e1)

        if maximum(abs,j*step-b) < forcing_term*norm(b)
            return step, true
        else
            x .= x + step
        end
        loop_count +=1
    end

    return step, false

end

function jacobian_free_gmres(f::Function,x::Array{T,1}) where {T <: AbstractFloat} # Jacobian-free exact GMRES

    b = -f(x)

    # Create Hessenberg form using Arnoldi iteration
    q, h, c = arnoldi(f,x,b)
    
    e1    = zeros(c)
    e1[1] = norm(b)

    g = Array{T,2}(undef,2,2)
    # Use Givens rotations to transform to upper-triangular form
    @views for i = 2:c
        givens!(g,h[i-1:i,i-1])
        h[i-1:i,:] = g'h[i-1:i,:]
        e1[i-1:i] = g'e1[i-1:i]
    end

    step = q*(h\e1)

    return step

end

function jacobian_free_gmres(f::Function,x::Array{T,1},forcing_term::T) where {T <: AbstractFloat} # Jacobian-free inexact GMRES

    b = -f(x)

    # Create Hessenberg form using Arnoldi iteration
    q, h, c = arnoldi(f,x,b,forcing_term*norm(b))
    
    e1    = zeros(c)
    e1[1] = norm(b)

    g = Array{T,2}(undef,2,2)
    # Use Givens rotations to transform to upper-triangular form
    @views for i = 2:c
        givens!(g,h[i-1:i,i-1])
        h[i-1:i,:] = g'h[i-1:i,:]
        e1[i-1:i] = g'e1[i-1:i]
    end

    step = q*(h\e1)

    return step

end

function jacobian_free_gmres(f::Function,x::Array{T,1},forcing_term::T,m::S) where {T <: AbstractFloat, S <: Integer} # Jacobian-free restarted inexact GMRES

    x = copy(x)
    n = length(x)
    jvec = zeros(n)
    step = similar(x)

    m = min(m,n)

    b = Array{T,1}(undef,n)

    g = Array{T,2}(undef,2,2)
    
    loop_max = ceil(Int,n/m)
    loop_count = 1
    while loop_count <= loop_max 

        b .= -f(x)

        # Create Hessenberg form using Arnoldi iteration
        q, h, c = arnoldi(f,x,b,forcing_term*norm(b),m)
    
        e1    = zeros(c)
        e1[1] = norm(b)

        # Use Givens rotations to transform to upper-triangular form
        @views for i = 2:c
            givens!(g,h[i-1:i,i-1])
            h[i-1:i,:] = g'h[i-1:i,:]
            e1[i-1:i] = g'*e1[i-1:i]
        end

        step .= q*(h\e1)

        jacvec!(jvec,f,x,step)

        if maximum(abs,jvec-b) < forcing_term*norm(b)
            return step, true
        else
            x .= x + step
        end
        loop_count += 1
    end

    return step, false

end

function jacobian_free_gmres_inplace(f::Function,x::Array{T,1},forcing_term::T,m::S) where {T <: AbstractFloat, S <: Integer} # Jacobian-free restarted inexact GMRES

    x = copy(x)
    n = length(x)
    jvec = zeros(n)
    step = similar(x)

    m = min(m,n)

    b   = Array{T,1}(undef,n)
    ffk = Array{T,1}(undef,n)

    g = Array{T,2}(undef,2,2)
    
    loop_max = ceil(Int,n/m)
    loop_count = 1
    while loop_count <= loop_max 

        f(ffk,x)
        b .= -ffk

        # Create Hessenberg form using Arnoldi iteration
        q, h, c = arnoldi_inplace(f,x,b,forcing_term*norm(b),m)
    
        e1    = zeros(c)
        e1[1] = norm(b)

        # Use Givens rotations to transform to upper-triangular form
        @views for i = 2:c
            givens!(g,h[i-1:i,i-1])
            h[i-1:i,:] = g'h[i-1:i,:]
            e1[i-1:i] = g'*e1[i-1:i]
        end

        step .= q*(h\e1)

        jacvec_inplace!(jvec,f,x,step)

        if maximum(abs,jvec-b) < forcing_term*norm(b)
            return step, true
        else
            x .= x + step
        end
        loop_count += 1
    end

    return step, false

end
