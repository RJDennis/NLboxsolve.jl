function jacvec(f::Function,x::AbstractVector{T},v::AbstractVector{T}) where {T <: AbstractFloat}

    ep = 1e-10
    jvec = imag(f(x+ep*im*v))/ep

    return jvec

end

function jacvec!(jvec::AbstractVector{T},f::Function,x::AbstractVector{T},v::AbstractVector{T}) where {T <: AbstractFloat}

    ep = 1e-10
    jvec .= imag(f(x+ep*im*v))/ep

end

function jacvec_inplace!(jvec::AbstractVector{T},f::Function,x::AbstractVector{T},v::AbstractVector{T}) where {T <: AbstractFloat}

    n = length(x)
    Fk = Vector{Complex{T}}(undef,n)

    ep = 1e-10
    f(Fk,x+ep*im*v)

    jvec .= imag(Fk)/ep

end