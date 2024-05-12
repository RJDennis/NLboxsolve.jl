function jacvec(f::Function,x::Array{T,1},v::Array{T,1}) where {T <: AbstractFloat}

    ep = 1e-10
    jvec = imag(f(x+ep*im*v))/ep

    return jvec

end

function jacvec!(jvec::Array{T,1},f::Function,x::Array{T,1},v::AbstractArray{T,1}) where {T <: AbstractFloat}

    ep = 1e-10
    jvec .= imag(f(x+ep*im*v))/ep

end

function jacvec_inplace!(jvec::Array{T,1},f::Function,x::Array{T,1},v::AbstractArray{T,1}) where {T <: AbstractFloat}

    n = length(x)
    ffk = Array{Complex{T},1}(undef,n)

    ep = 1e-10
    f(ffk,x+ep*im*v)

    jvec .= imag(ffk)/ep

end