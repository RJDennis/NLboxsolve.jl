function jacvec(f::Function,x::Array{T,1},v::Array{T,1}) where {T <: AbstractFloat}

    #ep = sqrt(1.0+norm(x)*eps(T))/norm(v)
    #jvec = (f(x+ep*v) - f(x-eps*v))/(2*ep)

    ep = 1e-10
    jvec = imag(f(x+ep*im*v))/ep

    return jvec

end

function jacvec!(f::Function,x::Array{T,1},v::AbstractArray{T,1},jvec::Array{T,1}) where {T <: AbstractFloat}

    #ep = sqrt(1.0+norm(x)*eps(T))/norm(v)
    #jvec .= (f(x+ep*v) - f(x-ep*v))/(2*ep)

    ep = 1e-10
    jvec .= imag(f(x+ep*im*v))/ep

end
