# try to boost the performance using the selective reorthogonalization 
# strategies for the Lanczos Algorithm. 


mutable struct ConjGradBoosted{T<:Number}
    r::Vector{T}
    rnew::Vector{T}
    d::Vector{T}
    A::Function
    x::Vector{T}

    b::AbstractArray{T}
    itr::Int64

    function ConjGradBoosted(A::Function, b, x0=nothing)
        
        return this
    end

    function ConjGradBoosted(A::AbstractArray, b::AbstractArray)
        
    end
    
end

function (this::ConjGradBoosted)()
    r = this.r
    if r == 0
        return 0 # The problem is solved already. 
    end
    A = this.A
    d = this.d
    Ad = A(d)

    a = dot(r, r)/dot(d, Ad)
    if a < 0 
        error("CG got a non-definite matrix")
    end

    this.x += a*d
    this.rnew = r - a*Ad                # update rnew 
    b = dot(this.rnew, this.rnew)/dot(r, r)
    # @assert abs(dot(rnew + β*d, Ad)) < 1e-8 "Not conjugate"
    this.d = this.rnew + b*d
    this.r = this.rnew                      # Override
    this.itr += 1 
    return norm(this.rnew)
end

function GetCurrentResidualNorm(this::ConjGradBoosted)
    return norm(this.r)
end

