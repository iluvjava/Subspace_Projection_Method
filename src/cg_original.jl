# A original flavor of the Conjugate Gradient Algorithm. 
# * It performs the Conjugate Gradient and record all the residuals vectors for 
#   computing the lanczos iterations. 

mutable struct CG
    r
    rnew
    d
    A::Function
    x
    b
    itr

    function CG(A::Function, b, x0=nothing)
        this = new()
        this.A = A
        this.x = x0 === nothing ? b .+ 0.1  : x0  # just to handle matrix A that has eigenvalue of exactly 1.
        this.r = b - A(this.x)
        this.rnew = similar(this.r)
        this.d = this.r
        this.itr = 0

        this.record_lanczos = false
        return this
    end

    function CG(A::AbstractArray, b::AbstractArray)
        return CG((x)->A*x, b)
    end
    
end

function (this::CG)()
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
    return convert(Float64, norm(this.rnew))
end

function GetCurrentResidualNorm(this::CG)
    return norm(this.r)
end

