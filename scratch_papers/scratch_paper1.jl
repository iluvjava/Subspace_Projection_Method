# Question to answer, what if you orthogonalize the residual vectors against all
# historical residual vectors??? 


# ------------------------------------------------------------------------------
# A original flavor of the Conjugate Gradient Algorithm. 
# * It performs the Conjugate Gradient and record all the residuals vectors for 
#   computing the lanczos iterations. 

using LinearAlgebra
using Logging
using Plots


mutable struct CG
    r
    rnew
    d
    A::Function
    x
    b
    itr

    residuals
    reorthogonalize::Bool
    function CG(A::Function, b, x0=nothing)
        this = new()
        this.A = A
        this.x = x0 === nothing ? b .+ 0.1  : x0  # just to handle matrix A that has eigenvalue of exactly 1.
        this.r = b - A(this.x)
        this.rnew = similar(this.r)
        this.d = this.r
        this.itr = 0
        this.residuals = Vector{typeof(this.r)}()
        push!(this.residuals, this.r)
        this.reorthogonalize = true
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
    if this.reorthogonalize
        for residual in this.residuals
            this.rnew -= (dot(residual, this.rnew)/dot(residual, residual))*residual
        end
        push!(this.residuals, this.rnew)
    end
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

"""
    Turn on the reorthogonalizations using the residual vectors, 
    and add current residual to the list of residuals. 
"""
function TurnOnReorthgonalize(this::CG)
    this.reorthogonalize = true
    push!(this.residuals, this.r)
    return 
end

"""
    Turn off the reorthogonalization on the residual vectors. And then 
    clear all the stored residual vectors. 
"""
function TurnOffReorthgonalize(this::CG)
    this.reorthogonalize = false
    empty!(this.residuals)
    return
end

### -----------------------Testing Utility -------------------------------------


"""
    Accept a linear system, and a list of guesses from the Lanczos Algorithm, 
    it will return a list energy norm of the 
    error vector for the linear system. 
    
    Parameters: 
        A: The matrix
        b: The vector on the RHS
        Xs: All the guesses vector from the CG Algorithm, including the initial guess! 
"""
function EnergyErrorNorm(A, b, Xs)
    XStar = A\b
    InitialError = dot(Xs[1] - XStar, A, Xs[1] - XStar)
    err = (x) -> dot(x - XStar, A, x - XStar)/InitialError

    return err.(Xs)
end


"""
    Bad eigen value distribution, they are distributed from 0.001 to 1 like a 
    geometric series. 
     
    * A lot of small eigenvalues are clustered close to zero, few larger ones are 
    far away from the other Eigenvalues. 

    Parameters:
        rho: Number: 
            An number between 0 and 1 that paramaterize the distribution of the 
            eigenvalues for the PSD matrix. 
        N=20: 
            The size of the PSD matrix.
    
    returns: 
        a diagonal matrix. 
"""
function GetNastyPSDMatrix(rho::Number, N=20, inverted=false)
    @assert rho <= 1 && rho >= 0
    EigenValues = zeros(N)
    EigenMin, EigenMax = 0.001, 1    # Min Max Eigenvalues. 
    EigenValues[1] = EigenMin
    for IdexI in 2:N
        EigenValues[IdexI] = EigenMin + 
            ((IdexI - 1)/(N - 1))*(EigenMax - EigenMin)*rho^(N - IdexI)  # formulas
    end
    if inverted
        return I - diagm(EigenValues)
    end
    return diagm(EigenValues)
end



N = 100
A = SymTridiagonal(randn(N), randn(N - 1))
A = A'*A
b = rand(N)
cg1 = CG(A, b)
cg2 = CG(A, b)
cg1Soln = Vector(); push!(cg1Soln, cg1.x)
cg2Soln = Vector(); push!(cg2Soln, cg2.x)

TurnOffReorthgonalize(cg2)
for itr in 1: 1.5*N
    Res2Norm1 = cg1()
    push!(cg1Soln, cg1.x)
    Res2Norm2 = cg2()
    push!(cg2Soln, cg2.x)
    println("With Reorthogonalization: $Res2Norm1, Without: $Res2Norm2")
end


relError1 = EnergyErrorNorm(A, b, cg1Soln)
relError2 = EnergyErrorNorm(A, b, cg2Soln)

fig1 = plot(1:length(relError1), log10.(relError1), label="With Re-Orthogonalization", title="CG Relative Error")
plot!(fig1, 1:length(relError2), log10.(relError2), label="Without Re-Orthogonalization")
