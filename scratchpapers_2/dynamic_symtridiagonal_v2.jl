### A symmetric tridiagonal matrix that supports appending onto the bottom 
### right corner of the matrix and keeping track of eigenvalues of the matrix. 
### * Support dynamic LU decomposition in real time, without pivoting. 
### * Fast characteristic polynomial computations for shifted system of this 
###   matrix. 

mutable struct DynamicSymTridiagonal{T<:AbstractFloat}
    # Parameters groupd 1
    alphas::Vector{T}       # Diaognal
    betas::Vector{T}        # Lower & upper Diaognal
    L::Vector{T}            # The lower diagonal of the unit-bidiagonal matrix L
    U::Vector{T}            # Lower diagonal of the upper bi-diagonal matrix U
    k::Int64                # Size of the matrix. 

    # Parameters group 2 
    last_update::Int64          # last iteration where eigenvalues are updated for this matrix.   
    thetas::Vector{T}           # Ritz values. 
    converged::Vector{Bool}     # Indicates convergence for matching index ritz value from last iteration.
    
    function DynamicSymTridiagonal{T}(alpha::T) where {T<:Float64}
        this = new{T}()
        this.alphas = Vector{T}()
        push!(this.alphas, alpha)
        this.betas = Vector{T}()
        this.L = Vector{T}()
        this.U = Vector{T}()
        push!(this.U, alpha)
        this.k = 1
        this.last_update = 0
    return this end

    function DynamicSymTridiagonal(alpha::AbstractFloat)
        T = typeof(alpha)
        return DynamicSymTridiagonal{T}(alpha)
    end
end


"""
    Opverting the opertor (alpha,beta) to append new element to the matrix. 
    append an alpha and a beta (diagonal and sub & super diagonal) elemnt to
    the current sym diagonal matrix. 
"""
function (this::DynamicSymTridiagonal{T})(alpha::T, beta::T) where {T<:AbstractFloat}
    push!(this.alphas, alpha)
    push!(this.betas, beta)
    push!(this.U, alpha - beta^2/this.U[end])
    push!(this.L, beta/this.U[end - 1])
    this.k += 1
return this end


"""
    Evaluate for an instance of the symmetric tridiagonal symmetric matrix for 
    a given shift value of x, which evalutes: det(A - xI) for the instance matrix. 
    
    * It will return Nan if the polynomial exploded. 
"""
function CharPolyShifted(this::DynamicSymTridiagonal{T}, x::Number=0.0) where {T<:AbstractFloat}
    pPrevious = 1
    pNow = this.alphas[1] - x
    for j in 2:this.k
        pNew = (this.alphas[j] - x)*pNow - (this.betas[j - 1]^2)*pPrevious
        pPrevious = pNow
        pNow = pNew
    end
return pNow end


"""
    Evalute the derivetive of the characteristic polynoial for a shifted quantity x.
    det(T - xI)
"""
function CharacteristicPolyDerivative(this::DynamicSymTridiagonal{T}, x::Number) where {T<:AbstractFloat}
    error("I haven't implement it yet. ")
return end


"""
    Locate the eigenvalue given only the left bound on the eigenvalue. 
    * Eigenvalue CANNOT be on the boundary that passed in, they might cause 
    sign problem violating the bisection pre-conditions. 
"""
function EigenValueLocate(
    this::DynamicSymTridiagonal{T},
    left_bound::T,
    right_bound::T
) where {T<:AbstractFloat}
    @assert !(isnan(left_bound) || isnan(right_bound)) "Any of the bound for "*
    "the eigenvalue for dynamic tridiagonal symmetric matrix cannot be nan. "
    @assert !(isinf(left_bound) && isinf(right_bound)) "Both left and "*
    "right_bound cannot be inf. "
    @assert left_bound < right_bound "The left boundary should be less than the"*
    "right boundary, this is by the definition of interval. "

    # perform search when one of the bound is infinity. 
    P(x) = CharPolyShifted(this, x)
    if isinf(left_bound) || isinf(right_bound)
        stepSize = isinf(left_bound) ? -1e-8 : 1e-8
        if stepSize > 0
            while sign(P(left_bound + stepSize)) == sign(P(left_bound))
                    stepSize += 2*stepSize
            end
        else
            while sign(P(right_bound + stepSize)) == sign(P(right_bound))
                stepSize += 2*stepSize
            end
        end
        if isinf(left_bound)
            left_bound = right_bound + stepSize
        else
            right_bound = left_bound + stepSize
        end
    end
    
    # test preconditions for bisection
    pL = P(left_bound); pR = P(right_bound)
    @assert sign(pL) != sign(pR) "Bisection search error;"*
    "the left and right bondary turns out to have the same sign, please check the "*
    "caller routine which is responsible. "
    @assert !isnan(pL) && !isnan(pR) "Polynomial exploded with Nan on one of "*
    "the boundaries OR both boundaries after expontneial probing "*
    "(or without the probing). Please refine search interval. "

    # perform bisection search to locate a root. 
    midX = (left_bound + right_bound)/2
    midP = P(midX)
    itrLimit = 100
    while midP != 0 && 
        itrLimit > 0 && 
        right_bound != midX && 
        left_bound != midX

        if sign(midP) == sign(P(left_bound))
            left_bound = midX
        else 
            right_bound = midX
        end
        itrLimit -= 1
        midX = (left_bound + right_bound)/2
        midP = P(midX)
        println("Bisection: [$left_bound, $right_bound]")
    end

return midX end

"""
    Update the eigenvalues based on all the eigenvalues 
    from the previous sub-matrix. 
        * If current eigenvalues are not from the previous iterations, this 
        will give an error, it can only compute eigenvalues for this iteration
        based on the eigenvalues from the previous iterations. 
"""
function EigenvaluesUpdate(this::DynamicSymTridiagonal{T}) where {T <: AbstractFloat}
    if this.k == 1  # current matrix is 1 x 1. 
        EstablishEigenSystem(this)
        this.last_update = this.k

        return
    end

    if this.k == this.last_update  # eigen system already updated.
        return
    elseif this.k - this.last_update == 1
        # update the eigensystem using the interlace properties and search routine. 
        thetas = this.thetas
        newThetas = Array{T}(undef, size(thetas) + 1)
        push!(thetas, convert(T, Inf64))
        pushfirst!(thetas, Convert(T, -Inf64))

        for (i, w) in enumerate(view(thetas, 2, length(thetas) - 1))
            if i <= ceil(i/2)|>Int64
                
            else

            end
        end

    else
        # Maybeshould consdier re-establishing the system. 
        error("Must updated after each iterations, or else we lost track of the eigenvalues. ")
    end

return end

"""
    Function establish the eigen system of the Tridiagonal Matrix using Lapack 
    Routine, and erase all running parameters about the eigensystem. 

    * Compute all the eigenvalues of the tridiagonal system and mark all as
    unconvergend
"""
function EstablishEigenSystem(this::DynamicSymTridiagonal{T}) where {T<:AbstractFloat}
    TMatrix = GetT(this)
    eigenValues = sort!(eigenvals(TMatrix))
    this.thetas = eigenValues
    this.betas = fill(true, length(eigenValues))
return end


"""
    Get the L matrix for this instance. 
"""
function GetL(this::DynamicSymTridiagonal)
    return Bidiagonal(fill(1, this.k), convert(Vector{Float64}, this.L), :L) 
end


"""
    Get the U matrix for this instance. 
"""
function GetU(this::DynamicSymTridiagonal)
    return Bidiagonal(convert(Vector{Float64}, this.U), this.betas, :U) 
end


"""
    Get the Tridiagonal Matrix 
"""
function GetT(this::DynamicSymTridiagonal)
    return SymTridiagonal(this.alphas, this.betas)
end


