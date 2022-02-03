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
    thetas::Vector{T}       # Eigen values. 


    function DynamicSymTridiagonal{T}(alpha::T) where {T<:Float64}
        this = new{T}()
        this.alphas = Vector{T}()
        push!(this.alphas, alpha)
        this.betas = Vector{T}()
        this.L = Vector{T}()
        this.U = Vector{T}()
        push!(this.U, alpha)
        this.k = 1
        return this
    end
    

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
    return this
end


"""
    Evaluate for an instance of the symmetric tridiagonal symmetric matrix for 
    a given shift value of x, which evalutes: det(A - xI) for the instance matrix. 
"""
function CharacteristicPoly(this::DynamicSymTridiagonal{T}, x::Number) where {T<:AbstractFloat}
    pPrevious = 1
    pNow = this.alphas[1] - x
    for j in 2:this.k
        pNew = (this.alphas[j] - x)*pNow - (this.betas[j - 1]^2)*pPrevious
        # Update
        pPrevious = pNow
        pNow = pNew
    end
return pNow end


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


### ---- Basic testing for the Shifted Characteristic Polynomial Evaluation: 

using LinearAlgebra, Logging

function Testit(n=10)
    mainDiag = rand(n)
    subDiag = rand(n - 1)
    dynamicT = DynamicSymTridiagonal(mainDiag[1])
    for Idx in 1:n - 1
        dynamicT(mainDiag[Idx + 1], subDiag[Idx])
    end
    @info "Characteristic poly without shift evaluated to be: "
    println(CharacteristicPoly(dynamicT, 0))
    
return GetT(dynamicT) end

T = Testit()