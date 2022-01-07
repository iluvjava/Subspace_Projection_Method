# Dynamically keep track of the eigenvalues system of a hermitian, positive 
# definite matrix that is growing diagonally. 
#   * We need to keep track of its inverse, which is not going to be very stable numerically, but it should do the trick
#   of tracking eigenvalues using the Cauchy Interlace Theorem. 

mutable struct DynamicSymTridiagonal{T<:AbstractFloat}
    alphas::Vector{T}       # Diaognal
    betas::Vector{T}        # Lower & upper Diaognal
    L::Vector{T}            # The lower diagonal of the unit-bidiagonal matrix L
    U::Vector{T}            # Lower diagonal of the upper bi-diagonal matrix U
    k::Int64                # Size of the matrix. 

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
    append an alpha and a beta (diagonal and sub & super diagonal) elemnt to the current sym diagonal matrix. 
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
    Get the L matrix for this instance. 
"""
function GetL(this::DynamicSymTridiagonal)
    return Bidiagonal(fill(1, this.k), convert(Vector{Float64}, this.L), :L) 
end


"""
    Get the L matrix for this instance. 
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


"""
    Solve this matrix against another matrix or vector. 
"""
function Base.:\(
    this::DynamicSymTridiagonal{T}, 
    b::Union{Matrix{T}, Vector{T}}
    ) where {T<:AbstractFloat}
    
    return 
end


"""
    Multiply a vector on the left hand size of this dynamically growing matrix 
    T. 
"""
function Base.*(this::DynamicSymTridiagonal{T}, b::Vector{T}) where {T <: AbstractFloat}
    
end


# BASIC TESTING ----------------------------------------------------------------
using LinearAlgebra
n = 128
A = SymTridiagonal(rand(n), rand(n - 1))
T = DynamicSymTridiagonal(A[1, 1])
for Idx in 2: n
    T(A[Idx, Idx], A[Idx - 1, Idx])
end

L = GetL(T)
U = GetU(T)
@assert norm(GetT(T) - L*U) < 1e-10