# Dynamically keep track of the eigenvalues system of a hermitian, positive 
# definite matrix that is growing diagonally. 
#   * We need to keep track of its inverse, which is not going to be very stable numerically, but it should do the trick
#   of tracking eigenvalues using the Cauchy Interlace Theorem. 
#   * We also want to keep track of it's eigensystem, which is going to serve for the Lanczos Algorithm with selective 
#   re-orthogonalizations! 
#   * It starts keeping track of the eigensystem when it's been asked to, and it will start doing it. 

mutable struct DynamicSymTridiagonal{T<:AbstractFloat}
    alphas::Vector{T}       # Diaognal
    betas::Vector{T}        # Lower & upper Diaognal
    L::Vector{T}            # The lower diagonal of the unit-bidiagonal matrix L
    U::Vector{T}            # Lower diagonal of the upper bi-diagonal matrix U
    k::Int64                # Size of the matrix. 

    V::Matrix{T}            # Eigenvectors
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
    Using the Lapack library to get the eigen system of the curent T, and then after 
    this, it will keep track of the eigensystem as elements are added to the dynamic matrix. 
"""
function InitializeEigenSystem(this::DynamicSymTridiagonal{T}, max_size::Int64) where {T<:AbstractFloat}
    if max_size > 32728 || max_size < 2
        error("Eigen system estimated max size can't be: $(max_size)")
    end
    Trid = GetT(this)
    theta, V = eigen(Trid)
    Vbigger = zeros(T, max_size, max_size)
    Vbigger[1:this.k, 1:this.k] = V
    this.V = Vbigger
    Theta = zeros(T, max_size)
    Theta[1:this.k] = theta
    this.thetas = Theta
    return this
end

"""
    Update the eigensystem of the bigger matrix using the preivous eigen system. Use Power Iterations 
    iteratively. 
"""
function UpdateEigenSystem(this::DynamicSymTridiagonal{T}) where {T <: AbstractFloat}
    # if !(isdefined(Base, :this.V))  # cheks if defined first. 
    #     error("Must initialize the eigensystem with an estimated size before calling this function. ")
    # end
    k = this.k
    V = this.V
    Θ = this.thetas
    A = this
    if k > size(this.V, 1)
        error("Hasn't implemented this part yet, k exceed the maximal size of eigen system.")
    end

    for j in 1: k
        # if j != k
        #     vⱼ = view(V, 1:k, j) .+ 1e-6*rand(T, k)
        # else
        #     vⱼ = rand(T, k)
        # end
        vⱼ = rand(T, k)
        if j >= 2   # Components on to previous ortho eigen vectors removed. 
            vⱼ .-= view(V, 1:k, 1:j - 1)*view(V, 1:k, 1:j - 1)'*vⱼ
        end
        for _ in 1: k^2             # Power iterations
            Avⱼ = A*vⱼ
            vⱼᵀvⱼ = dot(vⱼ, vⱼ)
            λ̃ = dot(vⱼ, Avⱼ)/vⱼᵀvⱼ  # Rayleigh Quotient. 
            ∇r = 2(Avⱼ - λ̃*vⱼ)      # Gradient of Rayleigh Quotient. 
            println("||∇r||: $(norm(∇r)) ")
            if norm(∇r, Inf) <= 1e-10
                V[1:k, j] .= vⱼ/sqrt(vⱼᵀvⱼ)
                Θ[j] = λ̃
                
                break               # go to Next Eigenvector. 
            end
            vⱼ = Avⱼ/norm(Avⱼ)
        end
    end
    return 
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
function Base.:*(this::DynamicSymTridiagonal{T}, b::AbstractArray{T}) where {T <: AbstractFloat}
    @assert length(b) == this.k "Dimension of vector b does match the number of rows in k. Expect $(this.k) but get $(length(b))"
    β = this.betas
    α = this.alphas
    v = similar(b)
    v[1] = α[1]*b[1] + β[1]*b[2]
    for Idx in 2: this.k - 1
        v[Idx] = β[Idx - 1]*b[Idx - 1] + α[Idx]*b[Idx] + β[Idx]*b[Idx + 1]
    end
    v[this.k] = β[this.k - 1]*b[end - 1] + α[this.k]*b[end]
    return v
end


# BASIC TESTING ----------------------------------------------------------------
using LinearAlgebra, Logging
@info "Basic Testing"
n = 8
A = SymTridiagonal(rand(n), rand(n - 1))
T_dy = DynamicSymTridiagonal(A[1, 1])
InitializeEigenSystem(T_dy, n)
for Idx in 2: n
    T_dy(A[Idx, Idx], A[Idx - 1, Idx])
    UpdateEigenSystem(T_dy)
    display(T_dy.V)
end

L = GetL(T_dy)
U = GetU(T_dy)
b = rand(n)
@assert norm(GetT(T_dy) - L*U) < 1e-10
@assert norm(T_dy*b - GetT(T_dy)*b) < 1e-10
