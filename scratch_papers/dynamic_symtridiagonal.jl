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
    # Sort by absolute values. 
    SortedIdx = sortperm(abs.(this.thetas))
    this.thetas = this.thetas[SortedIdx]
    this.V = this.V[:, SortedIdx]
    return this.thetas, this.V
end

"""
    Update the eigensystem of the bigger matrix using the preivous eigen system. Use Power Iterations 
    iteratively. 
"""
function UpdateEigenSystemPowItr(this::DynamicSymTridiagonal{T}) where {T <: AbstractFloat}
    # if !(isdefined(Base, :this.V))  # cheks if defined first. 
    #     error("Must initialize the eigensystem with an estimated size before calling this function. ")
    # end
    k = this.k
    V = this.V
    ?? = this.thetas
    A = this
    if k > size(this.V, 1)
        error("Hasn't implemented this part yet, k exceed the maximal size of eigen system.")
    end
    Av??? = zeros(T, k)
    for j in 1: k
        v??? = view(V, 1:k, j)
        v??? .+= 1e-10*rand(T, k)    
        if j >= 2   # Components on to previous ortho eigen vectors removed. 
            v??? .-= view(V, 1:k, 1:j - 1)*view(V, 1:k, 1:j - 1)'*v???
        end
        for Itr in 1: k^2 + 1e3      # Power iterations
            v??????v??? = dot(v???, v???)
            Apply!(A, v???, Av???)
            if j >= 2  # Components on to previous ortho eigen vectors removed. 
                Av??? .-= view(V, 1:k, 1:j - 1)*view(V, 1:k, 1:j - 1)'*Av???
            end
            ???? = dot(v???, Av???)/v??????v???  # Rayleigh Quotient. 
            ???r = 2(Av??? - ????*v???)      # Gradient of Rayleigh Quotient. 
            if norm(???r, Inf) <= 1e-8
                V[1:k, j] .= v???/sqrt(v??????v???)
                ??[j] = ????
                break               # go to Next Eigenvector. 
            end
            v??? .= Av???/norm(Av???)
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
    ?? = this.betas
    ?? = this.alphas
    v = similar(b)
    v[1] = ??[1]*b[1] + ??[1]*b[2]
    for Idx in 2: this.k - 1
        v[Idx] = ??[Idx - 1]*b[Idx - 1] + ??[Idx]*b[Idx] + ??[Idx]*b[Idx + 1]
    end
    v[this.k] = ??[this.k - 1]*b[end - 1] + ??[this.k]*b[end]
    return v
end

"""
    Memory friendly version of multiplications. 
"""
function Apply!(this::DynamicSymTridiagonal{T}, b::AbstractArray{T}, v::Vector{T}) where {T <: AbstractFloat}
    @assert length(b) == this.k "Dimension of vector b does match the number of rows in k. Expect $(this.k) but get $(length(b))"
    @assert length(v) == this.k "Dimension of mutable vector v deosn't match the number of rows in k. Expect $(this.k) but get $(length(v))"
    ?? = this.betas
    ?? = this.alphas
    v[1] = ??[1]*b[1] + ??[1]*b[2]
    for Idx in 2: this.k - 1
        v[Idx] = ??[Idx - 1]*b[Idx - 1] + ??[Idx]*b[Idx] + ??[Idx]*b[Idx + 1]
    end
    v[this.k] = ??[this.k - 1]*b[end - 1] + ??[this.k]*b[end]
    return v
end


# BASIC TESTING ----------------------------------------------------------------
using LinearAlgebra, Logging, Plots
@info "Basic Testing"
n = 64
A = SymTridiagonal(fill(-2.0, n), fill(1.0,n - 1))
T_dy = DynamicSymTridiagonal(A[1, 1])
InitializeEigenSystem(T_dy, n)
for Idx in 2: n
    T_dy(A[Idx, Idx], A[Idx - 1, Idx])
    @time UpdateEigenSystemPowItr(T_dy)
    # display(T_dy.V)
end
@info "Dynamic T Eigen System"
display(T_dy.V)
L = GetL(T_dy)
U = GetU(T_dy)
b = rand(n)
@assert norm(GetT(T_dy) - L*U) < 1e-10
@assert norm(T_dy*b - GetT(T_dy)*b) < 1e-10

