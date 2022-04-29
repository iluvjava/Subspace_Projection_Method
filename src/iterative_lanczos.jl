mutable struct IterativeLanczos
    A::Function
    betas::Dict    # The sub and super diagonal of the matrix
    alphas::Dict   # The diagonal of the matrix
    k::Int64       # when k=1, the algorithm haven't started yet. 
    q_store::Int64
    Aq
    Q::Dict
    
    function IterativeLanczos(A::Function, q1, q_store::Int64=typemax(Int64))
        @assert q_store > 1 "storage must be at least 2. "
        this = new()
        this.A = A
        this.betas = Dict()
        this.alphas = Dict()
        this.k = 1
        this.q_store = q_store
        q1 = q1/norm(q1)
        this.Aq = A(q1)
        this.Q = Dict{Int64, typeof(this.Aq)}()
        this.Q[1] = q1
        this.alphas[1] = dot(this.Q[1], this.Aq)  # Alpha is computed in Advance!
        return this
    end

    function IterativeLanczos(A::AbstractArray, q1, q_store::Int64=typemax(Int64))
        return IterativeLanczos((x) -> A*x, q1, q_store)
    end

end

# Override the () operator. 
"""
    It performs one iteration of the Lanczos Orthogonalization. 
"""
function (this::IterativeLanczos)()
    q = this.Q[this.k]
    Aq = this.Aq
    alphas = this.alphas
    if this.k == 1
        qNew = Aq
    else
        qPre = this.Q[this.k - 1]
        qNew = Aq - this.betas[this.k - 1]*qPre
    end
    qNew -= alphas[this.k]*q
    betaNew = norm(qNew)
    # Reorthogonalizations goes here

    qNew /= betaNew

    if length(this.Q) > this.q_store
        delete!(this.Q, this.k + 1 - length(this.Q))
    end

    this.Q[this.k + 1] = qNew
    this.Aq = this.A(qNew)
    this.alphas[this.k + 1] = dot(qNew, this.Aq)
    this.betas[this.k] = betaNew
    this.k += 1
    
    return qNew
end


"""
    It calls the () operator 3 j times, performing j iterations. 
"""
function (this::IterativeLanczos)(j::Int64)
    betas = Vector()
    for _ in 1: j
        push!(betas, this())
    end
    return betas
end


"""
    Get the Q matrix from the Lanczos Iterations, the number of columnos of the 
    matrix depends on this.q_store.
    * k by k matrix returned. 

"""
function GetQMatrix(this::IterativeLanczos)
    Q = this.Q
    qs = Vector{typeof(Q[1])}()
    for Idx in 1: this.k
        push!(qs, this.Q[Idx])
    end
    return hcat(qs...)
end


"""
    Return the Tridiagonal matrix, take note thta, the matrix only supports 
    floating point, and it's returned as a sparse matrix. 
    * A (k - 1) by (k - 1) matrix is returned. 

"""
function GetTMatrix(this::IterativeLanczos)
    if this.k == 1
        return this.alphas[1]
    end
    return SymTridiagonal{Float64}(
        [real(this.alphas[Idx]) for Idx in 1: this.k],
        [real(this.betas[Idx]) for Idx in 1:this.k - 1]
    )
end


"""
    Get the Hessenberg form of the Tridiagonal matrix, the shape of the 
    matrix is (k,k - 1). 
"""
function GetHMatrix(this::IterativeLanczos)
    T = GetTMatrix(this)
    return T[:, 1: end - 1]
end


"""
    Returns Q, T matrices such that QTQ^T approximates A
"""
function DecompositionQTQ(this::IterativeLanczos)
    return GetQMatrix(this), GetTMatrix(this)
end


"""
    Returns H, Q Matrix such that AQ = QH. 
"""
function DecompositionQH(this::IterativeLanczos)

    return GetQMatrix(this)[:, 1: end - 1], GetHMatrix(this)
end


"""
    Get the previous 3 Orthogonal vector from the Lanczos Algorithm. 
    * If we haven' generated 3 orthogonal vector 3, then 
    this will just return the first 2, or 1 orthgonal matrix. 
"""
function GetPrevious3OrthogonalVec(this::IterativeLanczos)
    if this.k == 1
        return this.Q[1]
    end
    if this.k == 2
        return hcat(this.Q[1], this.Q[2])
    end
    return hcat(this.Q[this.k], this.Q[this.k - 1], this.Q[this.k - 2])
end

