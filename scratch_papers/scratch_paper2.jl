"""
    Hm... Power method should work well for symmatric tridiagonal matrix that is 
    always changing its size... In fact it does, let's try inverse iterations 
    instead. 
"""


"""
    Left and right boundary can actually be infinite. 
    Because of left and right bound, we have to restrict it to type Abstract Floats so that it's 
    comparable, which is going to be used a lot for search the eigen values of the matrix. 
"""
function InversePowerIterSearch(
        A::AbstractMatrix{T}, 
        left_bound::T, 
        right_bound::T
    ) where {T<: AbstractFloat}
    # Get the mid point. 
    @assert size(A, 1) == size(A, 2) "It has to be square matrix."
    @assert left_bound != right_bound "The interval must be non-degenerate."
    if left_bound > right_bound
        left_bound, right_bound = right_bound, left_bound
    end
    if left_bound == -Inf && right_bound == Inf
        error("Left and right bound cannot both be Inf, then it's unbounded. ")
    end
    v = randn(T, size(A, 1))    # previous iterations
    u = similar(v)                 # current iterations
    Au = similar(v)
    t = similar(v)              # temp 
    MaxItr = 100
    while MaxItr > 0
        # Choose the right point. 
        if left_bound == -Inf || right_bound == Inf
            θ = left_bound == -Inf ? 2*left_bound : 2*right_bound
            if θ == -Inf || θ == Inf
                error("Inverse Iteration Igen Search failed, bound limit reached for half opened interval.  ")
            end
        else
            θ = (left_bound + right_bound)/2
        end
        println("θ: $θ ∈ [$left_bound, $right_bound]")
        # Inverse Power iterations.
        λ = θ
        while λ == θ || norm(Au - λ*u, 1) > 1e-10
            println(λ)
            t .= u
            u .= (A - λ*I)\v
            u ./= norm(u)
            if norm(u) == NaN
                u .= v
                println("coverted to $λ")
                break
            else
                v .= t
                Au = A*u
                λ = dot(u, Au)
            end
        end
        # Interval shrink.
        if λ > left_bound && λ < right_bound
            
            return λ, u
        else
            if λ <= left_bound
                left_bound = θ
                
            else
                right_bound = θ
            end
        end
        MaxItr -= 1
    end
    if MaxItr == 0
       error("Inverse Power iterations method reached maximal iterations without convergence. ") 
    end
end

# Basic Tests and shit
using LinearAlgebra
A = Diagonal(sort(randn(10)))
display(A)
λ, v = InversePowerIterSearch(A, A[1, 1], A[3, 3])
print("λ = $λ")
# λ, v = InversePowerIterSearch(A, -Inf, A[2, 2])
# print("λ = $λ")