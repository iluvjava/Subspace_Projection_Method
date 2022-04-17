include("util.jl")
include("../../src/CGPO.jl")  

function PerformCGFor(
    A::AbstractMatrix, 
    b::AbstractVecOrMat;
    epsilon=1e-2,
    exact::Bool=true, 
    partial_ortho=nothing,
)
    cg = CGPO(A, b)
    
    if exact
        
    else
        if partial_ortho === nothing
            cg |> TurnOffReorthgonalize!
        else
            StorageLimit!(cg, partial_ortho)
        end
    end

    ẋ = A\b
    ė = ẋ - cg.x
    ėAė = dot(ė, A*ė)
    E = Vector{Float64}()
    push!(E, 1)
    RelErr = 1
    while RelErr > epsilon
        cg()
        e = ẋ - cg.x
        RelErr = (dot(e, A*e)/ėAė)|>sqrt
        push!(E, RelErr)
    end
    return E
end

E = PerformCGFor(Diagonal(rand(10)), rand(10))

function PerformExperiment1()
    N = 256
    A = GetNastyPSDMatrix(N, 0.9)
    b = rand(N)
    A = convert(Matrix{Float16}, A)
    b = convert(Vector{Float16}, b)
    # TODO: Make the plot distinguishable without colors. 

    # ==========================================================================
    # The exact computations
    # ==========================================================================

    RelErr = PerformCGFor(A, b, epsilon=1e-3, exact=true)
    k = length(RelErr)
    fig1 = plot(
        log10.(RelErr), 
        label="Relative Energy (exact)", 
        legend=:bottomleft
    )

    # ==========================================================================
    # No-Orthogonalizations
    # ==========================================================================

    RelErr = PerformCGFor(A, b, exact=false, epsilon=1e-3)
    k = length(RelErr)
    plot!(
        fig1, 
        log10.(RelErr), 
        label="Relative Energy (floats)",
        linestyle=:dash
    )

    # ==========================================================================
    # Theoretical Bounds
    # ==========================================================================
    ErrorsBound = [TheoreticalErrorBound(A, idx) for idx in 1: k]
    plot!(
        fig1, 
        log10.(ErrorsBound), 
        label="Theoretical Bound (exact)",
        xlabel="iteration count", 
        ylabel="relative error energy norm.",
        linestyle=:dot
    )

    # ==========================================================================
    # Floating Points Partially Orthogonalized
    # ==========================================================================
    RelErr = PerformCGFor(A, b, exact=false, epsilon=1e-3, partial_ortho=div(N, 8))
    k = length(RelErr)
    plot!(
        fig1, 
        log10.(RelErr), 
        label="Relative Energy (partial)",
        legend=:bottomleft, 
        linestyle=:dashdot
    )

    display(fig1)
    SaveFigToCurrentScriptDir(fig1, "fig1.png")

    
return end

PerformExperiment1()
