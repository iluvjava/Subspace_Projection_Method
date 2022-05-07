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

function PerformExperiment1(filename, p=0.9)
    N = 256
    A = GetNastyPSDMatrix(N, p)
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
        RelErr, 
        label="Relative Energy (exact)", 
        legend=:bottomleft,
        yaxis=:log10, 
        xlabel="iterations: k", 
        ylabel="\$\\frac{\\Vert e_k \\Vert_A}{\\Vert e_0\\Vert_A}\$",
        title="CG Rel Energy Error \$\\rho = $(p)\$",
        size=(750, 500), dpi=300, 
        left_margin = 10Plots.mm
    )

    # ==========================================================================
    # No-Orthogonalizations
    # ==========================================================================

    RelErr = PerformCGFor(A, b, exact=false, epsilon=1e-3)
    k = length(RelErr)
    plot!(
        fig1, 
        RelErr, 
        label="Relative Energy (floats)",
        linestyle=:dash#, markershape=:+
    )

    # ==========================================================================
    # Theoretical Bounds
    # ==========================================================================
    ErrorsBound = [TheoreticalErrorBound(A, idx) for idx in 1: k]
    plot!(
        fig1, 
        ErrorsBound, 
        label="Theoretical Bound (exact)",
        linestyle=:dot
    )

    # ==========================================================================
    # Floating Points Partially Orthogonalized
    # ==========================================================================
    RelErr = PerformCGFor(A, b, exact=false, epsilon=1e-3, partial_ortho=div(N, 8))
    k = length(RelErr)
    plot!(
        fig1, 
        RelErr, 
        label="Relative Energy (partial)",
        legend=:bottomleft, 
        linestyle=:dashdot # ,  markershape=:x
    )

    display(fig1)
    SaveFigToCurrentScriptDir(fig1, "$(filename).png")

    
return end

PerformExperiment1("cg_convergence_0.9")
PerformExperiment1("cg_convergence_1", 1)
