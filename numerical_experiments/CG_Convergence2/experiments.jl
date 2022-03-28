include("util.jl")
include("../../src/CGPO.jl")  

function PerformCGFor(
    A::AbstractMatrix, 
    b::AbstractVecOrMat;
    epsilon=1e-2,
    immitate_exact::Bool=true
)
    cg = CGPO(A, b)
    
    if !immitate_exact 
        TurnOffReorthgonalize!(cg)
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
    N = 128
    A = GetNastyPSDMatrix(N, 0.95)
    b = rand(N)
    A = convert(Matrix{Float16}, A)
    b = convert(Vector{Float16}, b)
    RelErr = PerformCGFor(A, b, epsilon=1e-2)
    k = length(RelErr)
    ErrorsBound = [TheoreticalErrorBound(A, idx) for idx in 1: k]

    fig1 = plot(log10.(RelErr), label="Relative Energy (exact)", legend=:bottomleft)
    plot!(
        fig1, 
        log10.(ErrorsBound), 
        label="Theoretical Bound (exact)",
        xlabel="iteration count", 
        ylabel="relative error energy norm."
    )


    # ==============================================================================
    # Floating Points Error Bound.
    # ==============================================================================

    # A = GetUniformPSDMatrix(N)
    # b = rand(N)
    RelErr = PerformCGFor(A, b, immitate_exact=false, epsilon=1e-2)
    k = length(RelErr)
    ErrorsBound = [TheoreticalErrorBound(A, idx) for idx in 1: k]

    # fig2 = plot(log10.(RelErr), label="Relative Energy (floats)", legend=:bottomleft)
    # plot!(
    #     fig1, 
    #     log10.(ErrorsBound), 
    #     label="Theoretical Error Bound (exact)", 
    #     xlabel="iteration count", 
    #     ylabel="relative error energy norm"
    # )
    # display(fig2)
    # SaveFigToCurrentScriptDir(fig2, "fig2.png")
    plot!(
        fig1, 
        log10.(RelErr), 
        label="Relative Energy (floats)",
        legend=:bottomleft
    )
    display(fig1)
    SaveFigToCurrentScriptDir(fig1, "fig1.png")
return end

PerformExperiment1()
