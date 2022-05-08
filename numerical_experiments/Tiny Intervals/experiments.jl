include("./utilities.jl")
include("../../src/CGPO.jl")


# ==============================================================================
# Floats on the exact matrix.

function FloatsOnExactMatrix()
    N = 64
    function PerformForRho(rho)
        Eigens = EigDistribution(N, rho=rho, eigen_min=1e-5)
        A, _ = TinyIntervalTestMatrices(
            Eigens, 
            scale=N*1e-5(maximum(Eigens)/minimum(Eigens))
        )
        b = ones(size(A, 1))
        RelErrors = PerformCGFor(A, b, exact=false, epsilon=1e-10) 
    return RelErrors end

    RelErrors = PerformForRho(0.8)
    fig = plot(
        RelErrors, 
        yaxis=:log10,
        xlabel="iterations: k", 
        ylabel="\$\\frac{\\Vert e_k \\Vert_A}{\\Vert e_0\\Vert_A}\$",
        label="\$\\rho=0.8\$",
        left_margin = 10Plots.mm, 
        size=(750, 500), dpi=300,
    )
    RelErrors = PerformForRho(0.9)
    plot!(
        fig, 
        RelErrors, 
        label="\$\\rho=0.9\$", 
        linestyle=:dot
    )
    RelErrors = PerformForRho(0.7)
    plot!(
        fig, 
        RelErrors, 
        label="\$\\rho=0.7\$", 
        linestyle=:dashdot
    )
    RelErrors = PerformForRho(1)
    plot!(
        fig,
        RelErrors, 
        label="\$\\rho=1\$", 
        linestyle=:dash
    )
    title!(fig, "Float CG on Exact Matrix")
    fig |> display
    savefig(fig, "$(@__DIR__)/plots/floats_cg_on_exact_matrix")
return nothing end

FloatsOnExactMatrix()

# ==============================================================================
# Exact on the Smeared out matrix. 


function ExactOnSmearedMatrix()
    N = 64
    function PerformForRho(rho)
        Eigens = EigDistribution(N, rho=rho, eigen_min=1e-5)
        ConditionNumb = maximum(Eigens)/minimum(Eigens)
        _, A = TinyIntervalTestMatrices(
            Eigens, 
            scale=2e-5ConditionNumb,
            m=100
        )
        b = ones(size(A, 1))
        RelErrors = PerformCGFor(A, b, exact=true, epsilon=1e-10) 
    return RelErrors end

    RelErrors = PerformForRho(0.8)
    fig = plot(
        RelErrors, 
        yaxis=:log10,
        xlabel="iterations: k", 
        ylabel="\$\\frac{\\Vert e_k \\Vert_A}{\\Vert e_0\\Vert_A}\$",
        label="\$\\rho=0.8\$",
        left_margin = 10Plots.mm, 
        size=(750, 500), dpi=300,
    )
    RelErrors = PerformForRho(0.9)
    plot!(
        fig, 
        RelErrors, 
        label="\$\\rho=0.9\$", 
        linestyle=:dot
    )
    RelErrors = PerformForRho(0.7)
    plot!(
        fig, 
        RelErrors, 
        label="\$\\rho=0.7\$", 
        linestyle=:dashdot
    )
    RelErrors = PerformForRho(1)
    plot!(
        fig,
        RelErrors, 
        label="\$\\rho=1\$", 
        linestyle=:dash
    )
    title!(fig, "Exact CG on Smeared Matrix")
    savefig(fig, "$(@__DIR__)/plots/exact_cg_on_smeared_matrix")
    fig |> display
return nothing end

ExactOnSmearedMatrix()
