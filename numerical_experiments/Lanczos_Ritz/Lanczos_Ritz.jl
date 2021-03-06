include("Lanczos_Ritz_Utilities.jl")
include("../../src/iterative_lanczos.jl")

# ------------------------------------------------------------------------------
# Demonstrating the loss of orthogonality of the lanczos vectors. 
# And the linear dependence of eigenvalues of the exterior of the spectrum. 
# ------------------------------------------------------------------------------
n = 64
A = Diagonal(LinRange(-1, 1, n).^3)
il = IterativeLanczos(A, rand(n))
for _ in 1: n - 1
    il()
end
Q = GetQMatrix(il)
T = GetTMatrix(il)

fig = heatmap(Q'*Q, size=(722, 512))
fig2 = heatmap(Q'*A*Q, size=(722, 512))
fig3 = heatmap(
    (A*Q[:, 1:end - 1] - Q*T[:, 1:end - 1]).|> abs, 
    size=(1200, 768), dpi=250
)
savefig(fig, "$(@__DIR__)/plots/fig3.png")
savefig(fig2, "$(@__DIR__)/plots/fig4.png")



# ------------------------------------------------------------------------------
# All the ritz values during the computations process. and plotting it. 
# ------------------------------------------------------------------------------

function RiztTrajectoryPlot(filename, n=64; itr_offset=20, full_ortho=false, title="")
    A = Diagonal(LinRange(-1, 1, n).^3)
    il = IterativeLanczos(A, ones(n))
    il.reorthogonalize = full_ortho
    FoundRitzValues = Vector{Vector{Float64}}()
    # push!(FoundRitzValues, [GetTMatrix(il)])
    TrueEigenValues = diag(A)
    
    for II in 1: n - 1
        il()
        T = GetTMatrix(il)
        λs, _ = eigen(T)
        if II >= itr_offset
            push!(FoundRitzValues, λs)
        end
    end
    
    for RitzValue in FoundRitzValues
        sort!(RitzValue, rev=true)
    end
    
    fig4 = scatter(title=title, legend=false)
    for Idx in 1: div(itr_offset,2)
        RitzTrajectory = Vector{Float64}()
        for RitzValues in FoundRitzValues
            if Idx <= length(RitzValues)
                push!(RitzTrajectory, RitzValues[Idx])
            end
        end
        plot!(
            fig4, 
            # Idx: length(RitzTrajectory) + Idx - 1, 
            itr_offset:(length(RitzTrajectory) + itr_offset - 1), 
            RitzTrajectory, size=(750,750), 
            dpi=250, 
            markershape=:cross, 
            linestyle=:solid
        )
    end
    for Idx in 1:div(itr_offset, 2)
        RitzTrajectory = Vector{Float64}()
        for RitzValues in FoundRitzValues
            if Idx <= length(RitzValues)
                push!(RitzTrajectory, RitzValues[end - Idx + 1])
            end
        end
        plot!(
            fig4, 
            # Idx: length(RitzTrajectory) + Idx - 1, 
            itr_offset:(length(RitzTrajectory) + itr_offset - 1),
            RitzTrajectory,
            dpi=250, 
            markershape=:xcross,
            linestyle=:solid
        )
    end
    xlabel!(fig4, "iterations")
    ylabel!(fig4, "ritz values")
    display(fig4)
    savefig(fig4, "$(@__DIR__)/plots/$(filename).png")
    
return end 

RiztTrajectoryPlot("ritz_trajectory_plot_floats", title="ritz trajectory no-reorthgonalization")
RiztTrajectoryPlot("ritz_trajectory_plot_exact", full_ortho=true, title="ritz trajectory with full-reorthogonalizatoin")

