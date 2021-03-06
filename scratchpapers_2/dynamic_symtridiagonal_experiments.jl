include("dynamic_symtridiagonal.jl")
using LinearAlgebra, Plots

function CharPolyEvolution(n=10)
    mainDiag = rand(n)
    subDiag = rand(n - 1)
    dynamicT = DynamicSymTridiagonal(mainDiag[1])
    Grid = LinRange(-1, 1, 101) |> collect
    fig = plot(legend=false)
    for Idx in 1:n - 1
        dynamicT(mainDiag[Idx + 1], subDiag[Idx])
        p = Grid .|>(x) -> CharPolyShifted(dynamicT, x) 
        plot!(fig, Grid, sign.(p).*log10.(abs.(p) .+ 1))
    end
    display(fig)
return end

# CharPolyEvolution()

function MakeMeRandDynamicTridiagonal(n=10)
    mainDiag = rand(n)
    subDiag = rand(n - 1)
    dynamicT = DynamicSymTridiagonal(mainDiag[1])
    for Idx in 1:n - 1
        dynamicT(mainDiag[Idx + 1], subDiag[Idx]) 
    end
return dynamicT end

# DymT = MakeMeRandDynamicTridiagonal()

function ViewConvergence(n=2048)
    a = 1; b = 1/2
    mainDiag = -a*ones(n)
    subDiag = b*ones(n)
    mainDiag .+= a + 2b
    referenceT = SymTridiagonal(mainDiag, subDiag)
    # referenceT = referenceT*3
    mainDiag = referenceT.dv
    subDiag = referenceT.ev
    dynamicT = DynamicSymTridiagonal(mainDiag[1])
    dynamicT.velocity_tol = 1e-4
    fig1=plot(legend=nothing)
    for Idx in n÷16:-1:2
        dynamicT(mainDiag[Idx + 1], subDiag[Idx]) 
        EigenvaluesUpdate(dynamicT)
        dynamicEigs = dynamicT.thetas |> sort
        scatter!(
            fig1,
            dynamicEigs,
            Idx*ones(dynamicEigs |> length),  
            markershape=:cross
        )
        convergedRitz = GetConverged(dynamicT)
        scatter!(
            fig1,
            convergedRitz,
            Idx*(convergedRitz |> length |> ones),
            markershape=:square
        )
        if dynamicT.converged_this_step |> length >= 1
            println("Ritzvalues: $(dynamicT.converged_this_step); converged at step $Idx")
        end
    end
    
    referenceEigs = eigvals(referenceT) |> sort
    scatter!(
        fig1,
        referenceEigs,
        zeros(referenceEigs |> length),
        markershape=:x
    )
    
    display(fig1)
end

ViewConvergence()
