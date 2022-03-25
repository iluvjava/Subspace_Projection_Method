### ----------------------------------------------------------------------------
### ---- Basic testing for the Shifted Characteristic Polynomial Evaluation: 
### ----------------------------------------------------------------------------

include("dynamic_symtridiagonal_v2.jl")

using LinearAlgebra, Logging, Plots

"""
    Basic test on the characteristic polynomial for the tridiagonal Matrix. 
"""
function Test1(n=50)
    mainDiag = rand(n)
    subDiag = rand(n - 1)
    dynamicT = DynamicSymTridiagonal(mainDiag[1])
    T = GetT(dynamicT)
    for Idx in 1:n - 1
        dynamicT(mainDiag[Idx + 1], subDiag[Idx])
    end
    @info "Characteristic poly without shift evaluated to be: "
    println(CharPolyShifted(dynamicT, 0))
    println(det(T))
    @info "Plot out the characteric polynomial over the min, max eigenvalue. "
    minEigen = eigmin(T); maxEigen = eigmax(T)
    grid = LinRange(minEigen, maxEigen, 1024)
    p = grid .|> (x) -> CharPolyShifted(dynamicT, x)
    p2 = grid .|> (x) -> det(T - x*I)
    fig = plot(grid, sign.(p).*log10.(abs.(p) .+ 1))
    display(plot!(fig, grid, sign.(p2).*log10.(abs.(p2) .+ 1)))

return dynamicT end

Test1();

function Test2(n = 200)
    mainDiag = rand(n)
    subDiag = rand(n - 1)
    dynamicT = DynamicSymTridiagonal(mainDiag[1])
    T = GetT(dynamicT)
    for Idx in 1:n - 1
        dynamicT(mainDiag[Idx + 1], subDiag[Idx])
    end
    # try locating one of the inner eigenvalue of matrix T. 
    @info "Trying out the EigenValueLocate function. "
    eigenVals = eigvals(T)
    sort!(eigenVals)
    P(x) = CharPolyShifted(dynamicT, x)
    locatedEig = EigenValueLocate(
        dynamicT, 
        (eigenVals[end - 1] + eigenVals[end])/2,
        Inf)
    println("locatedEig: $locatedEig")
    println("Expected: $(eigenVals[end])")
    error = abs(locatedEig - eigenVals[end])
    println("error: $(error)")
    println("This is the lsit of all eigenvalues: ")
    display(eigenVals)    
    println("The corresonding eigenvector is: ")
    LocatedEigenVec = eigvecs(T, [locatedEig])
    if error > 1e-10
        @warn "Error is bigger than expected. "
        # might over probe and cause big problem. 
    end
    


return T, locatedEig, LocatedEigenVec end


T, locatedEig, locatedEigenVec = Test2()
display(plot(locatedEigenVec))
