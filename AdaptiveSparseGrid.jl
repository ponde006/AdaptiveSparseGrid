module AdaptiveSparseGrid

export spinterp, spvals

########################################################################
# Generic Interpolation Type
########################################################################
type InterpolationObject
    lb::Array{Float64}
    ub::Array{Float64}
    d::Int64
    n::Int64
    surplus::Array
    points::Array
end


maxN = 20
maxD = 5
maxND = maxN + maxD

#######################################################################
####### Creating a function to determine Curtis Clenshaw Points #######
#######################################################################
function X_i( i::Int )
    i == 0 && return Float64[]
    i == 1 && return Float64[0.5]
    m_i = 2^(i-1) + 1
    x_i = collect(((1:m_i) .- 1.0)./(m_i - 1.0))
end
ΔX_i = i -> setdiff(X_i(i),X_i(i-1))
ΔX_Dict = Array[]
for i = 1:maxND
    append!(ΔX_Dict, [ΔX_i(i)])
end

# I am forced to save the points in an array for efficiency reasons.
# Need to find a way to generalize or at least import for dimension
# of the problem
ΔX_Dict
ΔX_i2 = i -> ΔX_Dict[i]

######################################################################
## Creating a function to determine Curtis Clenshaw Basis Functions ##
######################################################################
function a_ij(x::Float64, i::Int, x_i::Float64)
    """
    Calculates basis function values for each dimension.
    For the Clenshaw-Curtis nodes, the basis functions are the hat (triangular) functions
    """
    i == 1 && return 1.0

    m_i = 2^(i-1) + 1
    dif = abs(x - x_i)
    dif < 1/(m_i - 1) && return 1.-(m_i-1)*dif

    return 0.0
end

#######################################################################
############ Creating a function to Generate Multi-indices ############
#######################################################################
function spget( nd::Int, d::Int)
    """
    Calculates the multi-indices for a given degree/dimension
    """
    mySize =(repeated(nd,d)...)
    ΔS = Tuple[]
    for i in CartesianRange( mySize )
        sum(i.I) == nd && append!(ΔS, [i.I])
    end
    return ΔS
end

# Same problem as generating the Clenshaw-Curtis points
ΔS_Dict = Array{Array}(maxND + 1,maxD)
for i = 0:maxND
    for j = 1:maxD
        ΔS_Dict[i+1,j] = spget(i, j)
    end
end
spget2 = (i,j) -> ΔS_Dict[i + 1, j]

######################################################################
################# Calculate the Adaptive Sparse Grid #################
######################################################################
# This is the first change from the spinterp algorithm. Instead of
# calculating all grid points, this function will take the current
# set of points and only calculate the next level Clenshaw-Curtis
# points that are children.
# This function is prohibitively slow. A faster algorithm can be written
# for each dimension individually, but this one generalizes (slowly) for
# all dimensions.

function spgrid(n::Int, d::Int, initPt)
    n == 0 && return Array[Tuple[(repeated(0.5,d)...)]]
    ΔSnd = spget2(n+d, d)
    ΔSndOld = spget2(n+d-1, d)
    m = Base.size(ΔSnd, 1)
    ΔH = Array[]
    #@show size(vcat(vcat(initPt...)...))
    for i = 1:m
        sndNew = ΔSnd[i]
        h = ifelse([sndNew...] .== 1, 0.5, 2.0.^(-[sndNew...]+1))
        toComp = vcat(Base.product(map(ΔX_i2, sndNew)...)...)
        newPts = []
        for l in eachindex(initPt)
            p = initPt[l]
            snd = ΔSndOld[l]
            if prod([map(-, sndNew, snd)...]) == 0
                for c in p
                    for tp in toComp
                        tmp = abs([map(-, c, tp)...])
                        tmpAdd = 1.0
                        for i in eachindex(tmp)
                            tmpAdd = tmpAdd * ( (tmp[i] == h[i]) + (tmp[i] == 0.0))
                        end
                        #tpAdd =  prod((tmp .== h).+(tmp .== 0.0))
                        if tmpAdd == 1.0
                            ((tp in newPts) == 0) && push!(newPts, tp)
                        end
                    end
                end
            end
        end
         (((newPts in ΔH) == 0)+(newPts == [])) > 0 && push!(ΔH, newPts )
    end
    return ΔH
end
#####################################################################
################# Calculate the Interpolated Values #################
#####################################################################
function twoStep(P::Tuple, ΔS::Tuple, ΔH::Array)
    """
    Calculates tensor-product of basis function values to be used in
    spinterstep
    """
    tmp = Float64[]
    for i in ΔH
        α_ij = 1.0
        for j in eachindex(ΔS) # Columns in ΔS
            indx = ΔS[j]
            α_ij = α_ij * a_ij(P[j], indx, i[j])
            α_ij == 0.0 && break
        end
        append!(tmp, α_ij)
    end
    return tmp
end

function spinterpstep( d::Int, Z::Array, P, ΔH::Array, ΔS::Array, b = 1.0, a = 0.0 )
    """
    Interpolates points based on current surpluses
    Depends on the functions a_ij and twoStep
    """
    R = Base.size(P, 1)
    m = Base.size(ΔS, 1)
    ΔY = zeros(R)

    for r = 1:R

        for i = 1:m # obs in ΔS
            Z[i] == [] && break
            α_ij = twoStep(P[r], ΔS[i], ΔH[i])
            ΔY[r] += dot(α_ij, Z[i])
        end
    end
    return ΔY
end;

#####################################################################
################## Update Surpluses for new Degree ##################
#####################################################################
function spvalstep(d::Int, Zold::Array, ΔHnew::Array, ΔHold::Array, ΔSnew::Array, ΔSold::Array, b = 1.0, a = 0.0)
    """
    Calculates new surpluses for n + 1 is n is the original degree
    """
    mold = Base.size(ΔSold, 1)
    mnew = Base.size(ΔSnew, 1)
    ΔY = Array{Float64}[zeros(Base.size(ΔHnew[i])) for i = 1:Base.size(ΔHnew,1)]

    for lold = 1:mold
        iold = ΔSold[lold]
        for lnew = 1:mnew
            inew = ΔSnew[lnew]
            if prod([iold...] .<= [inew...])
                ΔY[lnew] .+= spinterpstep(d, Array[Zold[lold]], ΔHnew[lnew], Array[ΔHold[lold]], [iold], b, a )
            end
        end

    end
    return ΔY
end

#####################################################################
################## Recursively Calculate Surpluses ##################
#####################################################################
# Primary calculation of values
# Unlink Zlimke and Wohlmuth, this function calculates surpluses for
# an arbitrary rectangle and adaptively generates grid points. The critera
# for including a grid point is that abs(α_i...j) ≧ ϵ
# Calculations are done recursively. Surpluses are calculated for degree 1,
# then for degree 2 using the degree 1 interpolation, etc.
# There might be a more efficient way to program this, although I have not
# explored it yet. Unlike spinterp, n controls max depth here
function spvals(f::Function, d::Int, n::Int, b = 1.0, a = 0.0,  ϵ = .0001)
    """
    Function to calculate a surpluses for a degree n interpolation of function f of dimension d
    """
    k = 0
    Z = []
    ΔH = Array[Tuple[]] # Start with no points
    ΔH_tmp = Array[Tuple[]] # Start with no points
    ln0 = 0.0
    ln1 = 0.5
    ΔH_Full = Array[]
    while (ln1 > ln0)*(k < n)
        ln0 = ln1
        ΔSkd = spget2(k+d, d)
        ΔH = spgrid(k, d, ΔH_tmp) ### This is what needs to adapt
        push!(ΔH_Full, ΔH)
        Yk = Array{Float64}[Float64[f( ([i...].*(b .- a) .+ a)... ) for i in j] for j in ΔH]
        append!(Z , Array[Yk])
        ΔH_tmp = deepcopy(ΔH)
        if k > 0
            for m = 0:k-1
                ΔSmd = spget2(m+d, d)
                Z[k+1] = Z[k+1] .- spvalstep(d, Z[m+1], ΔH, ΔH_Full[m+1], ΔSkd, ΔSmd, b, a )
            end

            for zs = 1:length( Z[k+1] )
                indx2 = find(abs(Z[k+1][zs]) .<= ϵ)
                deleteat!(ΔH_tmp[zs], indx2)
#                deleteat!(Z[k+1][zs], indx2)
            end
        end

        k += 1
        ln1 = length(vcat(vcat(Z...)...))
    end

    t = InterpolationObject(a, b, d, n, Z, ΔH_Full)
end

#####################################################################
############### Interpolate Values based on Surpluses ###############
#####################################################################
function spinterp(d::Int, n::Int, Z::Array, P::Array, ΔH_Full::Array, b = 1.0, a = 0.0)
    """
    Calculates interpolated values from a vector of tuples P given surpluses Z
    Requires the adaptively generated grid ΔH
    """
    y = zeros(Base.size(P,1))

    for kn in eachindex(ΔH_Full)
        ΔSkd = spget2(kn+d - 1, d)
        y .+= spinterpstep( d, Z[kn], P, ΔH_Full[kn], ΔSkd, b, a )
    end
    return y
end


#####################################################################
# Should make this multi-dimensional
#####################################################################
function(itp::InterpolationObject)(x::Float64,y::Float64)
    a = itp.lb
    b = itp.ub
    interpPt = [(  (x-a[1])/(b[1]-a[1]), (y-a[2])/(b[2]-a[2])  ) ]
    return spinterp(itp.d, itp.n, itp.surplus, interpPt , itp.points, itp.ub, itp.lb)[1]
end

function(itp::InterpolationObject)(x::Array{Float64},y::Array{Float64})
    a = itp.lb
    b = itp.ub
    interpPt = vcat(Base.product((x-a[1])/(b[1]-a[1]),(y-a[2])/(b[2]-a[2]))...)
    return spinterp(itp.d, itp.n, itp.surplus, interpPt , itp.points, itp.ub, itp.lb)
end

function(itp::InterpolationObject)(x::Float64,y::Float64, z::Float64, w::Float64, t::Float64)
    a = itp.lb
    b = itp.ub
    interpPt = [(  (x-a[1])/(b[1]-a[1]), (y-a[2])/(b[2]-a[2]), (z-a[3])/(b[3]-a[3]), (w-a[4])/(b[4]-a[4]), (t-a[5])/(b[5]-a[5])  ) ]
    return spinterp(itp.d, itp.n, itp.surplus, interpPt , itp.points, itp.ub, itp.lb)[1]
end

end
