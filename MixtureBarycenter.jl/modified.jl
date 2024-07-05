"""
    pmc_barycenter( atoms::Vector{Gaussian{2}}, λ::Vector{Float64} )
	
	Compute the closest preserving marginal barycenter between atoms `atoms` with weights `λ`.
    Defined only for 2D Gaussian distributions.
"""
function pmc_barycenter( atoms::Vector{Gaussian{2}}, λ::Vector{Float64} )
    @assert length(atoms) == length(λ)

    b = barycenter(atoms, λ)
    sqrtbx = sum(λ .* [ sqrt(atom.Σ[1, 1]) for atom in atoms ])
    sqrtby = sum(λ .* [ sqrt(atom.Σ[2, 2]) for atom in atoms ])

    Xopt = sqrtbx / sqrt(b.Σ[1, 1]) * b.Σ[1, 2] * sqrtby / sqrt(b.Σ[2, 2])
    return Gaussian([ sqrtbx^2 Xopt; Xopt sqrtby^2 ], b.r)
end

"""
    pmc_barycenter( mixtures::Vector{Mixture{2, Gaussian{2}}}, λ::Vector{Float64}, w::Dict{CartesianIndex{N}, Float64} ) where N

	Compute the closest preserving marginal barycenter between the mixtures `mixtures` with weights `λ`. Previously computed weights ``w^*`` are required (see weights_min).
    Defined only for 2D Gaussian mixture distributions.
"""
function pmc_barycenter( mixtures::Vector{Mixture{2, Gaussian{2}}}, λ::Vector{Float64}, w::Dict{CartesianIndex{N}, Float64} ) where N
	begin
		@assert N isa Int64
		@assert length(mixtures) == length(λ) == N
	end

	atoms = [ pmc_barycenter([ @inbounds mixtures[n].atoms[kbar[n]] for n in 1:N ], λ) for kbar in keys(w) ]
	weights = collect(values(w))
	Mixture(atoms, weights)
end


function _sqrt22(M)
    a, b, _, c = vec(M)
    s = sqrt(a*c - b^2)
    τ = a + c
    t = sqrt(τ + 2*s)

    [ 1/t * (a + s) 1/t * b; 1/t * b 1/t * (c + s) ]
end

"""
    pmmp_barycenter( atoms::Vector{Gaussian{2}}, λ::Vector{Float64} )
	
	Compute the minimization problem preserving marginal barycenter between atoms `atoms` with weights `λ`.
    Defined only for 2D Gaussian distributions.
"""
function pmmp_barycenter( atoms::Vector{Gaussian{2}}, λ::Vector{Float64} )
    @assert length(atoms) == length(λ)

    sqrtbx = sum( λ .* [ sqrt(atom.Σ[1, 1]) for atom in atoms ]) 
    sqrtby = sum( λ .* [ sqrt(atom.Σ[2, 2]) for atom in atoms ]) 

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, -sqrtbx*sqrtby + 1e-7 <= X <= sqrtbx*sqrtby - 1e-7, start = 0.0)

    B = [ sqrtbx^2 X; X sqrtby^2 ]
    Bh = _sqrt22(B)
    Ms = [ atom.Σ + B - 2*_sqrt22(Bh * atom.Σ * Bh) for atom in atoms ]
    d = sum( λ .* [ M[1, 1] + M[2, 2] for M in Ms ])
    @objective(model, Min, d)

    optimize!(model)
    Σ = Matrix(Hermitian(value.(B)))
    r = sum(λ .* [ atom.r for atom in atoms ])
    Gaussian(Σ, r)
end

"""
    pmmp_barycenter( mixtures::Vector{Mixture{2, Gaussian{2}}}, λ::Vector{Float64}, w::Dict{CartesianIndex{N}, Float64} ) where N

	Compute the minimization problem preserving marginal barycenter between the mixtures `mixtures` with weights `λ`. Previously computed weights ``w^*`` are required (see weights_min).
    Defined only for 2D Gaussian mixture distributions.
"""
function pmmp_barycenter( mixtures::Vector{Mixture{2, Gaussian{2}}}, λ::Vector{Float64}, w::Dict{CartesianIndex{N}, Float64} ) where N
	begin
		@assert N isa Int64
		@assert length(mixtures) == length(λ) == N
	end

	atoms = [ pmmp_barycenter([ @inbounds mixtures[n].atoms[kbar[n]] for n in 1:N ], λ) for kbar in keys(w) ]
	weights = collect(values(w))
	Mixture(atoms, weights)
end
