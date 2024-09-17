"""
    pmc_barycenter( atoms::Vector{Gaussian{N}}, λ::Vector{Float64}, dimx::Int64 ) where N
	
	Compute the closest preserving marginal barycenter between atoms `atoms` with weights `λ`.
"""
function pmc_barycenter( atoms::Vector{Gaussian{N}}, λ::Vector{Float64}, dimx::Int64 ) where N
    b = barycenter(atoms, λ)
    bx = b.Σ[1:dimx, 1:dimx]
    by = b.Σ[dimx+1:N, dimx+1:N]
    bxy = b.Σ[1:dimx, dimx+1:N]

    mbx = barycenter([ marg(atom, 1:dimx) for atom in atoms ], λ).Σ
    mby = barycenter([ marg(atom, dimx+1:N) for atom in atoms ], λ).Σ
    smbx = sqrt(mbx)
    smby = sqrt(mby)

    Xopt = smbx * inv(sqrt(smbx * bx * smbx)) * smbx * bxy * smby * inv(sqrt(smby * by * smby)) * smby
    Gaussian([ mbx Xopt; transpose(Xopt) mby ], b.r)
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
    function apmc_barycenter( mixtures::Vector{Mixture{N, Gaussian{N}}}, λ::Vector{Float64}, dimx::Int64 ) where N
	
	Compute the approximated closest preserving marginal barycenter between mixtures `mixtures` with weights `λ`.
"""
function apmc_barycenter( mixtures::Vector{Mixture{N, Gaussian{N}}}, λ::Vector{Float64}, dimx::Int64 ) where N
    mask = λ .!= 0.0
    λ = λ[mask]
    mixtures = mixtures[mask]

    margsx = [ marg(mixture, 1:dimx) for mixture in mixtures ]
    margsy = [ marg(mixture, dimx+1:N) for mixture in mixtures ]

    m = barycenter(mixtures, λ, weights_min(mixtures, λ))
    ρx = barycenter(margsx, λ, weights_min(margsx, λ)) 
    ρy = barycenter(margsy, λ, weights_min(margsy, λ)) 

    sΣρxs = [ sqrt(atom.Σ) for atom in ρx.atoms ]
    sΣρys = [ sqrt(atom.Σ) for atom in ρy.atoms ]

    atoms = [ Gaussian(Matrix(Hermitian([ ρx.atoms[k].Σ sΣρxs[k] * inv(sqrt(sΣρxs[k] * m.atoms[j].Σ[1:dimx, 1:dimx] * sΣρxs[k])) * sΣρxs[k] * m.atoms[j].Σ[1:dimx, dimx+1:N] * sΣρys[l] * inv(sqrt(sΣρys[l] * m.atoms[j].Σ[dimx+1:N, dimx+1:N] * sΣρys[l])) * sΣρys[l];
                                          zeros(N-dimx, dimx) ρy.atoms[l].Σ ])),
                       [ ρx.atoms[k].r; ρy.atoms[l].r ]) for k in 1:length(ρx.atoms), l in 1:length(ρy.atoms), j in 1:length(m.atoms) ]
    mat = [ sq_atomic_distance(atoms[k, l, j], m.atoms[i]) for k in 1:length(ρx.atoms), l in 1:length(ρy.atoms), j in 1:length(m.atoms), i in 1:length(m.atoms) ]

	model = Model(GLPK.Optimizer)
    @variable(model, w[1:length(ρx.atoms), 1:length(ρy.atoms), 1:length(m.atoms), 1:length(m.atoms)] >= 0.0)
    @constraint(model, [ k=1:length(ρx.atoms) ], sum(w[k, :, :, :]) == ρx.weights[k])
    @constraint(model, [ l=1:length(ρy.atoms) ], sum(w[:, l, :, :]) == ρy.weights[l])
    @constraint(model, [ i=1:length(m.atoms) ], sum(w[:, :, :, i]) == m.weights[i])

	@objective(model, Min, sum(w .* mat))
	optimize!(model)

	w = value.(w)
    γ = vec([ sum(w[k, l, j, :]) for k in 1:length(ρx.atoms), l in 1:length(ρy.atoms), j in 1:length(m.atoms) ])
    
    atoms = vec(atoms)
    objective_value(model), m, Mixture(atoms[γ .> 0.], γ[γ .> 0.])
end
