module PreservingMarginals

using JuMP,
	  GLPK,
      Ipopt,
      LinearAlgebra,
      FileIO,
      GaussianMixtures,
      ScikitLearn

include("util.jl")

include("atom.jl")
export Atom,
	   sq_atomic_distance,
	   atomic_distance,
	   sq_atomic_mm,
	   atomic_mm,
	   barycenter

include("mixture.jl")
export Mixture,
	   weights_min,
	   sq_mixture_distance,
	   mixture_distance

include("atoms/slater.jl")
export Slater,
	   MixtureSlater

include("atoms/gaussian.jl")
export Gaussian,
       MixtureGaussian,
       marg,
       im_to_mixture

include("modified.jl")
export pmc_barycenter,
       pmmp_barycenter,
       apmc_barycenter

end
