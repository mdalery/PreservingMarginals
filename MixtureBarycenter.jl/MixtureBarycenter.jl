############################################################################################
#####                                                                                  ##### 
#####                                                                                  ##### 
#####     MixtureBarycenter.jl                                                         #####
#####                                                                                  #####
#####                                                                                  #####
#####     Julia module for calculations of Mixture barycenters.                        #####
#####     Contains mixtures of:                                                        #####
#####     		* 1D-Slater with 2-Wasserstein distance                                #####
#####                                                                                  #####
#####                                                                                  ##### 
#####     Made with julia 1.9.0                                                        #####
#####                                                                                  ##### 
#####                                                                                  ##### 
############################################################################################

module MixtureBarycenter

using JuMP,
	  GLPK,
      Ipopt,
      LinearAlgebra

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
       MixtureGaussian

include("modified.jl")
export pmc_barycenter,
       pmmp_barycenter

end
