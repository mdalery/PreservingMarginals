include("util_plot.jl")
println("Initialization done.")

# First example
g = Gaussian([ 0.3 0.15; 0.15 0.15 ], [ 0.; 0. ])
h = Gaussian([ 0.1 -0.1; -0.1 0.2 ], [ 0.; 0. ])
make_ex1(g, h, "example1/")
println("First example done.")

# Second example
gaussians = [ Gaussian([ 1. 0.999; 0.999 1. ], [ 0.; 0. ]),
              Gaussian([ 1. -0.999*sqrt(2); -0.999*sqrt(2) 2. ], [ 0.; 0. ]) ]
make_ex2(gaussians, "example2/", 5)

gaussians = [ Gaussian([ 1. 0.; 0. 1. ], [ 0.; 0. ]),
              Gaussian([ 2. 1.; 1. 1. ], [ 0.; 0. ]) ]
make_ex2(gaussians, "example2b/", 5)
println("Second example done.")

# Third example
images = [ "im/star.png", "im/batman.png", "im/cross.png", "im/box.png" ]
make_ex3(images, 10, "example3/", 7)
println("Third example done.")
