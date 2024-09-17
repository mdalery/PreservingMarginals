using LinearAlgebra,
      Plots,
      Plots.PlotMeasures,
      LaTeXStrings,
      JLD2

using PreservingMarginals


###################################################### MAKE PLOTS GAUSSIANS ######################################


function make_ex1( g::Gaussian{2}, h::Gaussian{2}, save_string::String = "" )
    gx = marg(g, 1)
    gy = marg(g, 2)
    hx = marg(h, 1)
    hy = marg(h, 2)

    # Plot marg <-> bar gaussiennes

    λs = 0.1:0.2:0.9
    m = max(g.Σ[1, 1], h.Σ[1, 1])
    xs = -5*m:10*m/1000:5*m
    m = max(g.Σ[2, 2], h.Σ[2, 2])
    ys = -5*m:10*m/1000:5*m
    valsx = []
    valsy = []
    to_maxx = []
    to_maxy = []
    min = 0.
    for λ in λs
        b = barycenter([ g, h ], [ 1-λ, λ ])
        margxb = marg(b, 1)
        margyb = marg(b, 2)
        bx = barycenter([ gx, hx ], [ 1-λ, λ ])
        by = barycenter([ gy, hy ], [ 1-λ, λ ])

        vals_margxb = margxb.(xs)
        vals_bx = bx.(xs)
        vals_margyb = margyb.(xs)
        vals_by = by.(xs)

        push!(to_maxx, vals_margxb)
        push!(to_maxx, vals_bx)
        push!(to_maxy, vals_margyb)
        push!(to_maxy, vals_by)

        push!(valsx, (vals_margxb, vals_bx))
        push!(valsy, (vals_margyb, vals_by))
    end

    maxx = maximum(vcat(to_maxx...))
    plotsx = []
    for (vals_margxb, vals_bx) in valsx
        push!(plotsx, plot(xs, [ vals_margxb, vals_bx ],
                           linewidth = 2,
                           linestyle = [ :solid :dash ],
                           legend = false,
                           left_margin = 5mm,
                           ytickfonthalign = :left,
                           xtickfontvalign = :bottom,
                           ylim = (0., maxx)))
    end

    maxy = maximum(vcat(to_maxy...))
    plotsy = []
    for (vals_margyb, vals_by) in valsy
        push!(plotsy, plot(xs, [ vals_margyb,vals_by ],
                           linewidth = 2,
                           linestyle = [ :solid :dash ],
                           legend = false,
                           left_margin = 5mm,
                           ytickfonthalign = :left,
                           xtickfontvalign = :bottom,
                           ylim = (0., maxy)))
    end

    pmargx = plot(plotsx..., size = (1500, 300), layout = grid(1, 5))
    pmargy = plot(plotsy..., size = (1500, 300), layout = grid(1, 5))

    savefig(pmargx, save_string * "pmargx.pdf")
    savefig(pmargy, save_string * "pmargy.pdf")

    # Plot marg <-> bar projections

    pbxs = Float64[]
    pbys = Float64[]
    bxs = Float64[]
    bys = Float64[]
    λs = 0:0.01:1
    for λ in λs
        b = barycenter([g, h], [1-λ, λ])
        push!(pbxs, b.Σ[1, 1])
        push!(pbys, b.Σ[2, 2])

        bx = barycenter([gx, hx], [1-λ, λ])
        push!(bxs, bx.Σ[1])

        by = barycenter([gy, hy], [1-λ, λ])
        push!(bys, by.Σ[1])
    end

    ppx = plot(λs, [pbxs, bxs],
                   linewidth = 2,
                   linestyle = [:solid :dash],
                   legend = false,
                   xlabel = L"\lambda",
                   size = (500, 300))

    ppy = plot(λs, [pbys, bys],
                   linewidth = 2,
                   linestyle = [:solid :dash],
                   legend = false,
                   xlabel = L"\lambda",
                   size = (500, 300))

    savefig(ppx, save_string * "ppx.pdf")
    savefig(ppy, save_string * "ppy.pdf")

    # 2D gaussians

    λs = 0.1:0.2:0.9
    m = max(g.Σ[1, 1], h.Σ[1, 1])
    xs = -5*m:10*m/1000:5*m
    m = max(g.Σ[2, 2], h.Σ[2, 2])
    ys = -5*m:10*m/1000:5*m
    values = []
    for λ in λs
        b = barycenter([g, h], [1-λ, λ])

        push!(values, [ b(x, y) for y in ys, x in xs ])
    end

    maxcontour = maximum(vcat(values...))
    contours = []
    for value in values
        push!(contours, contour(xs, ys, value,
                                colorbar = :false,
                                levels = 5,
                                left_margin = 5mm,
                                ytickfonthalign = :left,
                                xtickfontvalign = :bottom,
                                zlim = (0., maxcontour)))
    end

    pbar = plot(contours..., size = (1500, 300), layout = grid(1, 5))
    savefig(pbar, save_string * "contours_original.pdf")
end


function make_ex2( gaussians::Vector{Gaussian{2}}, save_string::String = "", nb_images::Int64 = 7 )
    @assert length(gaussians) == 2

    m = maximum([ g.Σ[1, 1] for g in gaussians ])
    xs = -2*m:4*m/1000:2*m
    m = maximum([ g.Σ[2, 2] for g in gaussians ])
    ys = -2*m:4*m/1000:2*m
    values_bar = []
    values_pmc = []
    values_pmmp = []
    for i in 1:nb_images
        tx = (i-1) / (nb_images - 1)
        λ = (1 - tx) * [ 1.; 0. ] + tx * [ 0.; 1. ]

        b = barycenter(gaussians, λ)
        push!(values_bar, [ b(x, y) for y in ys, x in xs ])
        b = pmc_barycenter(gaussians, λ, 1)
        push!(values_pmc, [ b(x, y) for y in ys, x in xs ])
        b = pmmp_barycenter(gaussians, λ)
        push!(values_pmmp, [ b(x, y) for y in ys, x in xs ])
    end

    maxbar = maximum(vcat(values_bar...))
    maxpmc = maximum(vcat(values_pmc...))
    maxpmmp = maximum(vcat(values_pmmp...))
    contours = []
    contours_pmc = []
    contours_pmmp = []
    for (val_bar, val_pmc, val_pmmp) in zip(values_bar, values_pmc, values_pmmp)
        push!(contours, contour(xs, ys, val_bar,
                                colorbar = :false,
                                levels = 5,
                                linewidth = 2,
                                formatter = Returns(""),
                                zlim = (0., maxbar)))

        push!(contours_pmc, contour(xs, ys, val_pmc,
                                    colorbar = :false,
                                    levels = 5,
                                    linewidth = 2,
                                    formatter = Returns(""),
                                    zlim = (0., maxpmc)))

        push!(contours_pmmp, contour(xs, ys, val_pmmp, 
                                     colorbar = :false,
                                     levels = 5,
                                     linewidth = 2,
                                     formatter = Returns(""),
                                     zlim = (0., maxpmmp)))
    end

    pbar = plot(contours..., size = (nb_images * 512, 512), layout = grid(1, nb_images))
    savefig(pbar, save_string * "contours_original.pdf")

    pbarpmc = plot(contours_pmc..., size = (nb_images * 512, 512), layout = grid(1, nb_images))
    savefig(pbarpmc, save_string * "contours_pmc.pdf")

    pbarpmmp = plot(contours_pmmp..., size = (nb_images * 512, 512), layout = grid(1, nb_images))
    savefig(pbarpmmp, save_string * "contours_pmmp.pdf")

    # Comparison off-diagonal term two modifications

    λs = 0:0.01:1
    odmod1 = Float64[]
    odmod2 = Float64[]
    for λ in λs
        b = pmc_barycenter(gaussians, [ 1-λ, λ ], 1)
        offdiag = b.Σ[1, 2]
        push!(odmod1, offdiag)

        b = pmmp_barycenter(gaussians, [ 1-λ, λ ])
        offdiag = b.Σ[1, 2]
        push!(odmod2, offdiag)
    end

    pcompxy = plot(λs, [ odmod1, odmod2 ],
                   linewidth = 2,
                   linestyle = [ :solid :dash ],
                   legend = false,
                   xlabel = L"\lambda",
                   size = (500, 300))

    savefig(pcompxy, save_string * "pcompxy.pdf")
end


###################################################### MAKE PLOTS MIXTURES ############################################################"


function make_ex3( images::Vector{String}, N::Int64, save_string::String = "", nb_images::Int64 = 7 )
    @assert length(images) == 4

    v1 = [ 1.; 0.; 0.; 0. ]
    v2 = [ 0.; 1.; 0.; 0. ]
    v3 = [ 0.; 0.; 1.; 0. ]
    v4 = [ 0.; 0.; 0.; 1. ]
    ms = [ im_to_mixture(image, N) for image in images ]

    xs = 0:1.0:512
    ys = 0:1.0:512
    originals = []
    save_originals = Mixture{2, Gaussian{2}}[]
    modifieds = []
    save_modifieds = Mixture{2, Gaussian{2}}[]
    distances = Float64[]
    for i in 1:nb_images
        for j in 1:nb_images
            tx = (i-1) / (nb_images - 1)
            ty = (j-1) / (nb_images - 1)
            tmp1 = (1 - tx) * v1 + tx * v2
            tmp2 = (1 - tx) * v3 + tx * v4
            λ = (1 - ty) * tmp1 + ty * tmp2

            dist, bar, mod = apmc_barycenter(ms, λ, 1)
            push!(distances, dist)

            push!(save_originals, bar)
            push!(originals, contour(xs, ys, [ bar(x, y) for y in ys, x in xs ],
                                     colorbar = false,
                                     linewidth = 2,
                                     levels = 5,
                                     ticks = false,
                                     showaxis = false,
                                     grid = false))

            push!(save_modifieds, mod)
            push!(modifieds, contour(xs, ys, [ mod(x, y) for y in ys, x in xs ],
                                     colorbar = false,
                                     linewidth = 2,
                                     levels = 5,
                                     ticks = false,
                                     showaxis = false,
                                     grid = false))

            println("Contour ($i, $j) done.")
        end
    end

    poriginals = plot(originals..., size = nb_images .* (512, 512), layout = grid(nb_images, nb_images))
    savefig(poriginals, save_string * "originals.pdf")
    save_object(save_string * "originals.jld2", save_originals)

    pmodifieds = plot(modifieds..., size = nb_images .* (512, 512), layout = grid(nb_images, nb_images))
    savefig(pmodifieds, save_string * "modifieds.pdf")
    save_object(save_string * "modifieds.jld2", save_modifieds)
    
    save_object(save_string * "distances.jld2", distances)
end
