include("SubmodularMaximization.jl")
using .SubmodularMaximization

using PyPlot
using LinearAlgebra

num_sample = 1000
limits = [0 1; 0 1]

function evaluate_pdf(p, name)
  println("Evaluating $name")

  println("PDF at center")
  @show pdf(p, [0.5, 0.5])

  figure()
  @time visualize_pdf(p)
  title("$name")

  println("Sampling PDF")
  @time samples = hcat(map(x->sample_reject(p, limits), 1:num_sample)...)

  scatter(samples[1,:]', samples[2,:]', color = "k", marker = (5, 2, 0))
end

g = Gaussian([0.5, 0.5], Diagonal([0.01, 0.1]))
evaluate_pdf(g, "Gaussian")

bimodal_mixture = GaussianMixture([1.0, 1.5],
                                  [Gaussian([0.2, 0.5], Diagonal([0.001, 0.1])),
                                   Gaussian([0.8, 0.5], Diagonal([0.001, 0.1]))])
evaluate_pdf(bimodal_mixture, "Bimodal Mixture")

evaluate_pdf(standard_mixture(), "Standard Mixture")

nothing
