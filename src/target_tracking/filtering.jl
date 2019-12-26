# General bayesian filtering and updates

using Histograms
using LinearAlgebra
using Base.Iterators
using StatsBase
using Random

import Histograms.generate_prior

export process_update, process_update!, measurement_update, measurement_update!,
  Filter, get_data

# StatsBase also defines Histogram so we need an alias
const Filter = Histograms.Histogram

Filter(g::Grid) = Filter((1:g.width, 1:g.height))

# sample from the prior
function sample_state(grid::Grid, prior::Filter; rng=Random.GLOBAL_RNG)
  ind = sample(rng, 1:length(get_data(prior)), Weights(get_data(prior)[:]))
  index_to_state(grid, ind)
end

# General process update
function process_update!(prior::Filter, transition_matrix)
  @inbounds @views get_data(prior)[:] .= transition_matrix * get_data(prior)[:]

  prior
end
function process_update(prior::Filter, transition_matrix)
  # Copy
  posterior = Filter(prior)

  # Perform the update
  process_update!(posterior, transition_matrix)
end


# General measurement update
function measurement_update!(prior::Filter, likelihoods::Array{Float64})
  data = get_data(prior)

  # update belief in place
  data .= data .* likelihoods

  # normalize
  data .= data ./ sum(data)

  prior
end
function measurement_update(prior::Filter, xs...)
  # Copy
  posterior = Filter(prior)

  # Perform the update
  measurement_update!(posterior, xs...)
end

# compute measurement update by computing likelihood
function measurement_update!(prior::Filter, xs...)
  likelihoods = compute_likelihoods(xs...)
  measurement_update!(prior, likelihoods)
end
