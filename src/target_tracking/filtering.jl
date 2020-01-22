# General bayesian filtering and updates

using Histograms
using LinearAlgebra
using Base.Iterators
using StatsBase
using Random

import Histograms.generate_prior

export process_update, process_update!, measurement_update, measurement_update!,
  Filter, SparseFilter, get_data, sparsity, drop_below_threshold!

# StatsBase also defines Histogram so we need an alias
const Filter = Histograms.Histogram
const SparseFilter = Histograms.SparseHistogram
const AnyFilter = Histograms.AnyHistogram

Filter(g::Grid) = Filter((1:g.width, 1:g.height))

# Provide a more general method to compute the states in a grid
get_states(g::Grid, _::AnyFilter) = get_states(g)

get_states(::Grid, f::SparseFilter) = get_states(f)
function get_states(f::SparseFilter)
  data = get_data(f)

  col::Int64 = 1

  map(enumerate(rowvals(data))) do (ii, row)
    # Advance the column number until the next contains values past
    # the current(ii) index
    while col < size(data, 2) && data.colptr[col+1] <= ii
      col += 1
    end
    (row, col)
  end
end

# Compute cartesian indices in sparse arrays
index_to_state(f::SparseFilter, ind) = index_to_state(get_data(f), ind)
function index_to_state(data::AbstractSparseMatrix, ind)
  row = rowvals(data)[ind]
  col = findlast(x->x<=ind, data.colptr)
  (row, col)
end

# Constructor with known initial state
function Filter(g::Grid, initial_state::State)
  data = zeros(dims(g))
  data[initial_state...] = 1

  Filter((1:g.width, 1:g.height), data)
end
function SparseFilter(g::Grid, initial_state::State; kwargs...)
  data = spzeros(dims(g)...)
  data[initial_state...] = 1

  SparseFilter((1:g.width, 1:g.height), data; kwargs...)
end

# sample from the prior
function sample_state(grid::Grid, prior::Filter; rng=Random.GLOBAL_RNG)
  @views ind = sample(rng, 1:length(get_data(prior)),
                      Weights(get_data(prior)[:]))
  index_to_state(grid, ind)
end
function sample_state(grid::Grid, prior::SparseFilter; rng=Random.GLOBAL_RNG)
  vals = nonzeros(get_data(prior))

  @views ind = sample(rng, 1:length(vals), Weights(vals))

  # Here we go off our existing knowledge of the state representation...
  index_to_state(prior, ind)
end

# General process update
function process_update!(prior::Filter, transition_matrix)

  @inbounds @views mul!(get_buffer(prior)[:],
                        transition_matrix,
                        get_data(prior)[:])

  swap_buffer!(prior)

  prior
end
function process_update!(prior::SparseFilter, transition_matrix)

  prod = transition_matrix * get_data(prior)[:]

  prior.data = reshape(prod, size(prior))

  prior
end
function process_update(prior::AnyFilter, transition_matrix)
  # Copy
  posterior = duplicate(prior)

  # Perform the update
  process_update!(posterior, transition_matrix)
end


# General measurement update
#
# Note: likelihoods should match the positions of the non-zeros for sparse
# matrices
function measurement_update!(prior::AnyFilter, likelihoods::Vector{Float64})
  vals = get_values(prior)

  # update belief in place
  @inbounds @simd for ii in 1:length(vals)
    vals[ii] *= likelihoods[ii]
  end

  # normalize
  s = sum(vals)
  @inbounds @simd for ii in 1:length(vals)
    vals[ii] /= s
  end

  prior
end
function measurement_update(prior::AnyFilter, xs...; kwargs...)
  # Copy
  posterior = duplicate(prior)

  # Perform the update
  measurement_update!(posterior, xs...; kwargs...)
end

# compute measurement update by computing likelihood
function measurement_update!(prior::AnyFilter, xs...; kwargs...)
  likelihoods = compute_likelihoods(xs...; kwargs...)
  measurement_update!(prior, likelihoods)
end
