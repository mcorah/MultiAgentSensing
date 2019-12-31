using Base.Iterators
using Distributions
using Statistics
using SparseArrays
using Random

export Grid, State, get_states, dims, num_states, neighbors, random_state,
  target_dynamics, RangingSensor, generate_observation, compute_likelihoods,
  transition_matrix

import Distributions.mean

abstract type StateSpace end
# methods
#
# get_states

const State = Tuple{Int64,Int64}
struct Grid <: StateSpace
  width::Int64
  height::Int64
  states::Array{State,2}
  transition_matrix::SparseMatrixCSC{Float64,Int64}

  # We will precompute some of the large objects that we use frequently
  function Grid(width, height)
    x = new(width, height, get_states(width, height),
            sparse([], [], Float64[], width*height, width*height)
           )
    x.transition_matrix .= generate_transition_matrix(x)

    x
  end
end
get_states(width::Real, height::Real) = collect(product(1:width, 1:height))
get_states(g::Grid) = g.states
dims(g::Grid) = (g.width, g.height)
num_states(g::Grid) = length(g.states)

in_bounds(g::Grid, state) = all(x->in(x[1],1:x[2]), zip(state, dims(g)))

state_to_index(g::Grid, s) = s[1] + (g.width) * (s[2] - 1)
function index_to_state(g::Grid, x)
  b = div(x-1, g.width) + 1
  a = x - (b - 1) * g.width
  (a, b)
end

# Produces the out neighbors of the transition graph based on a grid model
# We preallocate the array with the max number of neighbors
neighbors_buffer() = Vector{State}(undef, 5)
function neighbors(g::Grid, state; buffer=neighbors_buffer())
  dirs = ((1,0), (0,1))
  offsets = (-1, 1)

  resize!(buffer, 0)

  for dir in dirs, offset in offsets
    candidate = state .+ offset .* dir

    if in_bounds(g, candidate)
      push!(buffer, candidate)
    end
  end

  push!(buffer, state)

  buffer
end

# Returns a left stochastic matrix A so that posterior = A * prior
# This should only be run while initializing the grid
function generate_transition_matrix(g::Grid)
  rows = Int64[]
  columns = Int64[]
  weights = Float64[]

  # Push uniform transition probabilities for each state
  for state in get_states(g)
    ns = neighbors(g, state)

    source = state_to_index(g, state)
    weight = 1 / length(ns)

    for neighbor in ns
      dest = state_to_index(g, neighbor)

      push!(columns, source)
      push!(rows, dest)
      push!(weights, weight)
    end
  end

  size = length(get_states(g))
  sparse(rows, columns, weights, size, size)
end
transition_matrix(grid::Grid) = grid.transition_matrix

random_state(g::Grid; rng=Random.GLOBAL_RNG) = sample(rng, get_states(g))
function target_dynamics(g::Grid, s; rng=Random.GLOBAL_RNG,
                         buffer=neighbors_buffer())

  sample(rng, neighbors(g, s, buffer=buffer))
end

# variance of observations is: constant + scaling * norm_squared
struct RangingSensor
  variance_constant::Float64
  variance_scaling_factor::Float64
end

variance(r::RangingSensor, distance) =
  r.variance_constant + r.variance_scaling_factor * distance^2

stddev(r::RangingSensor, distance) = sqrt(variance(r, distance))

mean(r::RangingSensor, a, b) = norm(a .- b)

# sample a ranging observation
function generate_observation(r::RangingSensor, a, b; rng=Random.GLOBAL_RNG)
  m = mean(r,a,b)
  m + randn(rng) * stddev(r, m)
end

# Compute likeihoods of observations
likelihoods_buffer(states) = Array{Float64}(undef, size(states))
function compute_likelihoods(robot_state, target_states, sensor::RangingSensor,
                             range::Real;
                             buffer = likelihoods_buffer(target_states)
                            )

  for (ii, target_state) in enumerate(target_states)
    distance = mean(sensor, robot_state, target_state)

    buffer[ii] = pdf(Normal(distance, stddev(sensor, distance)), range)
  end

  buffer
end
