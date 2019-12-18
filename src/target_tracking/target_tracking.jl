using Base.Iterators
using Distributions
using Statistics
using SparseArrays

export Grid, State, get_states, dims, neighbors, random_state, target_dynamics,
  RangingSensor, generate_observation, transition_matrix

import Distributions.mean

abstract type StateSpace end
# methods
#
# get_states

struct Grid <: StateSpace
  width::Int64
  height::Int64
end
const State = Tuple{Int64,Int64}
get_states(g::Grid) = collect(product(1:g.width, 1:g.height))
dims(g::Grid) = (g.width, g.height)
num_states(g::Grid) = g.width * g.height

in_bounds(g::Grid, state) = all(x->in(x[1],1:x[2]), zip(state, dims(g)))

state_to_index(g::Grid, s) = s[1] + (g.width) * (s[2] - 1)
function index_to_state(g::Grid, x)
  b = div(x-1, g.width) + 1
  a = x - (b - 1) * g.width
  (a, b)
end

# Produces the out neighbors of the transition graph based on a grid model
function neighbors(g::Grid, state)
  dirs = ((1,0), (0,1))
  offsets = (-1, 1)

  candidates = map(product(dirs, offsets)) do (d, o)
    state .+ o .* d
  end[:]
  push!(candidates, state)

  filter(x->in_bounds(g, x), candidates)
end

# returns a left stochastic matrix A so that posterior = A * prior
function transition_matrix(g::Grid; states = get_states(g))
  rows = Int64[]
  columns = Int64[]
  weights = Float64[]

  # Push uniform transition probabilities for each state
  for state in states
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

  size = length(states)
  sparse(rows, columns, weights, size, size)
end

random_state(g::Grid) = sample(get_states(g))
target_dynamics(g::Grid, s) = sample(neighbors(g, s))

# variance of observations is: constant + scaling * norm_squared
struct RangingSensor
  variance_constant::Real
  variance_scaling_factor::Real
end

variance(r::RangingSensor, distance) =
  r.variance_constant + r.variance_scaling_factor * distance^2

stddev(r::RangingSensor, distance) = sqrt(variance(r, distance))

mean(r::RangingSensor, a, b) = norm(a .- b)

# sample a ranging observation
function generate_observation(r::RangingSensor, a, b)
  m = mean(r,a,b)
  m + randn() * stddev(r, m)
end

# Compute likeihoods of observations
function compute_likelihoods(robot_state, target_states, sensor::RangingSensor,
                             range::Real)

  likelihoods = Array{Float64}(undef, size(target_states))
  for (ii, target_state) in enumerate(target_states)
    distance = mean(sensor, robot_state, target_state)

    likelihoods[ii] = pdf(Normal(distance, stddev(sensor, distance)), range)
  end

  likelihoods
end
