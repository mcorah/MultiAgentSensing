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

# State and trajectory objects for convenience
const State = Tuple{Int64,Int64}
const Trajectory = Vector{State}

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

# Wrap states back into the grid
wrap(g::Grid, s) = mod.(s .- 1, dims(g)) .+ 1

# Produces the out neighbors of the transition graph based on a grid model
# We preallocate the array with the max number of neighbors
neighbors_buffer() = Vector{State}(undef, 5)
function neighbors(g::Grid, state; buffer=neighbors_buffer())
  dirs = ((1,0), (0,1))
  offsets = (-1, 1)

  resize!(buffer, 0)

  for dir in dirs, offset in offsets
    candidate = wrap(g, state .+ offset .* dir)

    push!(buffer, candidate)
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

  function RangingSensor(;variance_constant=0.5, variance_scaling_factor=0.2)
    new(variance_constant, variance_scaling_factor)
  end
end

variance(r::RangingSensor, distance) =
  r.variance_constant + r.variance_scaling_factor * distance^2

stddev(r::RangingSensor, distance) = sqrt(variance(r, distance))

# Distances in each direction are minimum distances around the grid
function mean(g::Grid, r::RangingSensor, a, b)
  diff = abs.(a .- b)

  dists = min.(diff, dims(g) .- diff)

  norm(dists)
end

# sample a ranging observation
function generate_observation(g::Grid, r::RangingSensor, a, b;
                              rng=Random.GLOBAL_RNG)
  m = mean(g, r,a,b)
  m + randn(rng) * stddev(r, m)
end

const normal_lookup_table = NormalLookup(increment=0.001, max=4.0)

likelihoods_buffer(states) = Array{Float64}(undef, size(states))

# Computes (non-normalize) likelihoods of data
function compute_likelihoods(robot_state, target_states, sensor::RangingSensor,
                             grid::Grid,
                             range::Real;
                             buffer = likelihoods_buffer(target_states)
                            )

  for (ii, target_state) in enumerate(target_states)
    distance = mean(grid, sensor, robot_state, target_state)

    buffer[ii] = evaluate_no_norm(normal_lookup_table, error=range - distance,
                                  stddev=stddev(sensor, distance))
  end

  buffer
end

# produces an array of continuous ranges of states in the intput trajectory
# (as the robot may cros over the grid)
continuous(a::State, b::State) = sum(abs.(a .- b)) <= 1
function continuous_ranges(states::Trajectory)
  ret = Vector{State}[]

  # Ends of ranges are states that are discontinuous with the following
  range_ends = findall(1:length(states)-1) do ii
    !continuous(states[ii], states[ii+1])
  end
  push!(range_ends, length(states))

  # Push the ranges of states
  start = 1
  for range_end in range_ends
    push!(ret, states[start:range_end])
    start = range_end + 1
  end

  ret
end
