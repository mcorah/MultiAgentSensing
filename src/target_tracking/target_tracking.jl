using Base.Iterators
using Distributions
using Statistics
using SparseArrays
using Random

export Grid, State, get_states, dims, num_states, neighbors, random_state,
  target_dynamics, RangingSensor, CoverageSensor, generate_observation,
  compute_likelihoods, transition_matrix, default_num_targets, continuous

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

# Construct default grids according to the number of robots
const grid_cells_per_robot = 12.5
function Grid(;num_robots)
  grid_size = round(Int64, sqrt(grid_cells_per_robot * num_robots))
  Grid(grid_size, grid_size)
end

# default number of targets as a function of the number of robots
const targets_per_robot = 1
default_num_targets(num_robots) = round(Int64, num_robots * targets_per_robot)

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

#################
# Ranging sensors
#################

# variance of observations is: constant + scaling * norm_squared
struct RangingSensor
  variance_constant::Float64
  variance_scaling_factor::Float64
  range_limit::Float64

  function RangingSensor(;variance_constant=0.25,
                          variance_scaling_factor=0.5,
                          range_limit=10)
    new(variance_constant, variance_scaling_factor, range_limit)
  end
end

saturate_range(r::RangingSensor, distance) = min(r.range_limit, distance)

# Sensor forward distribution variance and standard deviation
sensor_variance(r::RangingSensor, distance) =
  (r.variance_constant
   + r.variance_scaling_factor * saturate_range(r, distance)^2)

sensor_stddev(r::RangingSensor, distance) = sqrt(sensor_variance(r, distance))

# Mean value for the sensor distribution
# * Saturated distance
sensor_mean(g::Grid, r::RangingSensor, a, b) = saturate_range(r, norm(a .- b))

# sample a ranging observation
function generate_observation(g::Grid, r::RangingSensor, a, b;
                              rng=Random.GLOBAL_RNG)
  m = sensor_mean(g, r,a,b)
  m + randn(rng) * sensor_stddev(r, m)
end

const normal_lookup_table = NormalLookup(increment=0.001, max=4.0)

likelihoods_buffer(states) = Array{Float64}(undef, length(states))

# Computes (non-normalized) likelihoods of data
function compute_likelihoods(robot_state, target_states, sensor::RangingSensor,
                             grid::Grid,
                             range::Real;
                             buffer = likelihoods_buffer(target_states)
                            )

  resize!(buffer, length(target_states))

  for (ii, target_state) in enumerate(target_states)
    distance = sensor_mean(grid, sensor, robot_state, target_state)

    buffer[ii] = pdf(Normal(distance, sensor_stddev(sensor, distance)), range)
  end

  buffer
end

##################
# Coverage sensors
##################

struct CoverageSensor
  range_limit::Float64

  function CoverageSensor(; range_limit=2)
    new(range_limit)
  end
end

# produces an array of continuous ranges of states in the intput trajectory
# (as the robot may cros over the grid)
continuous(a::State, b::State) = sum(abs.(a .- b)) <= 1
function continuous(states::Trajectory)
  for ii = 2:length(states)
    if !continuous(states[ii-1], states[ii])
      return false
    end
  end

  true
end
function continuous(grid::Grid, states::Trajectory)
  for ii = 2:length(states)
    if ! in(states[ii], neighbors(grid, states[ii-1]))
      return false
    end
  end

  true
end


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
