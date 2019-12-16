using Base.Iterators
using Distributions
using Statistics

export Grid, get_states, dims, neighbors, random_state, target_dynamics,
  RangingSensor, generate_observation

import Distributions.mean

abstract type StateSpace end
# methods
#
# get_states

struct Grid <: StateSpace
  width::Int64
  height::Int64
end
get_states(g::Grid) = collect(product(1:g.width, 1:g.height))
dims(g::Grid) = (g.width, g.height)

in_bounds(g::Grid, state) = all(x->in(x[1],1:x[2]), zip(state, dims(g)))

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
