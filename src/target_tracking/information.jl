# Implements the mutual information objective for the target tracking problem

using LinearAlgebra
using Base.Iterators
using Statistics
using Random

export finite_horizon_information, entropy, TimedObservation

import Base.length

nan_to_zero(x::Float64) = ifelse(isnan(x), 0.0, x)

entropy_sum(prior::AbstractFilter) = sum(x -> nan_to_zero(-x * log(2, x)),
                                         HistogramFilters.get_values(prior))
function entropy(prior::Filter)
  sum = 0.0
  @inbounds @simd for x = HistogramFilters.get_values(prior)
    sum += nan_to_zero(-x * log(2, x))
  end
  sum
end
function entropy(prior::SparseFilter)
  sum = 0.0
  @inbounds @simd for x = nonzeros(HistogramFilters.get_values(prior))
    sum += nan_to_zero(-x * log(2, x))
  end
  sum
end

function entropy(prior::Array{<:AbstractFilter})
  sum(entropy, prior)
end

#
# Measurement representation and updates
#
# Most of the sequential planning utilizes the trajectory objects. I prefer to
# avoid further processing of the trajectories as this code runs a lot.
#
# I also include a more general representation TimedObservation to support
# mutual information for collections of observations at specific times
#

#
# Timed observations allow flexibility in scheduling observations
#
struct TimedObservation
  step::Int64
  state::State
end
# Object that manages and sorts observations
mutable struct TimedObservations
  observations::Vector{TimedObservation}
  index::Int64

  function TimedObservations(observations::Vector{TimedObservation})
    new(sort(observations, by=x->x.step), 1)
  end
end
TimedObservations() = TimedObservations(TimedObservation[])
length(x::TimedObservations) = length(x.observations)
access(x::TimedObservations) = x.observations[x.index]
advance!(x::TimedObservations) = (x.index += 1; nothing)
all_processed(x::TimedObservations) = x.index > length(x)

#
# Combine sets of observations (so that we can obtain the set of all
# observations from the union of the prior and posterior)
#
function combine_observations(x::Vararg{X}) where
  X <: Union{Vector{Trajectory}, Vector{TimedObservation}}

  vcat(x...)
end

#
# Simulate observations and update the filter for a given time-step
#

# Assumption: all trajectories are at least as long as "step"
function simulate_update_filter!(grid::Grid, filter::AbstractFilter,
                                 sensor::RangingSensor, target_state::State,
                                 trajectories::Vector{Trajectory}, step;
                                 rng=Random.GLOBAL_RNG,
                                 likelihood_buffer =
                                  likelihoods_buffer(get_states(grid, filter))
                                )

    for trajectory in trajectories
      robot_state = trajectory[step]

      range_observation = generate_observation(grid, sensor, robot_state,
                                               target_state; rng=rng)

      measurement_update!(filter, robot_state, get_states(grid, filter), sensor,
                          grid, range_observation, buffer=likelihood_buffer)
    end
end
# Precondition: all observations before "step" have already been processed.
# (this is reasonable because the filter update design would prevent processing
# those observations afterward. As such, observations that precede the scope of
# updates are malformed)
function simulate_update_filter!(grid::Grid, filter::AbstractFilter,
                                 sensor::RangingSensor, target_state::State,
                                 observations::TimedObservations, step;
                                 rng=Random.GLOBAL_RNG,
                                 likelihood_buffer =
                                  likelihoods_buffer(get_states(grid, filter))
                                )

  # Process all observations at the current step
  #
  # Note that observations are sorted and that observations before the current
  # steps *have already been processed*
  while !all_processed(observations) && access(observations).step == step
    robot_state = access(observations).state

    range_observation = generate_observation(grid, sensor, robot_state,
                                             target_state; rng = rng)

    measurement_update!(filter, robot_state, get_states(grid, filter), sensor,
                        grid, range_observation, buffer=likelihood_buffer)

    advance!(observations)
  end

  nothing
end

# Sum mutual information objective as by Ryan and Hedrick.
# Computes:
#
# sum_i=1:t I(X_i;Y_1,...,Y_t)
#
# with observations for multiple robots. This is not the exact mutual
# information for the horizon but an upper bound.
#
# prior: prior at planning time (the first observation in the horizon,
# occurs after the first process update)
#
# sensor: sensor model (common to robots)
#
# trajectories: array of trajectories over the current horizon. Each trajectory
# is an array of states (trajectories then an array of arrays). The first state
# in each trajectory corresponds to the robot's position at the next time-step
# (after one process update on the prior)
#
const default_num_information_samples = 1000

# Package a single trajectory in an array
function finite_horizon_information(grid::Grid, prior::AbstractFilter,
                                    sensor::RangingSensor,
                                    observation::Trajectory;
                                    kwargs...
                                   )

  finite_horizon_information(grid, prior, sensor, [observation]; kwargs...)
end
# Pull the number of time-steps from a vector of trajectories as appropriate
# and drop that into the keyword arguments
function finite_horizon_information(grid::Grid, prior::AbstractFilter,
                                    sensor::RangingSensor,
                                    observations::Vector{Trajectory};
                                    prior_observations::Vector{Trajectory}=
                                      Trajectory[],
                                    kwargs...
                                   )

  if length(observations) == 0
    error("Please supply at least one trajectory for posterior")
  end

  horizon = length(observations[1])

  if horizon == 0
    error("Information horizon is zero")
  end

  for trajectories in (observations, prior_observations)
    for trajectory in trajectories
      if length(trajectory) != horizon
        @show observations
        @show prior_observations
        error("Information trajectory lengths do not match")
      end
    end
  end

  finite_horizon_information(grid, prior, sensor, observations, horizon;
                             prior_observations=prior_observations,
                             kwargs...)
end

# Base implementation of the information objective for finite horizons that is
# called by all others
#
# (Note from above that this is sum of conditional mutual informations at each
# step)
information_counter = Threads.Atomic{Int64}(0)
const samples_per_gc = 100_000
function finite_horizon_information(grid::Grid, prior::AbstractFilter,
                                    sensor::RangingSensor,
                                    posterior_observations::O,
                                    horizon::Integer;
                                    prior_observations::O = O(),
                                    num_samples::Integer =
                                      default_num_information_samples,
                                    kwargs...
                                   ) where O <: Union{Vector{TimedObservation},
                                                      Vector{Trajectory}}

  # add one to the ccounter, and add one to the old value
  old = Threads.atomic_add!(information_counter, num_samples)
  new = old + num_samples
  # Run the garbage collector if we pass a sample boundary
  if div(new, samples_per_gc) > div(old, samples_per_gc)
    GC.gc()
  end

  prior_buffer = HistogramFilters.duplicate(prior)
  reset_prior() = copy_filter!(prior, out=prior_buffer)

  # Compute entropies over the horizon, conditional on the prior trajectories
  # to compute the entropies:
  #
  # sum_i=1^t H(Xi|Y_{prior_observations}, belief)
  prior_entropies = mean(1:num_samples) do _
    reset_prior()
    sample_finite_horizon_entropy!(grid, prior_buffer, sensor,
                                   prior_observations, horizon; kwargs...)
  end

  # Compute entropies over the horizon conditional on the input trajectory
  # (produces an array with one entry per time-step) to compute entropies:
  #
  # sum_i=1^t H(Xi|Y_{all_trajectories}, belief)
  all_observations = combine_observations(prior_observations,
                                          posterior_observations)
  conditional_entropies = mean(1:num_samples) do _
    reset_prior()
    sample_finite_horizon_entropy!(grid, prior_buffer, sensor,
                                   all_observations, horizon; kwargs...)
  end

  (
   reward = sum(prior_entropies - conditional_entropies),
   mutual_information = prior_entropies - conditional_entropies,
   prior_entropies = prior_entropies,
   conditional_entropies = conditional_entropies
  )
end

# We compute the mutual information by sampling the conditional entropy given
# ranging observations.
#
# This method samples the observations and computes the entropy
#
# Returns an array of entropies with entries for each step
# *on which entropy is evaluated*
#
# The "entropy_only_at_end" keyword argument allows the designer
# to evaluate entropy at only one time-step (the last in the horizon)
#
#
# WARNING: We operate on the prior/filter in place! It is up to the user to
# maintain the original.

# Package timed observations before passing them into the main implementation
function sample_finite_horizon_entropy!(grid::Grid, prior::AbstractFilter,
                                       sensor::RangingSensor,
                                       observations::Vector{TimedObservation},
                                       b...;
                                       kwargs...)

  sample_finite_horizon_entropy!(grid, prior, sensor,
                                TimedObservations(observations), b...;
                                kwargs...)
end
function sample_finite_horizon_entropy!(grid::Grid, prior::AbstractFilter,
                                       sensor::RangingSensor,
                                       observations::Union{Vector{Trajectory},
                                                           TimedObservations},
                                       horizon::Integer;
                                       rng = Random.GLOBAL_RNG,
                                       entropy_only_at_end=false
                                      )

  target_state = sample_state(grid, prior; rng=rng)

  # Simulate trajectories and compute updates
  #
  # Sample ranging observations
  # (by also sampling sampling target trajectories)
  # Note: we will reuse samples accross the horizon

  conditional_entropies = Float64[]

  neighbor_buffer = neighbors_buffer()
  likelihood_buffer = likelihoods_buffer(get_states(grid, prior))

  for step = 1:horizon
    # Update from prior state to the first time-step in the horizon
    target_state = target_dynamics(grid, target_state, buffer=neighbor_buffer)

    process_update!(prior, transition_matrix(grid))

    # Sample observations based on the target state and each robot's state at
    # the given time-step, and then perform measurement updates
    simulate_update_filter!(grid, prior, sensor, target_state, observations,
                            step; rng=rng, likelihood_buffer=likelihood_buffer)

    # Evaluate entropy if evaluating entropy at all time-steps or if at the last
    # time-step
    if !entropy_only_at_end || step == horizon
      push!(conditional_entropies, entropy(prior))
    end
  end

  conditional_entropies
end
