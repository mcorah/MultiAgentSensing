# Implements the coverage objective for the target coverage problem

using LinearAlgebra

export finite_horizon_coverage

# A target is covered if any robot is in range
function is_covered(sensor::CoverageSensor, target_state::State,
                    robot_state::State)
  distance = norm(target_state .- robot_state)

  distance <= sensor.range_limit
end
function is_covered(sensor::CoverageSensor, target_state::State,
                    robot_states::Vector{State})
  any(robot_states) do robot_state
    is_covered(sensor, target_state, robot_state)
  end
end

# Returns coverage for a target state distribution
#
# Coverage (f) is conditional on prior observations:
# * f(states, prior_states) - f(prior_states)
function target_coverage(target_state_distribution::AnyFilter,
                         sensor::CoverageSensor,
                         primary_states::Vector{State};
                         prior_states::Vector{State} = State[])
  state_distribution_data = Histograms.get_values(target_state_distribution)

  # The coverage value is the sum of probabilities for covered states that are
  # not covered by "prior" states (coverage conditional on prior states)
  # * Note that the target state distribution should always have non-empty
  #   support (so we do not have to worry about empty arrays)
  sum(zip(state_distribution_data.nzind,
          state_distribution_data.nzval)) do (index, probability)
    target_state = index_to_state(target_state_distribution, index)

    # Covered and not previously covered
    newly_covered =
      is_covered(sensor, target_state, primary_states) &&
      !is_covered(sensor, target_state, prior_states)

    newly_covered ? probability : 0.0
  end
end

# Returns the sum of probabilities that a given target will be covered
# TODO: Add a discount
#
# Note: prior trajectories encode conditioning.
# This returns coverage (f) where conditional coverage is:
# f(trajectories, prior_trajectories) - f(prior_trajectories)
function finite_horizon_coverage(grid::Grid, target_state::State,
                                 sensor::CoverageSensor,
                                 trajectories::Vector{Trajectory};
                                 prior_trajectories::Vector{Trajectory}=
                                 Trajectory[])

  horizon = length(first(trajectories))

  # We will include coverage from the initial state on
  # (note that coverage at the initial state is deterministic)
  incremental_coverage = Vector{Float64}(undef, horizon)

  # Initialize a sparse target distribution. Note that the threshold will not
  # come into play as states will not be filtered.
  target_state_distribution = SparseFilter(grid, target_state, threshold=0.0)

  # Iterate over the horizon
  for ii in 1:horizon
    # Extract the current state of each robot
    primary_states = [trajectory[ii] for trajectory in trajectories]
    prior_states = [trajectory[ii] for trajectory in prior_trajectories]

    # Check whether the target is in range

    # Check whether uncovered by previous
    coverage_value = target_coverage(target_state_distribution, sensor,
                                     primary_states, prior_states=prior_states)

    incremental_coverage[ii] = coverage_value

    # Propagate the state forward (sparsely)
    process_update!(target_state_distribution, transition_matrix(grid))
  end

  # Return reward
  (
   reward = sum(incremental_coverage),
   incremental_coverage = incremental_coverage
  )
end

# Package a single trajectory in an array
function finite_horizon_coverage(grid, prior, sensor,
                                    trajectory::Trajectory;
                                    kwargs...
                                   )

  finite_horizon_coverage(grid, prior, sensor, [trajectory]; kwargs...)
end
