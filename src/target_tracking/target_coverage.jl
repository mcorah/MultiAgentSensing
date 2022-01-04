# Implements the coverage objective for the target coverage problem

export finite_horizon_coverage

# Returns the sum of probabilities that a given target will be covered
# TODO: Add a discount
function finite_horizon_coverage(grid::Grid, target_state::State,
                                 sensor::CoverageSensor,
                                 trajectories::Vector{Trajectory};
                                 prior_observations::Vector{Trajectory}=
                                 Trajectory[])
  # Propagate the state forward (sparsely)

  # Check whether the target is in range

  # Return reward
  incremental_coverage = [1.0, 1.0]

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
