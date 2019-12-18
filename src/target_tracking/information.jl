using LinearAlgebra
using Base.Iterators
using Statistics

export finite_horizon_information, entropy

entropy(prior::Filter) = sum(x -> -x * log(2, x), get_data(prior))


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
function finite_horizon_information(grid::Grid, prior::Filter,
                                  sensor::RangingSensor, trajectories;
                                  num_samples::Integer = 1000,
                                  transition_matrix = transition_matrix(grid),
                                  grid_states = get_states(grid))
  steps = length(trajectories)

  if steps == 0
    println("Information horizon is zero")
    return 0
  end

  if !all(length.(trajectories) .== steps)
    println("Information trajectory lengths do not match")
    return 0
  end

  # Compute entropy over horizon
  process_entropies = Array{Float64}(undef, steps)
  let prior = deepcopy(prior)
    for ii = 1:steps
      process_update!(prior, transition_matrix)
      process_entropies[ii] = entropy(prior)
    end
  end

  # Compute conditional entropies over the horizon (produces an array with one
  # entry per time-step
  conditional_entropies =
  mean(1:num_samples) do _
    sample_finite_horizon_entropy(grid, prior, sensor, trajectories;
                                  transition_matrix = transition_matrix,
                                  grid_states = grid_states)
  end

  (
   reward = sum(process_entropies - conditional_entropies),
   mutual_information = process_entropies - conditional_entropies,
   process_entropies = process_entropies,
   conditional_entropies = conditional_entropies
  )
end

# We compute the mutual information by sampling the conditional entropy given
# ranging observations.
#
# This method samples the observations and computes the entropy
#
# Returns an array of entropies with entries for each step
function sample_finite_horizon_entropy(grid::Grid, prior::Filter,
                                   sensor::RangingSensor,
                                   trajectories;
                                   transition_matrix = transition_matrix(grid),
                                   grid_states = get_states(grid))
  steps = length(trajectories)

  if steps == 0
    println("Information horizon is zero")
    return 0
  end

  if !all(length.(trajectories) .== steps)
    println("Information trajectory lengths do not match")
    return 0
  end

  target_state = sample_state(grid, prior)

  # Simulate trajectories and compute updates
  #
  # Sample ranging observations
  # (by also sampling sampling target trajectories)
  # Note: we will reuse samples accross the horizon
  filter = deepcopy(prior)
  conditional_entropies = Array{Float64}(undef, steps)
  for step = 1:steps
    # Update from prior state to the first time-step in the horizon
    target_state = target_dynamics(grid, target_state)
    process_update!(filter, transition_matrix)

    # Sample observations based on the target state and each robot's state at
    # the given time-step
    #
    # Afterward, perform measurement updates
    for trajectory in trajectories
      robot_state = trajectory[step]

      range_observation = generate_observation(sensor, robot_state,
                                               target_state)


      measurement_update!(filter, robot_state, grid_states, sensor,
                          range_observation)
    end

    conditional_entropies[step] = entropy(filter)
  end

  conditional_entropies
end
