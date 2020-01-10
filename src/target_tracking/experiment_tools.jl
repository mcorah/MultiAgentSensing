# This file provides tools for design of multi-robot target tracking experiments

export iterate_target_tracking!

#function target_tracking_experiment()
#end

function iterate_target_tracking!(;robot_states::Vector{State},
                                  target_states::Vector{State},
                                  target_filters::Vector{<:Filter},
                                  configs::MultiRobotTargetTrackingConfigs,
                                  solver=solve_sequential)
  #
  # Update robot and target states
  #

  problem = MultiRobotTargetTrackingProblem(robot_states, target_filters,
                                            configs)
  solution = solver(problem)

  for (index, trajectory) in solution.elements
    robot_states[index] = trajectory[1]
  end
  trajectories = map(last, solution.elements)

  # Update Target states
  target_states = map(x->target_dynamics(configs.grid, x), target_states)

  #
  # Run filter updates (corresponding to new target states) and provide robots
  # with observations
  #

  # Process update
  for filter in target_filters
    process_update!(filter, transition_matrix(configs.grid))
  end

  # Sample ranging observations
  # (output is an array of arrays of robots' observations)
  range_observations = map(robot_states) do robot
    map(target_states) do target
      generate_observation(configs.grid, configs.sensor, robot, target)
    end
  end

  # Measurement update
  for (robot, observations) in zip(robot_states, range_observations)
    for (filter, observation) in zip(target_filters, observations)
      measurement_update!(filter, robot, get_states(configs.grid),
                          configs.sensor, configs.grid, observation)
    end
  end

  # Note: by default, we do not copy the filters
  (
   robot_states=robot_states,
   target_states=target_states,
   target_filters=target_filters,
   trajectories=trajectories,
   range_observations=range_observations
  )
end

# The default updater updates filters in place so we provide the option to copy
function copy_filters(x)
  (
   robot_states=x.robot_states,
   target_states=x.target_states,
   target_filters=map(Filter, x.target_filters),
   trajectories=x.trajectories,
   range_observations=x.range_observations
  )
end
