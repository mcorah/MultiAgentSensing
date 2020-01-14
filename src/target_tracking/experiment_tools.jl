# This file provides tools for design of multi-robot target tracking experiments

export target_tracking_experiment, target_tracking_instance,
  iterate_target_tracking!, visualize_experiment

# Run a target tracking experiment for a given number of steps
# By default, keep all data from each step
function target_tracking_experiment(;steps,
                                    num_robots,
                                    num_targets=default_num_targets(num_robots),
                                    configs::MultiRobotTargetTrackingConfigs,
                                    solver=solve_sequential,
                                    step_callback=copy_data
                                   )
  initialization = target_tracking_instance(num_robots=num_robots,
                                            num_targets=num_targets,
                                            configs=configs)

  robot_states = initialization.robot_states
  target_states = initialization.target_states
  target_filters = initialization.target_filters

  # Run target tracking. Store data in whatever the callback returns
  map(1:steps) do _
    # Simulate one step of tracking; update everything in place
    data = iterate_target_tracking!(robot_states=robot_states,
                                    target_states=target_states,
                                    target_filters=target_filters,
                                    configs=configs,
                                    solver=solver)

    step_callback(data)
  end
end

function target_tracking_instance(;num_robots::Int64,
                                  num_targets=default_num_targets(num_robots),
                                  configs::MultiRobotTargetTrackingConfigs)
  grid = configs.grid

  target_states = map(x->random_state(grid), 1:num_targets)

  (
   robot_states = map(x->random_state(grid), 1:num_robots),
   target_states = target_states,
   target_filters = map(x->Filter(grid, x), target_states)
  )
end

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
  for (ii, target_state) in enumerate(target_states)
    target_states[ii] = target_dynamics(configs.grid, target_state)
  end

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

# The default updater modifies the states and filters in place
function copy_data(x)
  (
   robot_states=Array(x.robot_states),
   target_states=Array(x.target_states),
   target_filters=map(Filter, x.target_filters),
   trajectories=x.trajectories,
   range_observations=x.range_observations
  )
end

#
# Visualize experiments
#

function visualize_experiment(data::Vector,
                              configs)
  if length(data) == 0
    println("There is no experiment data to visualize")
    return
  end

  grid = configs.grid

  figure()

  plot_state_space(grid)
  xlim([0, grid.width+1])
  ylim([0, grid.height+1])

  for (ii, step) in enumerate(data)
    println("Step: ", ii)

    plots = visualize_time_step(robot_states=step.robot_states,
                                target_states=step.target_states,
                                target_filters=step.target_filters,
                                trajectories=step.trajectories)

    # End of step (take input and clear the canvas

    line = readline()
    if line == "q"
      break
    end
    foreach(x->x.remove(), plots)
  end
end

function visualize_time_step(;robot_states,
                             target_states,
                             target_filters,
                             trajectories)
  plots=[]

  for robot in robot_states
    append!(plots, plot_trajectory([robot], color=:blue))
  end

  for trajectory in trajectories
    append!(plots, plot_states(trajectory, color=:blue, linestyle=":"))
  end

  for target in target_states
    append!(plots, plot_trajectory([target]))
  end

  append!(plots, visualize_filters(target_filters))

  plots
end
