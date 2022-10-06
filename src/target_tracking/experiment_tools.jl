# This file provides tools for design of multi-robot target tracking experiments

using JLD2
using Printf
using Random
using Base.Threads
using Base.Iterators

export target_tracking_experiment, target_tracking_instance,
  iterate_target_tracking!, iterate_target_coverage!, visualize_experiment,
  visualize_time_step, run_experiments

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
                                  target_filters::Vector{<:AbstractFilter},
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
      measurement_update!(filter, robot, get_states(configs.grid, filter),
                          configs.sensor, configs.grid, observation)
    end
  end

  # Note: by default, we do not copy the filters
  (
   robot_states=robot_states,
   target_states=target_states,
   target_filters=target_filters,
   trajectories=trajectories,
   range_observations=range_observations,
   objective=solution.value
  )
end

function iterate_target_coverage!(;robot_states::Vector{State},
                                  target_states::Vector{State},
                                  configs::MultiRobotTargetCoverageConfigs,
                                  solver=solve_sequential)
  #
  # Update robot and target states
  #

  problem = MultiRobotTargetCoverageProblem(robot_states, target_states,
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

  # Note: by default, we do not copy the filters
  (
   robot_states=robot_states,
   target_states=target_states,
   trajectories=trajectories,
   objective=solution.value
  )
end

# The default updater modifies the states and filters in place
function copy_data(x)
  (
   robot_states=Array(x.robot_states),
   target_states=Array(x.target_states),
   target_filters=map(HistogramFilters.duplicate, x.target_filters),
   trajectories=x.trajectories,
   range_observations=x.range_observations,
   objective=x.objective
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
    append!(plots, plot_quadrotor(robot, color=:blue, scale=0.3))
  end

  for trajectory in trajectories
    append!(plots, plot_states(trajectory, color=:blue, linestyle=":"))
  end

  for target in target_states
    append!(plots, plot_target(target))
  end

  append!(plots, visualize_filters(target_filters))

  plots
end

# Run a number of experiments with helpful features for batch processing
# * experiments run in parallel
# * individual results get cached
# * batch gets cached
# * provides helpful output as trials run
function run_experiments(tests;
                         trial_fun::Function,
                         print_summary::Function = x->nothing,
                         experiment_name::String,
                         data_folder::String,
                         reprocess=false,
                         threaded=true
                        )
  # produce file names
  data_file = string(data_folder, "/", experiment_name, ".jld2")
  trial_folder = string(data_folder, "/", experiment_name)
  get_trial_file = x->string(trial_folder, "/", experiment_name, " ", x, ".jld2")

  results = Dict{Any,Any}(key=>nothing for key in tests)

  if !isfile(data_file) || reprocess
    if !isdir(trial_folder)
      mkdir(trial_folder)
    end

    num_tests_completed = Atomic{Int64}(0)

    # First process any data that has been saved
    remaining_tests = filter(collect(tests)) do trial_spec
      run_test = true

      trial_file = get_trial_file(trial_spec)

      # Cache data from each trial
      if !reprocess && isfile(trial_file)
        try
          print("Loading: ", trial_file, "...")

          @load trial_file trial_results
          println("Loaded")

          results[trial_spec] = trial_results

          run_test = false
        catch e
          println("Failed to load ", trial_spec)
        end
      end

      run_test
    end

    println("\n", length(remaining_tests), " tests remain\n")

    # Mutex for load/save/output
    load_save_lock = ReentrantLock()

    start = time()
    shuffle!(remaining_tests)
    trial_backtrace = nothing

    spawn_for_each(remaining_tests, threaded=threaded) do trial_spec
      id = threadid()

      lock(load_save_lock)
      println("Thread-", id, " running: ", trial_spec)
      unlock(load_save_lock)

      trial_start = time()

      # Run trial
      trial_results = trial_fun(trial_spec)

      # Save results
      try
        lock(load_save_lock)
        print(id, "-Saving: ", trial_spec, "...")

        trial_file = get_trial_file(trial_spec)
        @save trial_file trial_results

        println(id, "-Saved")
      catch e
        # Save the error for later
        trial_backtrace = catch_backtrace()

        println(id, "-Failed to save ", trial_spec)
        return
      finally
        unlock(load_save_lock)
      end

      # Store value
      results[trial_spec] = trial_results

      # Summarize completion status
      elapsed = time() - start
      trial_elapsed = time() - trial_start

      # (returns old value)
      completed = atomic_add!(num_tests_completed, 1) + 1
      completion = completed/length(remaining_tests)
      projected = elapsed / completion
      hour = 3600

      lock(load_save_lock)
      print_summary(trial_spec)
      println(" (Done, ", completed, "/", length(remaining_tests),
              @sprintf(" %0.2f", 100completion), "%",
              " Trial time: ", @sprintf("%0.0fs", trial_elapsed),
              " Elapsed: ", @sprintf("%0.2fh", elapsed / hour),
              " Total: ", @sprintf("%0.2fh", projected / hour),
              " Remaining: ", @sprintf("%0.2fh", (projected - elapsed) / hour),
              ")"
             )
      unlock(load_save_lock)
    end

    if !isnothing(trial_backtrace)
      println("Loading or producing results produce errors. Backtrace:")
      map(println, stacktrace(trial_backtrace))
      println()

      error("One or more trials failed.")
    end

    @save data_file results
  else
    @load data_file results
  end

  results
end
