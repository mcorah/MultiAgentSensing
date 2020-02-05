# Loads trials and computes the weights for the given number of robots

using SubmodularMaximization
using PyPlot
using Base.Iterators
using Base.Threads
using JLD2
using Statistics
using Printf
using Random
using RosDataProcess

close("all")

experiment_name = "weights_by_number_of_robots"
data_file = string("./data/", experiment_name, ".jld2")
cache_folder = string("./data/", experiment_name, "/")

reprocess = false



# Values taken from the entropy_by_solver script
num_robots = [4, 8, 12, 16]
trials = 1:10
solver_ind = 6 # sequential
trial_steps = 20:100

# load and preprocess the entropy results

solver_results_file = "./data/entropy_by_solver.jld2"

let
  @load solver_results_file results
  global solver_results = results
end

all_tests = product(num_robots, trials)
all_configurations = num_robots

# Pull the relevant trials from the dataset
weights_trials = Dict(map(all_tests) do test
  test => solver_results[solver_ind, test...]
end)

function get_data()
  results = Dict{Any,Any}(key=>nothing for key in all_tests)

  if !isfile(data_file) || reprocess
    if !isdir(cache_folder)
      mkdir(cache_folder)
    end

    num_tests_completed = Atomic{Int64}(0)

    # First process any data that has been saved
    remaining_tests = filter(collect(all_tests)) do trial_spec
      run_test = true

      trial_file = string(cache_folder, experiment_name, " ", trial_spec, ".jld2")

      # Cache data from each trial
      if !reprocess && isfile(trial_file)
        try
          print(threadid(), "-Loading: ", trial_file, "...")

          @load trial_file trial_weights trial_data configs
          println(threadid(), "-Loaded")

          results[trial_spec] = (
                                 trial_weights=trial_weights,
                                 trial_data=trial_data,
                                 configs=configs
                                )
          run_test = false
        catch e
          println(threadid(), "-Failed to load ", trial_spec)
        end
      end

      run_test
    end

    println("\n", length(remaining_tests), " tests remain\n")

    # Mutex for load/save/output
    load_save_lock = SpinLock()

    start = time()
    shuffle!(remaining_tests)
    trial_error = nothing
    @threads for trial_spec in remaining_tests
      println("Thread-", threadid(), " running: ", trial_spec)

      trial_start = time()

      num_robots, trial = trial_spec

      trial_file = string(cache_folder, experiment_name, " ", trial_spec, ".jld2")

      # The nomenclature here is weird, but we will only be using a part of
      # the trial and will implement that at computation time.
      data, configs = weights_trials[trial_spec...]
      trial_data = data[trial_steps]

      # Iterate through the trial and compute weights
      trial_weights = map(trial_data) do trial_step
        robot_states = trial_step.robot_states
        target_filters = trial_step.target_filters

        problem = MultiRobotTargetTrackingProblem(robot_states, target_filters,
                                                  configs)

        compute_weight_matrix(problem)
      end

      try
        lock(load_save_lock)
        print(threadid(), "-Saving: ", trial_spec, "...")

        @save trial_file trial_weights trial_data configs
      catch e
        # Save the error for later
        trial_error = e

        println(threadid(), "-Failed to save ", trial_spec)
        continue
      finally
        println(threadid(), "-Saved")
        unlock(load_save_lock)
      end

      elapsed = time() - start
      trial_elapsed = time() - trial_start

      results[trial_spec] = (
                             trial_weights=trial_weights,
                             trial_data=trial_data,
                             configs=configs
                            )

      # (returns old value)
      completed = atomic_add!(num_tests_completed, 1) + 1
      completion = completed/length(remaining_tests)
      projected = elapsed / completion
      hour = 3600

      lock(load_save_lock)
      println("Num. Robots: ", num_robots,
              " Trial: ", trial)
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

    if !isnothing(trial_error)
      println("Loading or producing results produce errors.")
      error("One or more trials failed.")
    end

    @save data_file results
  else
    @load data_file results
  end

  results
end

@time results = get_data()

normalized_weights = map(num_robots) do num_robots
  trial_weights = map(trials) do trial
    trial = results[num_robots, trial]

    trial_weights = trial.trial_weights
    objectives = map(x->x.objective, trial.trial_data)

    total_weights = map(total_weight, trial_weights)

    total_weights ./ objectives
  end

  vcat(trial_weights...)
end

#boxplot([normalized_weights[n] for n in num_robots],
        #notch=false)

weights_series = TimeSeries(num_robots, hcat(normalized_weights...)')

plot_trials(weights_series, mean=true, marker=".", markersize=25, linewidth=3)


title("Target weights")
ylabel("Cost bound over obj.")
xlabel("Number of robots")
#xticks(1:length(num_robots), map(string, num_robots))

save_fig("fig", "weights_by_number_of_robots")

#
# Weights per num robots
#

figure()

normalized_weights = map(num_robots) do num_robots
  trial_weights = map(trials) do trial
    trial = results[num_robots, trial]

    trial_weights = trial.trial_weights

    total_weights = map(total_weight, trial_weights)

    total_weights / num_robots
  end

  vcat(trial_weights...)
end

boxplot(normalized_weights,
        notch=false)


title("Weights per num. robots")
ylabel("Cost bound per num. robots")
xlabel("Number of robots")
xticks(1:length(num_robots), map(string, num_robots))

save_fig("fig", "weights_per_num_robots")

#
# Plot objective values
#

figure()

objective_values = Dict(map(num_robots) do num_robots
  trial_objectives = map(trials) do trial
    trial = results[num_robots, trial]

    objectives = map(x->x.objective, trial.trial_data)

    objectives / num_robots
  end

  num_robots => vcat(trial_objectives...)
end)

boxplot([objective_values[n] for n in num_robots],
        notch=false)

title("Objective values")
ylabel("Cost bound over obj.")
xlabel("Number of robots")
xticks(1:length(num_robots), map(string, num_robots))

save_fig("fig", "objective_values_by_number_of_robots")
