# Test for weights with a small set of trials

using SubmodularMaximization
using PyPlot
using Base.Iterators
using Base.Threads
using JLD2
using Statistics
using RosDataProcess

close("all")

experiment_name = "large_scale_weights"
data_folder = "./data"

reprocess = false

num_robots = 8:8:100

trials = 1:5
steps = 100
trial_steps = 20:steps

# Range limits
communication_range = 20
robot_target_range_limit = 12
weights_range_limit = 20 # Use a larger range for weights for accurate results

solver = x->solve_communication_range_limit(x, num_partitions=4,
                                            communication_range=
                                            communication_range)

# Plotting configs

line_styles = ["-", "--"]
experiment_names = ["large-scale", "solver-trials"]

# Use colors from the default cycle
colors = get_cmap("tab10")
weights_color = colors(0)
objective_color = colors(1)

#
# Code to run experiments
#

# We will also load a separate set of results from the experiments
results_array = []

all_tests = product(num_robots, trials)
all_configurations = num_robots

function trial_fun(x)
  num_robots, trial = x

  num_targets = default_num_targets(num_robots)

  configs = MultiRobotTargetTrackingConfigs(num_robots,
                                            robot_target_range_limit=
                                              robot_target_range_limit
                                           )

  full_trial_data = target_tracking_experiment(steps=steps,
                                               num_robots=num_robots,
                                               configs=configs,
                                               solver=solver)

  weights = map(full_trial_data[trial_steps]) do step_data
        robot_states = step_data.robot_states
        target_filters = step_data.target_filters

        problem = MultiRobotTargetTrackingProblem(robot_states, target_filters,
                                                  configs)

        compute_weight_matrix(problem,
                              robot_target_range_limit=weights_range_limit
                             )
  end

  (trial_weights=weights,
   trial_data=full_trial_data[trial_steps],
   full_trial_data=full_trial_data,
   configs=configs)
end
function print_summary(x)
  (num_robots, trial) = x
  println("Num. Robots: ", num_robots,
          " Trial: ", trial)
end

let
  @time results = run_experiments(all_tests,
                                  trial_fun=trial_fun,
                                  print_summary=print_summary,
                                  experiment_name=experiment_name,
                                  data_folder=data_folder,
                                  reprocess=reprocess,
                                  threaded=true
                                 )
  push!(results_array, (results, num_robots, trials))
end

#
# *Optionally* load the weights dataset from the full set of trials if the data
# exists
#

experiment_weights_file = string(data_folder,
                                 "/weights_by_number_of_robots.jld2")

let
  if isfile(experiment_weights_file)
    @load experiment_weights_file results

    num_robots = sort(union([first(x) for x in keys(results)]))
    trials = sort(union([last(x) for x in keys(results)]))

    push!(results_array, (results, num_robots, trials))
  end
end

#
# Weights per num robots
#

figure()

for ii in 1:length(results_array)
  (results, num_robots, trials) = results_array[ii]
  style = line_styles[ii]
  experiment_name = experiment_names[ii]

  weight_per_robot = map(num_robots) do num_robots
    trial_weights = map(trials) do trial
      trial = results[num_robots, trial]

      trial_weights = trial.trial_weights

      total_weights = map(total_weight, trial_weights)

      total_weights / num_robots
    end

    vcat(trial_weights...)
  end

  weights_series = TimeSeries(num_robots, hcat(weight_per_robot...)')
  plot_trials(weights_series, mean=true, marker=".", markersize=25, linewidth=3,
              color=weights_color, linestyle=style,
              label=string("redundancy, ", experiment_name))
end

tick_params(axis=:y, labelcolor=weights_color)
ylabel("Redundancy per robot", color=weights_color)

weights_axis = gca()

#
# Plot objective values
#

twinx()

for ii in 1:length(results_array)
  (results, num_robots, trials) = results_array[ii]
  style = line_styles[ii]
  experiment_name = experiment_names[ii]

  objective_values = map(num_robots) do num_robots
    trial_objectives = map(trials) do trial
      trial = results[num_robots, trial]

      objectives = map(x->x.objective, trial.trial_data)

      objectives / num_robots
    end

    vcat(trial_objectives...)
  end

  objective_series = TimeSeries(num_robots, hcat(objective_values...)')
  plot_trials(objective_series, mean=true, marker=".", markersize=25,
              linewidth=3, color=objective_color, linestyle=style,
              label=string("objective, ", experiment_name))
end

tick_params(axis=:y, labelcolor=objective_color)
ylabel("Objective value per robot", color=objective_color)

objective_axis = gca()

lines1, labels1 = weights_axis.get_legend_handles_labels()
lines2, labels2 = objective_axis.get_legend_handles_labels()

legend(vcat(lines1, lines2), vcat(labels1, labels2), loc="lower right")

xlabel("Number of robots")
title("Weights and objective values per num. robots")

save_fig("fig", "large_scale_weights_per_num_robots")
