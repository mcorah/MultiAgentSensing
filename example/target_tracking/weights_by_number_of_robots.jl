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
using HistogramFilters

close("all")

experiment_name = "weights_by_number_of_robots"
data_folder = "./data"

reprocess = false

# Values taken from the entropy_by_solver script
num_robots = 8:8:40
trials = 1:20
solver_ind = 4 # dist-4

# Drop the first n and keep every few after
trial_steps = 20:100

all_tests = product(num_robots, trials)
all_configurations = num_robots

#
# Load and preprocess prior entropy results
#

solver_results_file = "./data/entropy_by_solver.jld2"

let
  @load solver_results_file results
  global solver_results = results
end

# Pull the relevant trials from the dataset
weights_trials = Dict(map(all_tests) do test
  test => solver_results[solver_ind, test...]
end)

#
# Compute weights
#

function trial_fun(x)
  num_robots, trial = x

  # The nomenclature here is weird, but we will only be using a part of
  # the trial and will implement that at computation time.
  data, configs = weights_trials[x...]
  trial_data = data[trial_steps]

  # Iterate through the trial and compute weights
  trial_weights = map(trial_data) do step_data
    robot_states = step_data.robot_states
    target_filters = step_data.target_filters

    problem = MultiRobotTargetTrackingProblem(robot_states, target_filters,
                                              configs)

    compute_weight_matrix(problem)
  end

  (trial_weights=trial_weights, trial_data=trial_data, configs=configs)
end
function print_summary(x)
  num_robots, trial = x

  println("Num. Robots: ", num_robots,
          " Trial: ", trial)
end

@time results = run_experiments(all_tests,
                                trial_fun=trial_fun,
                                print_summary=print_summary,
                                experiment_name=experiment_name,
                                data_folder=data_folder,
                                reprocess=reprocess
                               )

#
# Plot weights by objective values
#

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
ylabel("Redundancy over obj.")
xlabel("Number of robots")

save_fig("fig", "weights_by_number_of_robots")

#
# Weights per num robots
#

figure()

weight_per_robot = map(num_robots) do num_robots
  trial_weights = map(trials) do trial
    trial = results[num_robots, trial]

    trial_weights = trial.trial_weights

    total_weights = map(total_weight, trial_weights)

    total_weights / num_robots
  end

  vcat(trial_weights...)
end

boxplot(weight_per_robot, notch=false)


title("Weights per num. robots")
ylabel("Redundancy per robot")
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
ylabel("Objective value per robot")
xlabel("Number of robots")
xticks(1:length(num_robots), map(string, num_robots))

save_fig("fig", "objective_values_by_number_of_robots")
