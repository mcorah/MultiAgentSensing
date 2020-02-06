using SubmodularMaximization
using PyPlot
using Base.Iterators
using Base.Threads
using JLD2
using Statistics
using Histograms
using RosDataProcess

close("all")

data_folder = "./data"

reprocess = false

# Configuration
targets_per_robot = 8 / 8
variance_scaling_factor = 0.5
grid_cells_per_robot = 10^2 / 8

experiment_name = string("minimal_weights_by_num_robots",
                         "_", targets_per_robot,
                         "_", variance_scaling_factor,
                         "_", grid_cells_per_robot)


num_robots = [4, 8, 12, 16, 20, 24, 28, 32]

steps = 20
trial_steps = 15:steps

# Note: we will run trials in threads so the solvers do not have to be threaded
solver = x->solve_n_partitions(2, x)

#
# Code to run experiments
#

all_tests = num_robots

function trial_fun(num_robots)
  grid_size = round(Int64, sqrt(grid_cells_per_robot * num_robots))
  num_targets = round(Int64, targets_per_robot * num_robots)

  sensor = RangingSensor(variance_scaling_factor=variance_scaling_factor)
  grid = Grid(grid_size, grid_size)

  configs = MultiRobotTargetTrackingConfigs(grid=grid, sensor=sensor)

  trial_data = target_tracking_experiment(steps=steps,
                                          num_robots=num_robots,
                                          num_targets=num_targets,
                                          configs=configs,
                                          solver=solver)

  weights = map(trial_data[trial_steps]) do step_data
        robot_states = step_data.robot_states
        target_filters = step_data.target_filters

        problem = MultiRobotTargetTrackingProblem(robot_states, target_filters,
                                                  configs)

        compute_weight_matrix(problem)
  end

  (trial_weights=weights, trial_data=trial_data, configs=configs)
end
function print_summary(num_robots)
  println("Num. robots: ", num_robots)
end

@time data = run_experiments(all_tests,
                             trial_fun=trial_fun,
                             print_summary=print_summary,
                             experiment_name=experiment_name,
                             data_folder=data_folder,
                             reprocess=reprocess
                            )

normalized_weights = map(num_robots) do num_robots
  trial = data[num_robots]

  total_weights = map(total_weight, trial.trial_weights)
  objectives = map(x->x.objective, trial.trial_data[trial_steps])

  total_weights ./ objectives
end

weights_series = TimeSeries(num_robots, hcat(normalized_weights...)')
plot_trials(weights_series, mean=true, marker=".", markersize=25, linewidth=3)

title_string = string("Tgts. per rbt.:", targets_per_robot,
                      " Var. scale:", variance_scaling_factor,
                      " Cells per rbt.:", grid_cells_per_robot)
title(title_string)
ylabel("Cost bound over obj.")
xlabel("Number of robots")

save_fig("fig", experiment_name)
