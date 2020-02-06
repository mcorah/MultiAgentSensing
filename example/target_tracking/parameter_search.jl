using SubmodularMaximization
using PyPlot
using Base.Iterators
using Base.Threads
using JLD2
using Statistics
using Histograms

close("all")

experiment_name = "parameter_search"
data_folder = "./data"

reprocess = false

num_robots = 8

trials = 1:10

horizon = SubmodularMaximization.default_horizon

steps = 50
trial_steps = 20:steps

# Main parameters for variation
num_targets = [4, 6, 8]
variance_scaling_factors = [0.1, 0.5, 2.0, 3.0]
grid_sizes = [10, 12, 14, 16]

# Note: we will run trials in threads so the solvers do not have to be threaded
solvers = [solve_sequential, solve_myopic]
solver_strings = map(string, solvers)
solver_inds = 1:length(solvers)

#
# Code to run experiments
#

tests = product(solver_inds, num_targets, variance_scaling_factors, grid_sizes,
                trials)

function trial_fun(x)
  (solver_ind, num_targets, variance_scaling_factor, grid_size, trial) = x

  grid = Grid(grid_size, grid_size)
  sensor = RangingSensor(variance_scaling_factor=variance_scaling_factor)

  configs = MultiRobotTargetTrackingConfigs(horizon=horizon,
                                            grid=grid,
                                            sensor=sensor
                                           )

  trial_data = target_tracking_experiment(steps=steps,
                                          num_robots=num_robots,
                                          num_targets=num_targets,
                                          configs=configs,
                                          solver=solvers[solver_ind]
                                         )

  (data=trial_data, configs=configs)
end
function print_summary(x)
  (solver_ind, num_targets, variance_scaling_factor, grid_size, trial) = x

  println("Solver: ", solver_strings[solver_ind],
          " Num. targets: ", num_targets,
          " Var scale: ", variance_scaling_factor,
          " Grid size: ", grid_size,
          " Trial: ", trial
         )
end

@time data = run_experiments(tests,
                             trial_fun=trial_fun,
                             print_summary=print_summary,
                             experiment_name=experiment_name,
                             data_folder=data_folder,
                             reprocess=reprocess
                            )

#
# Preprocess data
#

entropies = Dict(key=>map(x->entropy(x.target_filters), value.data)
                 for (key, value) in data)

concatenated_entropy = map(product(solver_inds, num_targets,
                                   variance_scaling_factors, grid_sizes)) do x
  vcat(map(trials) do trial
    entropies[x..., trial][trial_steps]
  end...)
end[:]

solver_abbreviations = Dict(solve_myopic=>"myp.", solve_sequential=>"seq.")
titles = map(product(solvers, num_targets, variance_scaling_factors,
                     grid_sizes)) do (solver, num_targets,
                                      variance_scaling_factor, grid_size)

  solver_string = solver_abbreviations[solver]

  string(solver_string,
         " tgts(", num_targets,
         ") var-scal(", variance_scaling_factor,
         ") grid-len(", grid_size, ")")
end[:]

trials_per_figure = 20

#
# Produce bar plots of results
#

# Creates an "array" pairs of entropy/title-blocks
figure_partition = zip(partition.((concatenated_entropy, titles),
                                  trials_per_figure)...)

for (ii, (concatenated_entropy, titles)) in enumerate(figure_partition)
  figure()

  boxplot(concatenated_entropy, notch=false, vert=false)

  title_string = string("Entropy distribution (", ii, " of ",
                        length(figure_partition), ")")
  title(title_string)
  xlabel("Entropy (bits)")
  yticks(1:length(titles), titles)

  save_fig("fig", string("parameter_search_", ii))
end

#
# Summarize results
#

println("Trial means")
means = map(mean, concatenated_entropy)
for (title, mean) in zip(titles, means)
  println(title, " Mean: ", mean)
end


println()

println("Sequential entropy is so much lower (better) than myopic:")
println("(normalized by the pairwise maximum)")

sequential_results = means[1:2:end]
myopic_results   = means[2:2:end]
gaps = (myopic_results .- sequential_results) ./ max.(sequential_results,
                                                      myopic_results)

param_gaps = zip(product(num_targets, variance_scaling_factors,
                         grid_sizes), gaps)
param_gaps = sort(collect(param_gaps), by=last)

for (params, gap) in param_gaps
  num_targets, variance_scaling_factor, grid_size = params

  println("tgts(", num_targets,
         ") var-scal(", variance_scaling_factor,
         ") grid-len(", grid_size,
         ") Gap: ", gap)
end
