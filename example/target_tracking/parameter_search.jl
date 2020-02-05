using SubmodularMaximization
using PyPlot
using Base.Iterators
using Base.Threads
using JLD2
using Statistics

close("all")

data_file = "./data/parameter_search.jld2"
reprocess = false

num_robots = 4

trials = 1:10

horizon = SubmodularMaximization.default_horizon

steps = 60
trial_steps = 20:steps

# Main parameters for variation
num_targets = [1, 2, 3, 4, 5]
variance_scaling_factors = [0.1, 0.2, 0.3, 0.5]
grid_sizes = [10, 14, 20, 25]

# Note: we will run trials in threads so the solvers do not have to be threaded
solvers = [solve_sequential, solve_myopic]
solver_strings = map(string, solvers)

function get_data()
  data_mutex = Mutex()
  data = Dict()

  if !isfile(data_file) || reprocess
    tests = product(solvers, num_targets, variance_scaling_factors, grid_sizes,
                    trials)

    test_partition = partition(tests, nthreads())

    for (ii, block) in enumerate(test_partition)
      println("Block num: ", ii, " of ", length(test_partition))

      @time @threads for (solver, num_targets, variance_scaling_factor,
                          grid_size, trial) in block

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
                                                solver=solver
                                               )

        lock(data_mutex)
        data[string(solver), num_targets, variance_scaling_factor, grid_size,
             trial] = (data=trial_data, configs=configs)
        unlock(data_mutex)
      end
    end

    @save data_file data
  else
    @load data_file data
  end

  data
end

@time data = get_data()

entropies = Dict(key=>map(x->entropy(x.target_filters), value.data)
                 for (key, value) in data)

concatenated_entropy = map(product(solver_strings, num_targets,
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
