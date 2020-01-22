using SubmodularMaximization
using POMDPs
using MCTS
using PyPlot
using Base.Iterators
using Base.Threads
using JLD2
using Statistics

close()

data_file = "./data/horizon_entropy_data.jld2"
reprocess = false

steps = 100
num_robots = 4

trials = 1:20

horizons = 1:6

# We will drop a fraction of each trial so that the filters have time to
# converge to steady states
trial_steps = 20:steps


# Note: we will run trials in threads so the solvers do not have to be threaded
solvers = [solve_sequential, solve_myopic]
solver_strings = map(string, solvers)

function get_data()
  data_mutex = Mutex()
  data = Dict()

  if !isfile(data_file) || reprocess
    tests = product(solvers, horizons, trials)
    test_partition = partition(tests, nthreads())

    for (ii, block) in enumerate(test_partition)
      println("Block num: ", ii, " of ", length(test_partition))

      @time @threads for (solver, horizon, trial) in block

        configs = MultiRobotTargetTrackingConfigs(num_robots,
                                                  horizon=horizon,
                                                 )

        trial_data = target_tracking_experiment(steps=steps,
                                                num_robots=num_robots,
                                                configs=configs,
                                                solver=solver
                                               )

        lock(data_mutex)
        data[string(solver), horizon, trial] =
          (data=trial_data, configs=configs)
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

concatenated_entropy = map(product(solver_strings, horizons)) do (solver,
                                                                  horizon)
  vcat(map(trials) do trial
    entropies[solver, horizon, trial][trial_steps]
  end...)
end[:]
titles = map(product(solver_strings, horizons)) do (solver, horizon)
  string(solver, " ", horizon)
end[:]

boxplot(concatenated_entropy, notch=false, vert=false)
title("Entropy distribution")
xlabel("Entropy (bits)")
yticks(1:length(titles), titles)

save_fig("fig", "entropy_by_solver_horizon")

means = map(mean, concatenated_entropy)
for (title, mean) in zip(titles, means)
  println(title, "(mean): ", mean)
end

println()

println("Sequential entropy is so much lower (better) than myopic:")
gaps =  means[2:2:end] - means[1:2:end]
for (horizon, gap) in zip(horizons, gaps)
  println("Horizon: ", horizon, " Gap: ", gap)
end
