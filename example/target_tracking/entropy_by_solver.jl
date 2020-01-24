using SubmodularMaximization
using PyPlot
using Base.Iterators
using Base.Threads
using JLD2
using Statistics
using Printf

close("all")

data_file = "./data/entropy_by_solver.jld2"
reprocess = true

steps = 100
num_robots = [5, 10, 15, 20]

trials = 1:20

# We will drop a fraction of each trial so that the filters have time to
# converge to steady states
trial_steps = 20:steps

# Note: we will run trials in threads so the solvers do not have to be threaded
horizon = 4
solver_rounds = [2, 4, 8]
solvers = [solve_random,
           solve_myopic,
           map(num->prob->solve_n_partitions(num, prob), solver_rounds)...,
           solve_sequential]
solver_strings = ["random",
                  "myopic",
                  map(x->string("dist. ",x," rnds."), solver_rounds)...,
                  "sequential"]
solver_inds = 1:length(solvers)


all_tests = product(num_robots, solver_inds, trials)
all_configurations = product(num_robots, solver_inds)

function get_data()
  results = Dict{Any,Any}(key=>nothing for key in all_tests)


  if !isfile(data_file) || reprocess
    num_tests_completed = Atomic{Int64}(0)

    start = time()
    iters = collect(all_tests)
    @threads for trial_spec in iters
      (num_robots, solver_ind, trial) = trial_spec

      configs = MultiRobotTargetTrackingConfigs(num_robots,
                                                horizon=horizon,
                                               )

      trial_data = target_tracking_experiment(steps=steps,
                                              num_robots=num_robots,
                                              configs=configs,
                                              solver=solvers[solver_ind]
                                             )

      results[trial_spec] = (data=trial_data, configs=configs)

      elapsed = time() - start

      completed = atomic_add!(num_tests_completed, 1) + 1
      completion = completed/length(all_tests)
      projected = elapsed / completion
      hour = 3600

      println("Num. Robots: ", num_robots,
              " Solver: ", solver_strings[solver_ind],
              " Trial: ", trial)
      println(" (Done, ", completed, "/", length(all_tests),
              @sprintf(" %0.2f", 100completion), "%",
              " Elapsed: ", @sprintf(" %0.0fs", elapsed),
              " Total: ", @sprintf(" %0.2fh", projected / hour),
              " Remaining: ", @sprintf(" %0.2fh", (projected - elapsed) / hour),
              ")"
             )
    end

    @save data_file results
  else
    @load data_file results
  end

  results
end

@time results = get_data()

# Normalize by number of targets
entropies = Dict(map(all_tests) do key
                   key => map(results[key].data) do x
                     entropy(x.target_filters) / length(target_filters)
                   end
                 end)

# Concatenate all data for a set of trials
concatenated_entropy = Dict(map(all_configurations) do
  trial_data = map(trials) do trial
    entropies[configuration..., trial][trial_steps]
  end

  configuration => vcat(trial_data...)
end)

titles = map(all_configurations) do (num_robots, solver_ind)
  string(num_robots, "-robots ", solver_strings)
end[:]

boxplot(concatenated_entropy, notch=false, vert=false)
title("Entropy distribution")
xlabel("Entropy per target (bits)")
yticks(1:length(titles), titles)

save_fig("fig", "entropy_by_solver")
