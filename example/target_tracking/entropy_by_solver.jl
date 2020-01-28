using SubmodularMaximization
using PyPlot
using Base.Iterators
using Base.Threads
using JLD2
using Statistics
using Printf
using Random

close("all")

trial_name = "entropy_by_solver"
data_file = string("./data/", trial_name, ".jld2")
cache_folder = string("./data/", trial_name, "/")

reprocess = false

steps = 100
num_robots = [4, 8, 12, 16]

trials = 1:10

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


all_tests = product(solver_inds, num_robots, trials)
all_configurations = product(solver_inds, num_robots)

function get_data()
  results = Dict{Any,Any}(key=>nothing for key in all_tests)


  if !isfile(data_file) || reprocess
    if !isdir(cache_folder)
      mkdir(cache_folder)
    end

    num_tests_completed = Atomic{Int64}(0)

    start = time()
    iters = shuffle!(collect(all_tests))
    @threads for trial_spec in iters
      trial_start = time()

      local trial_data, configs
      trial_file = string(cache_folder, trial_name, " ", trial_spec, ".jld2")

      # Cache data from each trial
      if !isfile(trial_file) || reprocess
        (solver_ind, num_robots, trial) = trial_spec

        configs = MultiRobotTargetTrackingConfigs(num_robots,
                                                  horizon=horizon,
                                                 )

        trial_data = target_tracking_experiment(steps=steps,
                                                num_robots=num_robots,
                                                configs=configs,
                                                solver=solvers[solver_ind]
                                               )



        @save trial_file trial_data configs
      else
        pritln("Loading: ", trial_file)
        @load trial_file trial_data configs
      end

      elapsed = time() - start
      trial_elapsed = time() - trial_start

      results[trial_spec] = (data=trial_data, configs=configs)

      # (returns old value)
      completed = atomic_add!(num_tests_completed, 1) + 1
      completion = completed/length(all_tests)
      projected = elapsed / completion
      hour = 3600

      println("Num. Robots: ", num_robots,
              " Solver: ", solver_strings[solver_ind],
              " Trial: ", trial)
      println(" (Done, ", completed, "/", length(all_tests),
              @sprintf(" %0.2f", 100completion), "%",
              " Trial time: ", @sprintf("%0.0fs", trial_elapsed),
              " Elapsed: ", @sprintf("%0.2fh", elapsed / hour),
              " Total: ", @sprintf("%0.2fh", projected / hour),
              " Remaining: ", @sprintf("%0.2fh", (projected - elapsed) / hour),
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
                     entropy(x.target_filters) / length(x.target_filters)
                   end
                 end)

# Concatenate all data for a set of trials
concatenated_entropy = map(all_configurations) do configuration
  trial_data = map(trials) do trial
    entropies[configuration..., trial][trial_steps]
  end

  vcat(trial_data...)
end[:]

titles = map(all_configurations) do (solver_ind, num_robots)
  string(num_robots, "-robots ", solver_strings[solver_ind])
end[:]

boxplot(concatenated_entropy, notch=false, vert=false)
title("Entropy distribution")
xlabel("Entropy per target (bits)")
yticks(1:length(titles), titles)

save_fig("fig", "entropy_by_solver")
