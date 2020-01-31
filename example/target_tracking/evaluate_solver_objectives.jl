# Analyze convergence of Monte Carlo tree search for various horizon lenghts

using SubmodularMaximization
using PyPlot
using Statistics
using Base.Threads
using Base.Iterators
using JLD2
using Printf

close("all")

println("Loading rollout library")
library_file = "./data/rollout_library.jld2"
@time @load library_file data num_robots trials
rollouts = data

num_robots = 10

# Create a map from number of robots to a vector of configs sample pairs
samples = vcat(map(trials) do trial
                 rollout = rollouts[num_robots, trial]

                 [(configs=rollout.configs, sample=sample)
                  for sample in rollout.samples]
               end...)

# Everything has the same number of samples
sample_inds = 1:length(samples)

data_file = "./data/evaluate_solver_objectives.jld2"
reprocess = false

horizon = 4
solver_rounds = [2, 4, 8]
solvers = [solve_myopic,
           map(num->prob->solve_n_partitions(num, prob), solver_rounds)...,
           solve_sequential]
solver_strings = ["myopic",
                  map(x->string("dist. ",x," rnds."), solver_rounds)...,
                  "sequential"]
solver_inds = 1:length(solvers)

all_tests = product(solver_inds, sample_inds)
all_configurations = product(solver_inds)

# We have to override the number of samples used to generate the library
num_mcts_samples = SubmodularMaximization.default_num_iterations[horizon]

function get_data()
  results = Array{Float64}(undef, size(all_tests))

  num_tests_completed = Atomic{Int64}(0)

  if !isfile(data_file) || reprocess

    start = time()
    iters = collect(all_tests)
    @threads for (solver_ind, sample_ind) in iters
      trial_start = time()

      test = samples[sample_ind]

      sample = test.sample

      configs = MultiRobotTargetTrackingConfigs(test.configs;
                                          horizon = horizon,
                                          solver_iterations = num_mcts_samples)
      problem = MultiRobotTargetTrackingProblem(sample.robot_states,
                                                sample.target_filters,
                                                configs)
      solution = solvers[solver_ind](problem)
      reward = solution.value

      results[solver_ind, sample_ind] = reward

      elapsed = time() - start
      trial_elapsed = time() - trial_start

      # (returns old value)
      completed = atomic_add!(num_tests_completed, 1) + 1
      completion = completed/length(all_tests)
      projected = elapsed / completion
      hour = 3600

      println("Solver: ", solver_strings[solver_ind],
              " Trial: ", sample_ind)
      println(" (Done, ", completed, "/", length(all_tests),
              @sprintf(" %0.2f", 100completed/length(all_tests)), "%",
              " Trial time: ", @sprintf("%0.0fs", trial_elapsed),
              " Elapsed: ", @sprintf("%0.0fs", elapsed),
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

println("Processing rollouts")
@time results = get_data()

println("Analyzing results")

boxplot(results', notch=false, vert=false)
title("Solver Objective Performance")
xlabel("Mutual Information Objective (bits)")
yticks(solver_inds, solver_strings)

save_fig("fig", "solver_objectives")
