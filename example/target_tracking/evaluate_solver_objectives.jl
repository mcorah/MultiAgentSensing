# Analyze convergence of Monte Carlo tree search for various horizon lenghts

using SubmodularMaximization
using PyPlot
using Statistics
using Base.Threads
using Base.Iterators
using JLD2
using Printf
using Histograms

close("all")

trials = 1:10

experiment_name = "evaluate_solver_objectives"
data_folder = "./data"

reprocess = false

# Values taken from the entropy_by_solver script
num_robots = 16
trials = 1:10
solver_ind = 4 # dist-4

# Drop the first n and keep every few after
sample_inds = 20:5:100

library_solver_ind = 4 # dist-4

horizon = SubmodularMaximization.default_horizon

# We have to override the number of samples used to generate the library
num_mcts_samples = SubmodularMaximization.default_num_iterations[horizon]


solver_rounds = [2, 4, 8]
solvers = [solver_random,
           solve_myopic,
           map(num->prob->solve_n_partitions(num, prob), solver_rounds)...,
           solve_sequential]
solver_strings = ["myopic",
                  map(x->string("dist. ",x," rnds."), solver_rounds)...,
                  "sequential"]
solver_inds = 1:length(solvers)

#
# Load prior trials
#

println("Loading rollouts")
library_file = "./data/entropy_by_solver.jld2"

let
  @load library_file results
  global library_results = results
end

# Create a map from number of robots to a vector of configs sample pairs
samples = vcat(map(trials) do trial
                 rollout = library_results[library_solver_ind, num_robots, trial]

                 [(configs=rollout.configs, sample=sample)
                  for sample in rollout.data[sample_inds]]
               end...)


#
# Evaluate results on solvers
#

all_tests = product(solver_inds, sample_inds)

function trial_fun(x)
  solver_ind, sample_ind = x

  test = samples[sample_ind]

  sample = test.sample
  configs = MultiRobotTargetTrackingConfigs(test.configs;
                                            horizon = horizon,
                                            solver_iterations = num_mcts_samples)

  problem = MultiRobotTargetTrackingProblem(sample.robot_states,
                                            sample.target_filters,
                                            configs)
  solution = solvers[solver_ind](problem)

  solution.value
end
function print_summary(x)
  solver_ind, sample_ind = x

  println("Solver: ", solver_strings[solver_ind],
          " Sample: ", sample_ind)
end

@time results = run_experiments(all_tests,
                                trial_fun=trial_fun,
                                print_summary=print_summary,
                                experiment_name=experiment_name,
                                data_folder=data_folder,
                                reprocess=reprocess
                               )

println("Processing rollouts")
@time results = get_data()

println("Analyzing results")

results_matrix = map(x->results[x], all_tests)
boxplot(results_matrix', notch=false, vert=false)

title("Solver Objective Performance")
xlabel("Mutual Information Objective (bits)")
yticks(solver_inds, solver_strings)

save_fig("fig", "solver_objectives")
