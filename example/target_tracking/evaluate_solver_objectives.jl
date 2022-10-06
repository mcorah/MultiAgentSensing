# Analyze convergence of Monte Carlo tree search for various horizon lenghts

using SubmodularMaximization
using PyPlot
using Statistics
using Base.Threads
using Base.Iterators
using JLD2
using Printf

close("all")

experiment_name = "evaluate_solver_objectives"
data_folder = "./data"

reprocess = false

# Values taken from the entropy_by_solver script
num_robots = 16
trials = 1:20
solver_ind = 4 # dist-4

# Drop the first n and keep every few after
sample_inds = 20:100

library_solver_ind = 4 # dist-4

# Range limits
communication_range = 20
robot_target_range_limit = 12
range_limit_partitions = 4
range_limit_solver(x) = solve_communication_range_limit(x,
                                                        num_partitions=
                                                        range_limit_partitions,
                                                        communication_range=
                                                        communication_range)

solver_rounds = [2, 4, 8]
solvers = [(solve_random, Inf),
           (solve_myopic, Inf),
           map(num->(prob->solve_n_partitions(num, prob), Inf), solver_rounds)...,
           (range_limit_solver, robot_target_range_limit),
           (solve_sequential, Inf)]
solver_strings = ["random",
                  "myopic",
                  map(x->string("dist. ",x," rnds."), solver_rounds)...,
                  string("rng. lim. ", range_limit_partitions, " rnds."),
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

all_tests = product(solver_inds, 1:length(samples))

function trial_fun(x)
  solver_ind, sample_ind = x
  solver, robot_target_range_limit = solvers[solver_ind]

  test = samples[sample_ind]


  sample = test.sample
  configs = MultiRobotTargetTrackingConfigs(test.configs,
                                            robot_target_range_limit=
                                            robot_target_range_limit
                                           )

  problem = MultiRobotTargetTrackingProblem(sample.robot_states,
                                            sample.target_filters,
                                            configs)
  solution = solver(problem)

  solution.value
end
function print_summary(x)
  solver_ind, sample_ind = x
  _, robot_target_range_limit = solvers[solver_ind]

  println("Solver: ", solver_strings[solver_ind],
          " Robot-Target Range Limit: ", robot_target_range_limit,
          " Sample: ", sample_ind)
end

@time results = run_experiments(all_tests,
                                trial_fun=trial_fun,
                                print_summary=print_summary,
                                experiment_name=experiment_name,
                                data_folder=data_folder,
                                reprocess=reprocess
                               )


println("Analyzing results")

#
# Results by objective values
#

results_matrix = map(x->results[x], all_tests)
boxplot(results_matrix', notch=false, vert=false)

title("Solver Objective Performance")
xlabel("Mutual Information Objective (bits)")
yticks(solver_inds, solver_strings)

save_fig("fig", "solver_objectives")

#
# Results normalized by maximum objective values on each problem
#

# Max values over different solvers for each problem instance
figure()

maximum_values = maximum(results_matrix, dims=1)
normalized_results = results_matrix ./ maximum_values

boxplot(normalized_results', notch=false, vert=false)

title("Normalized Objective Performance")
xlabel("Objective (fraction of max)")
yticks(solver_inds, solver_strings)

save_fig("fig", "normalized_solver_objectives")
