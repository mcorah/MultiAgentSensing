using PyPlot
using SubmodularMaximization
using HDF5, JLD

pygui(false)
name = "compare_adaptive_solvers"
fig_path = "./fig/$name"
data_path = "./data/$name"
mkpath(fig_path)
mkpath(data_path)

########
# Params
########
num_trials = 100

num_events = 50
num_agents = 50
num_sensors = 10
nominal_area = 0.6

max_success_probability = 1.0
sensor_radius = sqrt(nominal_area / (num_agents * pi))
station_radius = 4*sensor_radius

acceptable_suboptimality = 0.4
suboptimality_per_agent = acceptable_suboptimality / num_agents
communication_range = 2 * station_radius

agent_specification = ProbabilisticAgentSpecification(max_success_probability,
                                                      sensor_radius,
                                                      station_radius,
                                                      num_sensors)

###################
# Generate problems
###################

# each problem has a different set of agents and events
problems = map(1:num_trials) do unused
  agents = generate_agents(agent_specification, num_agents)

  events = generate_events(num_events)
  f(x) = mean_detection_probability(x, events)

  problem = PartitionProblem(f, agents)
end
@save "$data_path/partition_matroids" map(x->x.partition_matroid, problems)

############################################
# Do intermediate processing of each problem
############################################

# now analyze weights
println("Generating weight matrices")
weight_matrices = map(problems, 1:length(problems)) do problem, ii
  println("Weight matrix $ii")
  compute_weight_matrix(problem)
end
@save "$data_path/weights" weight_matrices

println("Generating total weights")
total_weights = map(weight_matrices, 1:length(problems)) do weight_matrix, ii
  println("Total weights $ii")
  total_weight(weight_matrix)
end

println("Generating edge sets")
edge_sets = map(weight_matrices, 1:length(problems)) do weight_matrix, ii
  println("Triangle $ii")
  extract_triangle(weight_matrix)
end

println("Generating local partition sizes")
local_partition_sizes = map(problems) do problem
  compute_local_num_partitions(suboptimality_per_agent, problem)
end
@save "$data_path/local_partition_sizes" local_partition_sizes

println("Generating global partition sizes")
global_partition_sizes = map(problems) do problem
  compute_global_num_partitions(suboptimality_per_agent, problem)
end
@save "$data_path/global_partition_sizes" global_partition_sizes

###########################
# Generate adaptive solvers
###########################

dag_solver_array = []

println("Generating global solvers")
global_adaptive_solvers = map(1:length(problems)) do ii
  generate_by_global_partition_size(num_agents, global_partition_sizes[ii])
end
push!(dag_solver_array, global_adaptive_solvers)

println("Generating global range solvers")
global_range_solvers = map(1:length(problems)) do ii
  RangeSolver(problems[ii], global_adaptive_solvers[ii], communication_range)
end
push!(dag_solver_array, global_range_solvers)

println("Generating local solvers")
local_adaptive_solvers = map(1:length(problems)) do ii
  generate_by_local_partition_size(local_partition_sizes[ii])
end
push!(dag_solver_array, local_adaptive_solvers)

println("Generating local range solvers")
local_range_solvers = map(1:length(problems)) do ii
  RangeSolver(problems[ii], local_adaptive_solvers[ii], communication_range)
end
push!(dag_solver_array, local_range_solvers)

###################################
# Evaluate weights of deleted edges
###################################
function deleted_weights(solvers)
  println("...evaluating solver")
  mean(map(1:length(problems)) do ii
    deleted_edge_weight(solvers[ii], weight_matrices[ii])
  end)
end

println("Evaluating deleted weights")
mean_deleted = map(deleted_weights, dag_solver_array)

################
# Obtain results
################

function evaluate_adaptive_solver(solver_instances, title)
  values = map(1:length(problems)) do ii
    solve_dag(solver_instances[ii], problems[ii]).value
  end

  println("Evaluating $title")
  push!(titles, title)
  push!(results, values)
end

titles = String[]
results = Array[]

println("Evaluating Myopic")
push!(titles, "Myopic")
push!(results, map(x->solve_myopic(x).value, problems))

println("Evaluating Random")
push!(titles, "Random")
push!(results, map(x->solve_random(x).value, problems))

evaluate_adaptive_solver(global_adaptive_solvers, "Global Adaptive")
evaluate_adaptive_solver(global_range_solvers, "Global Range")
evaluate_adaptive_solver(local_adaptive_solvers, "Local Adaptive")
evaluate_adaptive_solver(local_range_solvers, "Local Range")

println("Evaluating Sequential")
push!(titles, "Sequential")
push!(results, map(x->solve_sequential(x).value, problems))

results = hcat(results...)
@save "$data_path/results" results

################
# Generate plots
################

function compute_range_frequency(x::Array{Int64, 1})
  range = collect(minimum(x):maximum(x))

  frequencies = zeros(Int64, length(range))

  for value in x
    frequencies[value - range[1] + 1] += 1
  end

  (range, frequencies)
end

function frequency_bar(x::Array{Int64, 1})
  r, f = compute_range_frequency(x)
  bar(r, f)
end

figure()
boxplot(results, notch=false, vert=false)
yticks(1:length(titles), titles)
xlabel("Expect. num. ident. events")
tight_layout()

save_fig(fig_path, "results")

# plot histograms of edge weights
figure()
PyPlot.plt[:hist](total_weights, 20)
tt = "Total Graph Weight Frequency"

save_fig(fig_path, tt)
title(tt)

figure()
PyPlot.plt[:hist](vcat(edge_sets...), 20)
tt = "Edge Weight Frequency"
save_fig(fig_path, tt)
title(tt)

# plot histograms of partition sizes
figure()

frequency_bar(global_partition_sizes)
tt = "Global Partition Size Frequency"
ylabel("Freq.")
xlabel("Partition size, \$n_d\$")
tight_layout()

save_fig(fig_path, tt)
title(tt)

figure()
frequency_bar(vcat(local_partition_sizes...))
tt = "Local Partition Size Frequency"
ylabel("Freq.")
xlabel("Local partition size, \$k_i\$")
tight_layout()

save_fig(fig_path, tt)
title(tt)

# plot deleted weights
bar_order = [1, 2, 3, 4]
figure()
barh(1:4, mean_deleted[bar_order])
yticks(1:4, titles[2:5][bar_order])
tt = "Deleted Edge Weights"
xlabel("Cumulative deleted edge weight")
tight_layout()

save_fig(fig_path, tt)
title(tt)
