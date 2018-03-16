using PyPlot
using SubmodularMaximization

num_trials = 100

num_agents = 50
num_sensors = 50
nominal_area = 2.0

sensor_radius = sqrt(nominal_area / (num_agents * pi))
station_radius = 2 * sensor_radius

agent_specification = CircleAgentSpecification(sensor_radius, station_radius,
                                         num_sensors)

f(x) = mean_area_coverage(x, 100)

solvers = Any[]

#push!(solvers, (solve_optimal, "Optimal"))
#push!(solvers, (solve_worst, "Worst-case"))
push!(solvers, (solve_random, "Random"))
push!(solvers, (solve_myopic, "Myopic"))

partitions = [2, 4, 8]
for num_partitions in partitions
  solve_n(p) = solve_n_partitions(num_partitions, p)
  push!(solvers, (solve_n, "Partition-$num_partitions"))
end

push!(solvers, (solve_sequential, "Sequential"))

results = zeros(num_trials, length(solvers))

problems = map(1:num_trials) do unused
  agents = generate_agents(agent_specification, num_agents)
  problem = PartitionProblem(f, agents)
end

for trial_num in 1:length(problems)
  println("Trial: $trial_num")

  for solver_num in 1:length(solvers)
    solver = solvers[solver_num][1]
    name = solvers[solver_num][2]

    solution = solver(problems[trial_num])

    results[trial_num, solver_num] = solution.value
  end
end

# now analyze weights
weight_matrices = map(problems, 1:length(problems)) do problem, ii
  println("Weight matrix $ii")
  compute_weight_matrix(problem)
end

total_weights = map(weight_matrices, 1:length(problems)) do weight_matrix, ii
  println("Total weights $ii")
  total_weight(weight_matrix)
end

edge_sets = map(weight_matrices, 1:length(problems)) do weight_matrix, ii
  println("Triangle $ii")
  extract_triangle(weight_matrix)
end


figure()
boxplot(results, notch=false, vert=false)
yticks(1:length(solvers), map(x->x[2], solvers))

# plot histograms of edge weights

figure()
PyPlot.plt[:hist](total_weights, 20)
title("Total Graph Weight Frequency")

figure()
PyPlot.plt[:hist](vcat(edge_sets...), 20)
title("Edge Weight Frequency")

# sequential solutions and edge weights
sequential_mean = mean(results[:,end][:])

partition_values = mean(results[:,3:end-1],1)[:]

mean_weight = mean(total_weights)
println("Mean weight: $mean_weight")
bounds = map(x->mean_weight/x, partitions)

#figure()
#plot([partitions[1], partitions[end]], [sequential_mean, sequential_mean])
#plot(partitions, partition_values)
#plot(partitions, partition_values + bounds)
#legend(["Sequential", "Partition-n"])
