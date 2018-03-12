using PyPlot
using SubmodularMaximization

num_trials = 1000

num_agents = 20
num_sensors = 3
nominal_area = 1.0

sensor_radius = sqrt(nominal_area / (num_agents * pi))
station_radius = 2 * sensor_radius

agent_specification = AgentSpecification(sensor_radius, station_radius,
                                         num_sensors)

f(x) = mean_area_coverage(x, 100)

solvers = Any[]

#push!(solvers, (solve_optimal, "Optimal"))
#push!(solvers, (solve_worst, "Worst-case"))
push!(solvers, (solve_random, "Random"))
push!(solvers, (solve_myopic, "Myopic"))

for num_partitions in [2, 4, 6]
  solve_n(p) = solve_n_partitions(num_partitions, p)
  push!(solvers, (solve_n, "Partition-$num_partitions"))
end

push!(solvers, (solve_sequential, "Sequential"))

results = zeros(num_trials, length(solvers))

for trial_num in 1:num_trials
  println("Trial: $trial_num")

  agents = generate_coverage_agents(agent_specification, num_agents)
  problem = PartitionProblem(f, agents)

  for solver_num in 1:length(solvers)
    solver = solvers[solver_num][1]
    name = solvers[solver_num][2]

    solution = solver(problem)

    results[trial_num, solver_num] = solution.value
  end
end

mean_value = mean(results, 2)

figure()
boxplot(results, notch=false, vert=false)
yticks(1:length(solvers), map(x->x[2], solvers))
