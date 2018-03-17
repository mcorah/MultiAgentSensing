using PyPlot
using SubmodularMaximization

pygui(false)
fig_path = "./fig/probabilistic_coverage_test"
mkpath(fig_path)

########
# Params
########

num_events = 50
num_agents = 50
num_sensors = 10
nominal_area = 0.6

max_success_probability = 1.0
sensor_radius = sqrt(nominal_area / (num_agents * pi))
station_radius = 4*sensor_radius

#############
# environment
#############

figure()
events = generate_events(num_events)


########
# agents
########

agent_specification = ProbabilisticAgentSpecification(max_success_probability,
                                                      sensor_radius,
                                                      station_radius,
                                                      num_sensors)

agents = generate_agents(agent_specification, num_agents)
f(x) = mean_detection_probability(x, events)
problem = PartitionProblem(f, agents)


####################
# visualize scenario
####################

figure()
visualize_pdf(standard_mixture())
visualize_events(events)
colors = generate_colors(agents)
xlim([0, 1])
ylim([0, 1])
savefig("$(fig_path)/scenario.png", pad_inches=0.00, bbox_inches="tight")
title("Scenario")

############
# evaluation
############
function evaluate_solver(solver, name)
  println("$name solver running")
  @time solution = solver(problem)

  figure()
  xlim([0, 1])
  ylim([0, 1])
  colors = generate_colors(agents)
  visualize_agents(agents, colors)
  visualize_solution(problem, solution, colors)
  visualize_events(events)

  coverage = solution.value

  savefig("$(fig_path)/$(to_file(name)).png", pad_inches=0.00, bbox_inches="tight")

  title("$name Solver ($(round(coverage, 3)))")

  @show coverage
end

# evaluate basic solvers
#evaluate_solver(solve_myopic, "Myopic")
#evaluate_solver(solve_random, "Random")
evaluate_solver(solve_sequential, "Sequential")

# evaluate partition solvers
for num_partitions in [2, 4, 8]
  solve_n(p) = solve_n_partitions(num_partitions, p)
  evaluate_solver(solve_n, "Partition-$num_partitions")
end

#@show mean_weight(problem)
#@show total_weight(problem)
