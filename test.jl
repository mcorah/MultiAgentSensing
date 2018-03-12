using PyPlot
using SubmodularMaximization

num_agents = 10
num_sensors = 3
nominal_area = 1.0

sensor_radius = sqrt(nominal_area / (num_agents * pi))
station_radius = 2 * sensor_radius

agent_specification = AgentSpecification(sensor_radius, station_radius,
                                         num_sensors)

agents = generate_coverage_agents(agent_specification, num_agents)

f(x) = mean_area_coverage(x, 100)
problem = PartitionProblem(f, agents)

function evaluate_solver(solver, name)
  println("$name solver running")
  @time solution = solver(problem)

  figure()
  xlim([0, 1])
  ylim([0, 1])
  colors = generate_colors(agents)
  visualize_agents(agents, colors)
  visualize_solution(problem, solution, colors)

  coverage = solution.value

  title("$name Solver ($coverage)")

  @show coverage
end

evaluate_solver(solve_optimal, "Optimal")
evaluate_solver(solve_worst, "Worst-case")
evaluate_solver(solve_myopic, "Myopic")
evaluate_solver(solve_random, "Random")
evaluate_solver(solve_sequential, "Sequential")
