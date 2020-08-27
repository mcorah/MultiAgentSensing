using PyPlot
using HDF5, JLD
using Statistics

using SubmodularMaximization

close("all")

name = "connectiviy_example"
data_path = "./data/$name"
mkpath(data_path)

#num_agents = 50
num_agents = 10

num_sensors = 10
nominal_area = 2.0

num_hops = 2
num_partitions = 3

sensor_radius = sqrt(nominal_area / (num_agents * pi))
station_radius = 2 * sensor_radius

# works out to 2/3 of the radius for all interactions after two hops
communication_radius = station_radius

agent_specification = CircleAgentSpecification(sensor_radius, station_radius,
                                               num_sensors)

agents = generate_agents(agent_specification, num_agents)

f(x) = mean_area_coverage(x, 100)

problem = ExplicitPartitionProblem(f, agents)

function evaluate_solver(make_solver, name)
  println("$name solver running")
  solver = make_solver(problem)

  @time solution = solve(solver, problem, threaded=true)

  figure()
  xlim([0, 1])
  ylim([0, 1])
  colors = generate_colors(agents)
  visualize_agents(agents, colors)
  #visualize_solution(problem, solution, colors)

  plot_adjacency(problem, communication_radius)

  gca().set_aspect("equal")
  #savefig("$(fig_path)/$(to_file(name)).png", pad_inches=0.00, bbox_inches="tight")

  coverage = solution.value

  title("$name Solver ($coverage)")

  println("Coverage: ", coverage)
  println("Span: ", communication_span(solver))
  println("Messages: ", communication_messages(solver))
  println("Volume: ", communication_volume(solver))
end

function make_hop_solver(p)
  num_agents = length(p.partition_matroid)

  partition_solver = generate_by_global_partition_size(num_agents,
                                                       num_partitions)

  MultiHopSolver(p, solver=partition_solver,
                 communication_range=communication_radius,
                 num_hops=num_hops)
end

evaluate_solver(make_hop_solver, "RSP Hops")
