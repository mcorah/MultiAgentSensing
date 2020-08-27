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

sensor_radius = sqrt(nominal_area / (num_agents * pi))
station_radius = 2 * sensor_radius

# works out to 2/3 of the radius for all interactions after two hops
communication_radius = station_radius

agent_specification = CircleAgentSpecification(sensor_radius, station_radius,
                                               num_sensors)

agents = generate_agents(agent_specification, num_agents)

f(x) = mean_area_coverage(x, 100)

problem = ExplicitPartitionProblem(f, agents)

function evaluate_solver(solver, name)
  println("$name solver running")
  @time solution = solver(problem)

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

  @show coverage
end

function solve_hops(p)
  solve_multi_hop(p::PartitionProblem;
                  num_partitions = 3,
                  num_hops = 2,
                  communication_range = communication_radius,
                  threaded=false)
end

evaluate_solver(solve_hops, "Sequential")
