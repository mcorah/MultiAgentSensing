using PyPlot
using HDF5, JLD
using Statistics

using SubmodularMaximization

close("all")

name = "connectivity_example"

#num_agents = 50
num_agents = 10

num_sensors = 10
nominal_area = 2.0

num_hops = 1
num_partitions = 8

sensor_radius = sqrt(nominal_area / (num_agents * pi))
station_radius = 2 * sensor_radius

# works out to the maximum radius for all interactions
communication_radius = 3*station_radius

agent_specification = CircleAgentSpecification(sensor_radius, station_radius,
                                               num_sensors)

f(x) = mean_area_coverage(x, 100)

problem = generate_connected_problem(agent_specification, num_agents,
                                     communication_radius=communication_radius,
                                     objective=f)

# Leave an adjacency matrix around for fun
adjacency = make_adjacency_matrix(problem, communication_radius)

function evaluate_solver(make_solver, name)
  println("$name solver running")
  solver = make_solver(problem)

  @time solution = solve_problem(solver, problem, threaded=true)

  figure()
  xlim([0, 1])
  ylim([0, 1])
  colors = generate_colors(problem.partition_matroid)
  visualize_agents(problem.partition_matroid, colors)
  visualize_solution(problem, solution, colors)

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

function make_sequential_solver(p)
  SequentialCommunicationSolver(p, communication_radius)
end

# Converging auction solver
# (should converge in the same time as the sequential solver or faster)
function make_auction_solver(p)
  AuctionSolver(p, communication_radius)
end

# Converging auction solver
# (should converge in the same time as the sequential solver or faster)
function make_local_auction_solver(p)
  LocalAuctionSolver(p, communication_radius)
end

for (make_solver, name) in [
                            (make_hop_solver, "RSP Hops"),
                            (make_sequential_solver, "Sequential"),
                            (make_auction_solver, "Auction"),
                            (make_local_auction_solver, "Local Auction")
                           ]
  evaluate_solver(make_solver, name)
end
