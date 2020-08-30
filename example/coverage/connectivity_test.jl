using PyPlot
using HDF5, JLD
using Statistics
using Base.Iterators

using SubmodularMaximization

close("all")

experiment_name = "connectivity_test"
data_folder = "./data"

reprocess = false

num_sensors = 10
nominal_area = 2.0

num_hops = 1

num_agents = 10:10:100
trials = 1:100

num_steps = 2 .^ (2:4)

sensor_radius(num_agents) = sqrt(nominal_area / (num_agents * pi))
station_radius(num_agents) = 2 * sensor_radius(num_agents)

# works out to the maximum radius for all interactions
communication_radius(num_agents) = 3*station_radius(num_agents)

#
# Define the solver factories
#

function make_hop_solver(p, num_partitions)
  num_agents = length(p.partition_matroid)
  radius = communication_radius(num_agents)

  partition_solver = generate_by_global_partition_size(num_agents,
                                                       num_partitions)

  MultiHopSolver(p, solver=partition_solver,
                 communication_range=radius,
                 num_hops=num_hops)
end

function make_sequential_solver(p)
  num_agents = length(p.partition_matroid)
  radius = communication_radius(num_agents)

  SequentialCommunicationSolver(p, radius)
end

# Converging auction solver
# (should converge in the same time as the sequential solver or faster)
function make_auction_solver(p, x...)
  num_agents = length(p.partition_matroid)
  radius = communication_radius(num_agents)

  AuctionSolver(p, radius, x...)
end

# Converging auction solver
# (should converge in the same time as the sequential solver or faster)
function make_local_auction_solver(p, x...)
  num_agents = length(p.partition_matroid)
  radius = communication_radius(num_agents)

  LocalAuctionSolver(p, radius, x...)
end

#
# Assemble solvers and test specifications
#

solvers_strings =
  [map(n -> (p->make_hop_solver(p, n), string("RSP-",n)), num_steps)...,
   (make_sequential_solver, "Sequential"),
   map(n -> (p->make_auction_solver(p, n), string("Auction-",n)), num_steps)...,
   (p->make_auction_solver(p), string("Auction")),
   map(n -> (p->make_local_auction_solver(p, n), string("Local-Auction-",n)),
       num_steps)...,
   (p->make_local_auction_solver(p), string("Local-Auction")),
  ]

solver_factories = map(first, solvers_strings)
solver_strings = map(last, solvers_strings)

solver_inds = 1:length(solver_factories)
all_tests = product(solver_inds, num_agents, trials)

# We need the problem to be connected to avoid bad things
function generate_connected_problem(agent_specification, num_agents)
  agents = generate_agents(agent_specification, num_agents)

  f(x) = mean_area_coverage(x, 100)

  problem = ExplicitPartitionProblem(f, agents)

  radius = communication_radius(num_agents)
  adjacency = make_adjacency_matrix(problem, radius)

  if is_connected(adjacency)
    problem
  else
    generate_connected_problem(agent_specification, num_agents)
  end
end

#
# Construct and run trials
#

function trial_fun(x)
  (solver_ind, num_agents, trial) = x
  make_solver = solver_factories[solver_ind]

  agent_specification = CircleAgentSpecification(sensor_radius(num_agents),
                                                 station_radius(num_agents),
                                                 num_sensors)

  problem = generate_connected_problem(agent_specification, num_agents)

  solver = make_solver(problem)

  solution = solve_problem(solver, problem, threaded=false)

  (solver=solver, problem=problem, solution=solution)
end
function print_summary(x)
  (solver_ind, num_agents, trial) = x

  println("Num. Agents: ", num_agents,
          " Solver: ", solver_strings[solver_ind],
          " Trial: ", trial)
end

@time results = run_experiments(all_tests,
                                trial_fun=trial_fun,
                                print_summary=print_summary,
                                experiment_name=experiment_name,
                                data_folder=data_folder,
                                reprocess=reprocess
                               )
