using PyPlot
using HDF5, JLD
using Statistics
using Base.Iterators

using SubmodularMaximization
using RosDataProcess

close("all")

experiment_name = "connectivity_test"
data_folder = "./data"

reprocess = false

num_sensors = 10
nominal_area = 2.0

num_hops = 1

num_agents = 10:10:50
trials = 1:50

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

#
# Construct and run trials
#

function trial_fun(x)
  (solver_ind, num_agents, trial) = x
  make_solver = solver_factories[solver_ind]

  agent_specification = CircleAgentSpecification(sensor_radius(num_agents),
                                                 station_radius(num_agents),
                                                 num_sensors)

  problem = generate_connected_problem(agent_specification, num_agents,
                                       objective =
                                        x -> mean_area_coverage(x, 100),
                                       communication_radius =
                                        communication_radius(num_agents))

  solver = make_solver(problem)

  solution = solve_problem(solver, problem, threaded=false)

  (solver=solver, solution=solution)
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

solver_groups = 3
solver_configurations = length(num_steps) + 1
# One color for each solver group
colors = repeat(RosDataProcess.generate_colors(solver_groups), inner=num_steps)
# One style for each solver configuration
style_atoms = [(0, (1, 1)), # densely dotted
               (0, (2, 1, 1, 0)), # densely dashed
               (0, (2, 1, 1, 1, 1, 0)), # dashdotted
               "-"
              ]
styles = repeat(style_atoms, outer=solver_groups)

# Plot objective values
series_by_solver = Any[]
for solver_ind in solver_inds
  solver_data = map(product(num_agents, trials)) do (num_agents, trial)
    results[solver_ind, num_agents, trial]
  end
  push!(series_by_solver, TimeSeries(num_agents, solver_data))
end

plot_specs = [
              (get = x -> x.solution.value,
               ylabel = "Objective (fraction of unit area)",
               name = "objective"),
              (get = x -> communication_span(x.solver),
               ylabel = "Messaging span",
               name = "span"),
              (get = x -> communication_messages(x.solver),
               ylabel = "Messages sent",
               name = "messages",
               after = [loglog]
              ),
              (get = x -> communication_volume(x.solver),
               ylabel = "Messaging volume",
               name = "volume",
               after = [loglog]
              ),
             ]

fig_dir = "fig/connectivity_test"
if !isdir(fig_dir)
  mkpath(fig_dir)
end

for spec in plot_specs
  figure()
  for solver_ind in reverse(solver_inds)
    data = map(spec.get, series_by_solver[solver_ind])
    plot_trials(data,
                label=solver_strings[solver_ind],
                linestyle=styles[solver_ind],
                color=colors[solver_ind],
                mean=true,
                standard_error=true)
  end
  legend(ncol=3)
  xlabel("Num. Agents")
  ylabel(spec.ylabel)

  if in(:after, keys(spec))
    foreach(x->x(), spec.after)
  end

  save_latex(fig_dir, spec.name)
end
