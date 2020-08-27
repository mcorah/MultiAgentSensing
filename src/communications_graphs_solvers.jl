# This file provides versions of solvers based on communication graphs

using Statistics
using LinearAlgebra

# Methods for evaluating communication statistics
#
# WARNING: Call after solving a problem
#
# * communication_span: "time" span for communications.
# * communication_messages: Total number of messages sent during execution
# * communication_volume: Sum of messages multiplied by the number of decisions
#   in each message

###############################
# Adjacenty matrix and plotting
###############################

function make_adjacency_matrix(problem::ExplicitPartitionProblem, range)
  num_agents = get_num_agents(problem)

  adjacency = zeros(Int64, num_agents, num_agents)

  for ii in 1:num_agents, jj in 1:ii-1
    center_ii = get_center(get_agent(problem, ii))
    center_jj = get_center(get_agent(problem, jj))

    if norm(center_ii - center_jj) <= range
      adjacency[ii,jj] = 1
      adjacency[jj,ii] = 1
    end
  end

  adjacency
end

# Ugly lines to propagate the matrix a few hops
function make_hop_adjacency(adjacency::Matrix{Int64}, hops::Int64)
  mat = (adjacency + I)^hops
  map(x -> (x > 0 ? 1 : 0), mat)
  for ii = 1:size(mat, 1)
    mat[ii, ii] = 0
  end

  mat
end

function plot_adjacency(problem::ExplicitPartitionProblem, range::Real)
  adjacency = make_adjacency_matrix(problem, range)

  plot_adjacency(problem, adjacency)
end
function plot_adjacency(problem::ExplicitPartitionProblem, adjacency::Array)
  num_agents = get_num_agents(problem)

  for ii in 1:num_agents, jj in 1:ii-1
    center_ii = get_center(get_agent(problem, ii))
    center_jj = get_center(get_agent(problem, jj))

    if adjacency[ii,jj] > 0
      x = map(first, [center_ii, center_jj])
      y = map(last, [center_ii, center_jj])
      plot(x, y, color="black")
    end
  end
end

##################
# Multi-hop solver
##################

struct MultiHopSolver <: DAGSolver
  adjacency::Array
  hop_adjacency::Array
  hops::Integer
  nominal_solver::DAGSolver

  function MultiHopSolver(problem::PartitionProblem;
                          solver::DAGSolver,
                          communication_range,
                          num_hops)

    adjacency = make_adjacency_matrix(problem, communication_range)

    hop_adjacency = make_hop_adjacency(adjacency, num_hops)

    new(adjacency, hop_adjacency, num_hops, solver)
  end
end

sequence(x::MultiHopSolver) = sequence(x.nominal_solver)
partitions(x::MultiHopSolver) = partitions(x.nominal_solver)

# Keep only neighbors that are within the communication range
function in_neighbors(x::MultiHopSolver, agent_index)
  nominal_neighbors = in_neighbors(x.nominal_solver, agent_index)

  neighbors = Int64[]
  for neighbor in nominal_neighbors
    if x.hop_adjacency[agent_index, neighbor] > 0
      push!(neighbors, neighbor)
    end
  end

  neighbors
end

function solve_multi_hop(p::PartitionProblem;
                         num_partitions,
                         communication_range,
                         num_hops,
                         threaded=false)
  num_agents = length(p.partition_matroid)

  partition_solver = generate_by_global_partition_size(num_agents,
                                                       num_partitions)

  multi_hop_solver = MultiHopSolver(p, solver=partition_solver,
                                    communication_range=communication_range,
                                    num_hops=num_hops)

  solve_dag(multi_hop_solver, p, threaded=threaded)
end

# Multi-hop communications

rank(x::MultiHopSolver) = rank(x.nominal_solver)
communication_span(x::MultiHopSolver) =
(x.hops - 1) * communication_span(x.nominal_solver)
function communication_messages(x::MultiHopSolver)
  num_agents = rank(x)
  sum(agent_index->length(in_neighbors(x, agent_index)), 1:num_agents)
end
# Robots send a single decision at a time
communication_volume(x::MultiHopSolver) = communication_messages(x)
