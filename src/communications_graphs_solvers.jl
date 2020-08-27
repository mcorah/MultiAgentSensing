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

# Ugly implementation of Dijkstra
function shortest_path(adjacency::Matrix, a::Integer, b::Integer)
  dist = fill(Inf, size(adjacency, 1))
  dist[a] = 0
  prev = fill(0, size(adjacency, 1))

  inds = collect(1:size(adjacency, 1))

  sort_inds() = sort!(inds, by=x->dist[x], rev = true)

  while !isempty(inds)
    sort_inds()
    least = pop!(inds)

    # Update
    for index in findall(x -> x > 0, adjacency[least,:])
      new_dist = dist[least] + 1
      if dist[index] > new_dist
        dist[index] = new_dist
        prev[index] = least
      end
    end
  end

  path = [b]
  while prev[path[end]] != 0
    push!(path, prev[path[end]])
  end

  (dist[b], path)
end

function plot_adjacency(problem::ExplicitPartitionProblem, range::Real)
  adjacency = make_adjacency_matrix(problem, range)

  plot_adjacency(problem, adjacency)
end
function plot_adjacency(problem::ExplicitPartitionProblem, adjacency::Array)
  num_agents = get_num_agents(problem)

  for ii in 1:num_agents, jj in 1:ii-1
    center_ii = get_center(problem, ii)
    center_jj = get_center(problem, jj)

    if adjacency[ii,jj] > 0
      x = map(first, [center_ii, center_jj])
      y = map(last, [center_ii, center_jj])
      plot(x, y, color="black")
    end
  end
end

function plot_shortest_path(problem::ExplicitPartitionProblem, adjacency, a, b)
  (dist, path) = shortest_path(adjacency, a, b)

  x = Float64[]
  y = Float64[]
  for ii in path
    center = get_center(problem, ii)

    push!(x , first(center))
    push!(y, last(center))
  end

  plot(x, y, color="red")
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

# Multi-hop communications

solver_rank(x::MultiHopSolver) = solver_rank(x.nominal_solver)
communication_span(x::MultiHopSolver) =
(x.hops - 1) * communication_span(x.nominal_solver)
function communication_messages(x::MultiHopSolver)
  num_agents = solver_rank(x)
  sum(agent_index->length(in_neighbors(x, agent_index)), 1:num_agents)
end
# Robots send a single decision at a time
communication_volume(x::MultiHopSolver) = communication_messages(x)
