# This file provides versions of solvers based on communication graphs

using Statistics
using LinearAlgebra

# Methods for evaluating communication statistics
#
# WARNING: Call after solving a problem
#
# * communication_span: "time" span for communications.
# * communication_messages: Total number of messages sent during execution
#   including each hop
# * communication_volume: Sum of messages and hops multiplied by the number of
#   decisions in each message

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

neighbors(adjacency::Matrix{<:Integer}, index::Integer) =
  findall(x -> x > 0, adjacency[index,:])

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
    for index in neighbors(adjacency, least)
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

path_distance(x...) = shortest_path(x...)[1]

# Check connectivity via matrix exponentiation
function is_connected(adjacency::Matrix)
  paths = (adjacency + I) ^ (size(adjacency, 1) - 1)

  all(x -> x > 0, paths)
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

# Plots the shortest path.
# * Note that ret[1].remove() will remove the path
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
# Note: in-neighbors for the DAG solver
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
  x.hops * (communication_span(x.nominal_solver) - 1)
# Number of messages multiplied by the number of hops the message travels
function communication_messages(x::MultiHopSolver)
  num_agents = solver_rank(x)

  sum(1:num_agents) do a
    mapreduce(+, in_neighbors(x, a), init=0) do b
      path_distance(x.adjacency, a, b)
    end
  end
end
# Robots send a single decision at a time
communication_volume(x::MultiHopSolver) = communication_messages(x)

#################################################
# Sequential solver with communication statistics
#################################################

struct SequentialCommunicationSolver
  adjacency::Array
end

function SequentialCommunicationSolver(problem::PartitionProblem,
                                       communication_range)
  adjacency = make_adjacency_matrix(problem, communication_range)

  SequentialCommunicationSolver(adjacency)
end

solver_rank(x::SequentialCommunicationSolver) = size(x.adjacency, 1)

# Simply solve using the standard sequential solver
function solve_problem(::SequentialCommunicationSolver,
                       p::PartitionProblem; kwargs...)
  solve_sequential(p)
end
# Each robot except the first receives a single message from the previous
# containing all prior decisions. Include path lengths
function communication_messages(x::SequentialCommunicationSolver)
  num_agents = solver_rank(x)

  sum(1:num_agents-1) do index
    path_distance(x.adjacency, index, index + 1)
  end
end
# Sum of message hops times message size
# The message size is equal to the index of the sender
function communication_volume(x::SequentialCommunicationSolver)
  num_agents = solver_rank(x)

  sum(1:num_agents-1) do index
    index * path_distance(x.adjacency, index, index + 1)
  end
end
# Span is the distance from each agent to the next
# This will be the same as the communication_messages
function communication_span(x::SequentialCommunicationSolver)
  num_agents = solver_rank(x)

  sum(1:num_agents - 1) do index
    path_distance(x.adjacency, index, index + 1)
  end
end

################
# Auction solver
################

# Based on the distributed planner in
# by Choi et al. Transactions on Robots, 2009
# as well as the work of
# Luo et al. IROS, 2016
#
# Modifications:
# * Each agent has its own set of actions as for the definition of the
#   PartitionProblem type
# * Messages are maps from ids to values rather than vectors containing all
#   assignments

# We will evaluate message counts and such
mutable struct AuctionSolver
  adjacency::Matrix

  # We will allow for convergence before reaching the span
  nominal_span::Integer

  span::Integer
  messages::Integer
  volume::Integer

  solved::Bool

  AuctionSolver(adjacency, span) = new(adjacency, span, 0, 0, 0, false)
end

# Second term is the upper bound on convergence time
# (longest possible path times number of assignments)
AuctionSolver(adjacency) = AuctionSolver(adjacency, size(adjacency, 1)^2)

function AuctionSolver(problem::PartitionProblem, communication_range)
  adjacency = make_adjacency_matrix(problem, communication_range)

  AuctionSolver(adjacency)
end


# Contains details for a given agent (its assignments and index)
struct AuctionAgent
  index::Integer
  assignments::Vector{ExplicitSolutionElement}
end

AuctionAgent(index::Integer) = AuctionAgent(index, ExplicitSolutionElement[])
# Constructor for incrementing the assignments
AuctionAgent(a::AuctionAgent, s::Vector) = AuctionAgent(a.index, s)

get_index(x::AuctionAgent) = x.index
# Returns the agent's current list of assignments to all agents
get_assignments(x::AuctionAgent) = x.assignments
# Returns the assignment to this agent
get_assignment(agent::AuctionAgent) =
  first(filter(x->first(x) == get_index(agent), get_assignments(agent)))

neighbors(adjacency::Matrix, a::AuctionAgent) = neighbors(adjacency, a.index)

# Get the set of neighbors by adjacency and filtering
function neighbors(adjacency::Matrix, agents::Vector{AuctionAgent},
                   a::AuctionAgent)
  ns = neighbors(adjacency, a)

  filter(x -> in(x.index,ns), agents)
end

is_solved(x::AuctionSolver) = x.solved
assert_solved(x::AuctionSolver) =
  is_solved(x) ? true : error("Solve problem before calling")

communication_span(x::AuctionSolver) = assert_solved(x) && x.span
communication_messages(x::AuctionSolver) = assert_solved(x) && x.messages
communication_volume(x::AuctionSolver) = assert_solved(x) && x.volume

function converged(agents::Vector{AuctionAgent})
  # As a proxy for requiring maximal assignments, use the fact that the rank is
  # equal to the number of agents
  ( all(x -> length(x.assignments) == length(agents), agents)
  # Assert that all agents have the same assignments
  && all(x -> Set(x.assignments) == Set(first(agents).assignments), agents))
end

# Priority goes to agents with lower indices
priority(a::Int64, b::Int64) = a < b
priority(a::Tuple{Int64,Int64}, b::Tuple{Int64,Int64}) =
  priority(first(a), first(b)) ||
  (first(a) == first(b) && priority(last(a), last(b)))

# Run general greedy assignment
#
# Being careful to
# * Break ties deterministically, even within proposed assignments
#   (to ensure convergence)
# * Track independence
function greedy_assignment(problem::ExplicitPartitionProblem,
                           agent::AuctionAgent,
                           assignments::Vector)
  block = get_element_indices(problem, get_index(agent))

  # Construct a local ground with the current block and incoming assignments
  G = convert(Vector{ExplicitSolutionElement}, union(block, assignments...))

  # Assignments
  X = ExplicitSolutionElement[]

  # Greedy selection process
  while !isempty(G)
    values = map(G) do x
      marginal_gain(y->objective(problem, y), x, X)
    end

    max_value = maximum(values)
    maximizers = G[findall(x->x==max_value, values)]

    # Priority goes to least index
    max_priority = first(sort(maximizers; lt=priority))

    # Conditionally add the new solution element
    if independent(union(X, [max_priority]))
      X = union(X, [max_priority])
    end

    # Element either assigned or cannot be added to this or future solutions
    setdiff!(G, [max_priority])
  end

  X
end

function solve_problem(solver::AuctionSolver, problem::PartitionProblem;
                       kwargs...)
  num_agents = get_num_agents(problem)
  adjacency = solver.adjacency
  agents = map(x->AuctionAgent(x), 1:num_agents)

  for ii in 1:solver.nominal_span
    # Update span
    solver.span += 1

    updated_agents = map(agents) do agent
      ns = neighbors(adjacency, agents, agent)

      assignments = map(get_assignments, ns)

      # Update messages and volume
      solver.messages += length(ns)
      solver.volume += sum(length, assignments)

      # Construct the new set of assigments
      new_assignments = greedy_assignment(problem, agent, assignments)

      AuctionAgent(agent, new_assignments)
    end

    agents[:] = updated_agents

    if converged(agents)
      break
    end
  end

  solver.solved = true

  assignments = map(get_assignment, agents)

  # Return the solution
  evaluate_solution(problem, assignments)
end
