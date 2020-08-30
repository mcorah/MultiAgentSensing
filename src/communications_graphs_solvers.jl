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

# Abstract auctions type for brevity when defining the local auctions type
abstract type AbstractAuctionSolver end
abstract type AbstractAuctionAgent end

# We will evaluate message counts and such
mutable struct AuctionSolver <: AbstractAuctionSolver
  adjacency::Matrix

  # We will allow for convergence before reaching the maximum number of steps
  # (equivalent span plus 1)
  nominal_steps::Integer

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
struct AuctionAgent <: AbstractAuctionAgent
  index::Integer
  assignments::Vector{ExplicitSolutionElement}
end

AuctionAgent(index::Integer) = AuctionAgent(index, ExplicitSolutionElement[])
# Constructor for incrementing the assignments
AuctionAgent(a::AuctionAgent, s::Vector) = AuctionAgent(a.index, s)

get_index(x::AbstractAuctionAgent) = x.index
# Returns the agent's current list of assignments to all agents
get_assignments(x::AbstractAuctionAgent) = x.assignments
# Returns the assignment to this agent
get_assignment(agent::AbstractAuctionAgent) =
  first(filter(x->first(x) == get_index(agent), get_assignments(agent)))

num_assignments(agent::AbstractAuctionAgent) = length(get_assignments(agent))

neighbors(adjacency::Matrix, a::AbstractAuctionAgent) =
  neighbors(adjacency, a.index)

# Get the set of neighbors by adjacency and filtering
function neighbors(adjacency::Matrix, agents::Vector{<:AbstractAuctionAgent},
                   a::AbstractAuctionAgent)
  ns = neighbors(adjacency, a)

  filter(x -> in(x.index,ns), agents)
end

is_solved(x::AbstractAuctionSolver) = x.solved
assert_solved(x::AbstractAuctionSolver) =
  is_solved(x) ? true : error("Solve problem before calling")

communication_span(x::AbstractAuctionSolver) = assert_solved(x) && x.span
communication_messages(x::AbstractAuctionSolver) =
  assert_solved(x) && x.messages
communication_volume(x::AbstractAuctionSolver) = assert_solved(x) && x.volume

function converged(agents::Vector{<:AbstractAuctionAgent})
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

    # Remove elements that cannot be added to the solution
    filter!(x->can_augment(X, x), G)
  end

  X
end

function solve_problem(solver::AuctionSolver, problem::PartitionProblem;
                       kwargs...)
  num_agents = get_num_agents(problem)
  adjacency = solver.adjacency
  agents = map(x->AuctionAgent(x), 1:num_agents)

  for ii in 1:solver.nominal_steps
    # Update span
    if ii != 1
      # Skip the first round (which we can complete without communication)
      solver.span += 1
    end

    updated_agents = map(agents) do agent
      ns = neighbors(adjacency, agents, agent)

      assignments = map(get_assignments, ns)

      # Update messages and volume
      if ii != 1
        solver.messages += length(ns)
        solver.volume += mapreduce(num_assignments, +, ns, init=0)
      end

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

######################
# Local auction solver
######################

# This is a similar auction solver but does not assume that agents can
# re-evaluate others' objective values

# We will evaluate message counts and such
mutable struct LocalAuctionSolver <: AbstractAuctionSolver
  adjacency::Matrix

  # We will allow for convergence before reaching the maximum number of steps
  # (equivalent span plus 1)
  nominal_steps::Integer

  span::Integer
  messages::Integer
  volume::Integer

  solved::Bool

  LocalAuctionSolver(adjacency, span) = new(adjacency, span, 0, 0, 0, false)
end

# Second term is the upper bound on convergence time
# (longest possible path times number of assignments)
LocalAuctionSolver(adjacency) =
  LocalAuctionSolver(adjacency, size(adjacency, 1)^2)

function LocalAuctionSolver(problem::PartitionProblem, communication_range)
  adjacency = make_adjacency_matrix(problem, communication_range)

  LocalAuctionSolver(adjacency)
end


# Contains details for a given agent (its assignments and index)
struct LocalAuctionAgent <: AbstractAuctionAgent
  index::Integer
  assignments::Vector{ExplicitSolutionElement}
  values::Vector{Float64}
end

LocalAuctionAgent(index::Integer) =
  LocalAuctionAgent(index, ExplicitSolutionElement[], Float64[])
# Constructor for incrementing the assignments
LocalAuctionAgent(a::LocalAuctionAgent, s::Vector{ExplicitSolutionElement},
                  v::Vector{Float64}) = LocalAuctionAgent(a.index, s, v)

get_values(x::LocalAuctionAgent) = x.values

get_value_assignment(x::LocalAuctionAgent, index::Integer) =
  (get_values(x)[index], get_assignments(x)[index])

function first_difference(a::LocalAuctionAgent, b::LocalAuctionAgent)
  len = min(num_assignments(a), num_assignments(b))
  findfirst(x-> x[1] != x[2],
            collect(zip(get_assignments(a)[1:len],
                        get_assignments(b)[1:len])))
end

function dominates(a::Tuple{Float64,ExplicitSolutionElement},
                   b::Tuple{Float64,ExplicitSolutionElement})
  first(a) > first(b) ||
    (first(a) == first(b) && priority(last(a), last(b)))
end
# Return true if any solution element in A beats a solution in B
# before B beats A
function dominates(a::LocalAuctionAgent, b::LocalAuctionAgent)
  index = first_difference(a, b)

  if isnothing(index)
    # Default: solutions are identical or one has additional elements
    num_assignments(a) > num_assignments(b)
  else
    dominates(get_value_assignment(a, index), get_value_assignment(b, index))
  end
end

# Local maximization when updating another agents' solution.
# Chop off the incoming solution after winning because we cannot re-evaluate
# later values
#
# Proceed from the first difference. Maximize until winning or no other
# solution. Chop the incoming assignments and append new solution element
function local_agent_maximize(problem::ExplicitPartitionProblem,
                              agent::LocalAuctionAgent,
                              other_agent::LocalAuctionAgent)
  difference_index = first_difference(agent, other_agent)

  if isnothing(difference_index)
    # We already know that other_agent is not dominated
    difference_index = num_assignments(agent) + 1
  end

  for ii in difference_index:num_assignments(other_agent)+1
    prev_assignments = get_assignments(other_agent)[1:ii-1]
    prev_values = get_values(other_agent)[1:ii-1]

    x = solve_block(problem, get_index(agent), prev_assignments)

    value = marginal_gain(y->objective(problem, y), x, prev_assignments)

    if num_assignments(other_agent) < ii ||
      dominates((value, x), get_value_assignment(other_agent, ii))

      return LocalAuctionAgent(agent, vcat(prev_assignments, x),
                               vcat(prev_values, value))
    end
  end

  error("Failed to update solution")
end

# Update the agent using only local information
#
# * Compare solutions, solution with the first winning selection dominates
#   * Return the identity if the local agent wins
# * If agent already has a solution element in the new solution, return this
#   solution (both will have the same solution element for the local agent by
#   monotonicity of solutions)
# * Proceed from the first difference. Maximize until winning or no other
#   solution. Chop the incoming assignments and append new solution element
#
function local_agent_update(problem::ExplicitPartitionProblem,
                            agent::LocalAuctionAgent,
                            other_agent::LocalAuctionAgent)
  if dominates(agent, other_agent)
    agent
  elseif !can_augment(get_assignments(other_agent), get_index(agent))
    LocalAuctionAgent(agent, get_assignments(other_agent),
                      get_values(other_agent))
  else
    local_agent_maximize(problem, agent, other_agent)
  end
end

function solve_problem(solver::LocalAuctionSolver, problem::PartitionProblem;
                       kwargs...)
  num_agents = get_num_agents(problem)
  adjacency = solver.adjacency
  agents = map(x->LocalAuctionAgent(x), 1:num_agents)

  for ii in 1:solver.nominal_steps
    # Update span
    if ii != 1
      # Skip the first round (which we can complete without communication)
      solver.span += 1
    end

    updated_agents = map(agents) do agent
      ns = neighbors(adjacency, agents, agent)

      # Update messages and volume
      if ii != 1
        solver.messages += length(ns)
        solver.volume += mapreduce(num_assignments, +, ns, init=0)
      end

      # Try maximizing on all neighbors in sequence.
      #
      # Alternatively, I could identify which solution dominates
      dominating_agent = foldl(ns, init=agent) do best, next
        dominates(next, best) ? next : best
      end

      local_agent_update(problem, agent, dominating_agent)
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
