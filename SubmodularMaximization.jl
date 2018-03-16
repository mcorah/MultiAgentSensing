module SubmodularMaximization

using Iterators

import Base.<

export  PartitionProblem, PartitionElement, ElementArray, Solution, empty,
  get_num_agents, get_element, objective, evaluate_solution, marginal_gain,
  compute_weight, compute_weight_matrix, mean_weight, total_weight,
  extract_triangle, get_element_indices, visualize_solution

export Agent, generate_agents,
  generate_colors,
  visualize_agents

export solve_optimal, solve_worst, solve_myopic, solve_random, solve_sequential

# Interface
# get_block(Agent) = <array of objects associated with agents' block of the
#                     partition matroid>
# get_center(Agent) = <agent center>
#   (Note really a general property but currently defined for all agents)
# plot_element(x) = <Void>
# make_agent(agent_specification) = < agent >::AgentType
type Agent{T}
  center::Array{Float64, 1}
  radius::Float64
  sensors::Array{T, 1}
end

get_block(agent::Agent) = agent.sensors
get_center(agent::Agent) = agent.center

function generate_agents(agent_specification, num_agents)
  [make_agent(agent_specification) for agent in 1:num_agents]
end

# f({x} | Y)
# the objective takes in an array of elements of the blocks
type PartitionProblem
  objective
  partition_matroid::Array
end

# (agent_index, block_index)
PartitionElement = Tuple{Int64,Int64}
ElementArray = Array{PartitionElement, 1}

get_num_agents(p::PartitionProblem) = length(p.partition_matroid)

type Solution
  value::Float64
  elements::ElementArray
end

# empty tuple
empty() = ElementArray()

<(a::Solution, b::Solution) = a.value < b.value

get_element(partition_matroid, x::PartitionElement) =
  get_block(partition_matroid[x[1]])[x[2]]

objective(p::PartitionProblem, X::ElementArray) =
  p.objective(map(x->get_element(p.partition_matroid, x), X))

evaluate_solution(p::PartitionProblem, X::ElementArray) =
  Solution(objective(p, X), X)

marginal_gain(f, x, Y) = f(vcat([x], Y)) - f(Y)

compute_weight(f, x, y) = f([x]) - marginal_gain(f, x, [y])

compute_weight(f, X::Array, Y::Array) =
  maximum([compute_weight(f, x, y) for x in X, y in Y])

function compute_weight_matrix(p::PartitionProblem)
  indices = get_element_indices(p.partition_matroid)

  n = length(p.partition_matroid)
  weights = zeros(n, n)

  f(x) = objective(p, x)

  for ii in 2:n, jj in 1:ii-1
    w = compute_weight(f, indices[ii], indices[jj])

    weights[ii, jj] = w
    weights[jj, ii] = w
  end

  weights
end

function mean_weight(W::Array)
  n = size(W,1)

  twice_edges = n * (n - 1)

  sum(W) / twice_edges
end

total_weight(W::Array) = sum(W) / 2

mean_weight(p::PartitionProblem) = mean_weight(compute_weight_matrix(p))
total_weight(p::PartitionProblem) = total_weight(compute_weight_matrix(p))

function extract_triangle(A)
  values = Float64[]
  for ii in 2:size(A, 1), jj in 1:ii-1
    push!(values, A[ii, jj])
  end
  values
end

# indexing and solver tools

# construct index set to ease manipulation of the matroid
# note that the resulting index is an array of arrays whereas each array is a
# block of the partition matroid
get_element_indices(agents) = map(agents, 1:length(agents)) do agent, agent_index
  map(1:length(get_block(agent))) do block_index
    (agent_index, block_index)
  end
end

include("visualization.jl")
include("coverage.jl")
include("probabilistic_coverage.jl")

# solvers

# sequential solver

# dag solver

# optimal solver
function solve_optimal(p::PartitionProblem)
  indices = get_element_indices(p.partition_matroid)

  v0 = evaluate_solution(p, empty())

  # collect because product returns a tuple
  op(s::Solution, b) = max(s, evaluate_solution(p, collect(b)))

  foldl(op, v0, product(indices...))
end

# anti-optimal solver
function solve_worst(p::PartitionProblem)
  indices = get_element_indices(p.partition_matroid)

  v0 = Solution(Inf, empty())

  # collect because product returns a tuple
  op(s::Solution, b) = min(s, evaluate_solution(p, collect(b)))

  foldl(op, v0, product(indices...))
end

# myopic solver
function solve_myopic(p::PartitionProblem)
  indices = get_element_indices(p.partition_matroid)

  selection = map(indices) do block
    ii = indmax(map(x->objective(p, [x]), block))
    block[ii]
  end

  evaluate_solution(p, selection)
end

# random solver
function solve_random(p::PartitionProblem)
  indices = get_element_indices(p.partition_matroid)

  evaluate_solution(p, map(x->rand(x), indices))
end

# sequential solver
function solve_sequential(p::PartitionProblem)
  indices = get_element_indices(p.partition_matroid)

  selection = ElementArray()

  for block in indices
    ii = indmax(map(x->objective(p, [selection; x]), block))

    push!(selection, block[ii])
  end

  evaluate_solution(p, selection)
end

############
# dag solver
############

export DAGSolver, solve_dag, sequence, in_neighbors, deleted_edge_weight

abstract DAGSolver
# in_neighbors(d::DAGSolver, agent_index) = <agent in neighbors>
# sequence(d::DAGSolver) = <sequence of agent ids>

function solve_dag(d::DAGSolver, p::PartitionProblem)
  indices = get_element_indices(p.partition_matroid)

  # selection is a mapping from agent indices to block_indices
  selection = Dict{Int64, Int64}()

  for agent_index in sequence(d)
    neighbor_selection::ElementArray = map(x->(x, selection[x]), in_neighbors(d, agent_index))

    values = map(indices[agent_index]) do x
      objective(p, [neighbor_selection; x])
    end

    selection[agent_index] = indmax(values)
  end

  selection_tuples::ElementArray = map(x->(x, selection[x]), sequence(d))

  evaluate_solution(p, selection_tuples)
end

function deleted_edge_weight(d::DAGSolver, W::Array{Float64})
  weight = 0.0

  s = sequence(d)
  for ii in 1:length(s)
    agent_index = s[ii]

    nominal_neighbors = s[1:ii-1]
    neighbors = in_neighbors(d, agent_index)

    deleted_edges = setdiff(nominal_neighbors, neighbors)

    weight += reduce(0.0, deleted_edges) do w, edge
      w + W[agent_index, edge]
    end
  end

  weight
end

function deleted_edge_weight(d::DAGSolver, p::PartitionProblem)
  weights = compute_weight_matrix(p)

  deleted_edge_weight(d, weights)
end

###########################
# basic partitioned solvers
###########################

export PartitionSolver, generate_by_local_partition_size,
  generate_by_global_partition_size, solve_n_partitions

# helper function to construct the partitions for the solver
function construct_partitions(partition_numbers)
  num_partitions = maximum(partition_numbers)

  partitions = [Int64[] for x in 1:num_partitions]

  for agent_index in 1:length(partition_numbers)
    partition_index = partition_numbers[agent_index]

    push!(partitions[partition_index], agent_index)
  end

  partitions
end

# Generic partition solver
type PartitionSolver <: DAGSolver
  # Array partitioning agents
  # Inner arrays are blocks and elements are agent ids
  partitions::Array{Array{Int64,1},1}
  # the index of the block in "partitions" containing a given agent_index
  agent_partition_numbers::Array{Int64,1}

  # Construct the dag using just the partition numbers
  PartitionSolver(x) = new(construct_partitions(x), x)
end

sequence(p::PartitionSolver) = vcat(p.partitions...)

function in_neighbors(p::PartitionSolver, agent_index)
  partition_index = p.agent_partition_numbers[agent_index]

  vcat(p.partitions[1:(partition_index-1)]...)
end

# general random partition solver framework from paper
function generate_by_local_partition_size(local_partition_sizes)
  partition_numbers = map(x->rand(1:x), local_partition_sizes)

  PartitionSolver(partition_numbers)
end

function generate_by_global_partition_size(num_agents, partition_size)
  generate_by_local_partition_size(fill(partition_size, num_agents))
end

# fixed number of partitions
function solve_n_partitions(num_partitions, p::PartitionProblem)
  num_agents = length(p.partition_matroid)

  partition_solver = generate_by_global_partition_size(num_agents,
                                                       num_partitions)

  solve_dag(partition_solver, p)
end

#######################
# Adaptive partitioning
#######################

export compute_global_num_partitions, compute_local_num_partitions,
  generate_global_adaptive, generate_local_adaptive, solve_global_adaptive,
  solve_local_adaptive

# global adaptive number of partitions
function compute_global_num_partitions(desired_suboptimality,
                                       p::PartitionProblem)
  convert(Int64, ceil(total_weight(p)
                      / (get_num_agents(p)*desired_suboptimality)))
end

# compute nominal local number of partitions
function compute_local_num_partitions(desired_suboptimality,
                                      p::PartitionProblem)
  W  = compute_weight_matrix(p)

  map(1:length(p.partition_matroid)) do ii
    convert(Int64, ceil(sum(W[ii,:]) / (2 * desired_suboptimality)))
  end
end

# generate planner using global adaptive number of partitions
function generate_global_adaptive(desired_suboptimality, p::PartitionProblem)
  num_partitions = compute_global_num_partitions(desired_suboptimality, p)

  generate_by_global_partition_size(length(p.partition_matroid), num_partitions)
end

# generate planner using local adaptive number of partitions
function generate_local_adaptive(desired_suboptimality, p::PartitionProblem)
  local_partition_sizes = compute_local_num_partitions(desired_suboptimality, p)

  generate_by_local_partition_size(local_partition_sizes)
end

function solve_global_adaptive(desired_suboptimality, p::PartitionProblem)
  solver = generate_global_adaptive(desired_suboptimality, p)

  solve_dag(solver, p)
end

function solve_local_adaptive(desired_suboptimality, p::PartitionProblem)
  solver = generate_local_adaptive(desired_suboptimality, p)

  solve_dag(solver, p)
end

#######################
# range-limited solvers
#######################

export RangeSolver

type RangeSolver <: DAGSolver
  problem::PartitionProblem
  nominal_solver::DAGSolver
  communication_range::Float64
end

sequence(x::RangeSolver) = sequence(x.nominal_solver)

function in_neighbors(x::RangeSolver, agent_index)
  agents = x.problem.partition_matroid
  nominal_neighbors = in_neighbors(x.nominal_solver, agent_index)

  neighbors = Int64[]
  for neighbor in nominal_neighbors
    dist = norm(get_center(agents[neighbor]) - get_center(agents[agent_index]))

    if dist < x.communication_range
      push!(neighbors, neighbor)
    end
  end

  neighbors
end

end
