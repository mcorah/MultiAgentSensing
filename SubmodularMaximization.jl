module SubmodularMaximization

#
# Hack in python imports for now
#
using PyCall

const tikzplotlib = PyNULL()
const ag = PyNULL()

function __init__()
  copy!(tikzplotlib, pyimport("tikzplotlib"))
  copy!(ag, pyimport("mpl_toolkits.axes_grid1"))
end
#
# End hack
#

using Base.Iterators

import Base.<

# Abstract interface
export  PartitionProblem, PartitionElement, ElementArray, Solution,
  get_num_agents, solve_block, objective, evaluate_solution, empty

# Explicit interface
export ExplicitPartitionProblem, get_element

# General methods that probably need updates
export marginal_gain, compute_weight, compute_weight_matrix, mean_weight,
total_weight, extract_triangle, visualize_solution

# Agent specification for coverage problems
export Agent, generate_agents,
  generate_colors,
  visualize_agents,
  get_element_indices

# Solvers
export solve_optimal, solve_worst, solve_myopic, solve_random, solve_sequential

# Max now uses isless to exclude types that do not have a total order. We would
# like to have something like this defined for solutions but probably should not
# break things. This is a good compromise
partial_max(a, b) = ifelse(b < a, a, b)
partial_min(a, b) = ifelse(b > a, a, b)

# Interface
# get_block(Agent) = <array of objects associated with agents' block of the
#                     partition matroid>
# get_center(Agent) = <agent center>
#   (Note really a general property but currently defined for all agents)
# plot_element(x) = <Nothing>
# make_agent(agent_specification) = < agent >::AgentType
struct Agent{T}
  center::Array{Float64, 1}
  radius::Float64
  sensors::Array{T, 1}
end

get_block(agent::Agent) = agent.sensors
get_center(agent::Agent) = agent.center

function generate_agents(agent_specification, num_agents)
  [make_agent(agent_specification) for agent in 1:num_agents]
end

#
#
# Abstract partition matroid problems
#
# This interface is suitable for cases where the matroid is defined implicitly
# and solved suboptimally
#
# Interface
#
abstract type PartitionProblem end

# Define dependent types
PartitionElement(::T) where T <: PartitionProblem = PartitionElement(T)
# Subtypes should redefine this method
PartitionElement(::Type{<:PartitionProblem}) =
  error("Element type not defined for this partition problem")

# Defines the structure of a solution for the given matroid or its type
ElementArray(::T) where T = ElementArray(T)
ElementArray(::Type{T}) where T = Vector{PartitionElement(T)}

# Solutions consist of their value and a set of solution elements
struct Solution{PartitionElement}
  value::Float64
  elements::Vector{PartitionElement}
end

objective(p::PartitionProblem, X) = error("Objective not defined for ",
                                          typeof(p))

evaluate_solution(p::PartitionProblem, X) =
  Solution(objective(p, X), X)

# Compare solutions by value
<(a::Solution, b::Solution) = a.value < b.value

# empty array (vector) of solution elements
empty(::T) where T <: PartitionProblem = ElementArray(T)()
empty(::Type{T}) where T <: PartitionProblem = ElementArray(T)()

# By default, store the representation of the matroid in something like a vector
# vector
get_num_agents(p::PartitionProblem) = length(p.partition_matroid)

# This should be an optimal or subotimal solver that outputs the appropriate
# PartitionElement for a block given prior selections
solve_block(p::PartitionProblem, block::Integer, selections::Vector) =
  error("Single agent (block) solver not defined for ", typeof(p))

#
# Concrete partition matroid problems
#
# The objective provides f({x} | Y)
# as objective(x, Y)
#
# The inputs are elements of the blocks
#
# The partition matroid is a Vector of the blocks of the partition matroid.
# The blocks contain whatever the elements of the ground set correspond to.
# For example, the blocks may contain specifications of coverage regions.

struct ExplicitPartitionProblem <: PartitionProblem
  objective::Function
  partition_matroid::Vector
end

# Solution elemnts for concrete partition matroids are indices within the array
# of blocks
#
# (agent_index, block_index)
PartitionElement(::Type{ExplicitPartitionProblem}) = Tuple{Int64,Int64}

get_element(problem::ExplicitPartitionProblem, x) =
  get_element(problem.partition_matroid, x)
get_element(partition_matroid, x) =
  get_block(partition_matroid[x[1]])[x[2]]

objective(p::ExplicitPartitionProblem, X) =
  p.objective(map(x->get_element(p.partition_matroid, x), X))

marginal_gain(f, x, Y) = f(vcat([x], Y)) - f(Y)

compute_weight(f, x, y) = f([x]) - marginal_gain(f, x, [y])

compute_weight(f, X::Array, Y::Array) =
  maximum([compute_weight(f, x, y) for x in X, y in Y])

# Solve explicit partition matroids by iteration over blocks given a set of
# prior selections
function solve_block(p::ExplicitPartitionProblem, block_index::Integer,
                     selections::Vector)

  block::ElementArray(p) = get_element_indices(p, block_index)

  _, index = findmax(map(x->objective(p, [selections; x]), block))

  block[index]
end

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

# Construct index set for ease in manipulation of the matroid
#
# The resulting index is an array of arrays whereas each array corresponds
# to indices for an element in a block of the partition matroid
get_element_indices(p::ExplicitPartitionProblem, xs...) =
  get_element_indices(p.partition_matroid, xs...)

function get_element_indices(agents::Vector{<:Agent}, block::Integer)
  map(1:length(get_block(agents[block]))) do block_index
    (block, block_index)
  end
end
function get_element_indices(agents::Vector{<:Agent})
  map(agents, 1:length(agents)) do agent, agent_index
    get_element_indices(agents, agent_index)
  end
end

include("src/visualization.jl")
include("src/normal_lookup_table.jl")
include("src/coverage/coverage.jl")
include("src/coverage/probabilistic_coverage.jl")
include("src/target_tracking/target_tracking.jl")
include("src/target_tracking/filtering.jl")
include("src/target_tracking/information.jl")
include("src/target_tracking/single_robot_solver.jl")
include("src/target_tracking/multi_robot_solver.jl")
include("src/target_tracking/visualization.jl")

# solvers

# sequential solver

# dag solver

# optimal solver
function solve_optimal(p::ExplicitPartitionProblem)
  indices = get_element_indices(p.partition_matroid)

  v0 = evaluate_solution(p, empty())

  # collect because product returns a tuple
  op(s::Solution, b) = partial_max(s, evaluate_solution(p, collect(b)))

  foldl(op, product(indices...); init=v0)
end

# anti-optimal solver
function solve_worst(p::ExplicitPartitionProblem)
  indices = get_element_indices(p.partition_matroid)

  v0 = Solution(Inf, empty())

  # collect because product returns a tuple
  op(s::Solution, b) = partial_min(s, evaluate_solution(p, collect(b)))

  foldl(op, product(indices...), init=v0)
end

# Myopic solver
function solve_myopic(p::PartitionProblem)
  for ii = 1:get_num_agents(p)
    # Solve given knowledge of no prior decisions
    solution_element = solve_block(p, ii, empty(p))

    push!(selection, solution_element)
  end

  evaluate_solution(p, selection)
end

# random solver
function solve_random(p::ExplicitPartitionProblem)
  indices = get_element_indices(p.partition_matroid)

  evaluate_solution(p, map(x->rand(x), indices))
end

# sequential solver
function solve_sequential(p::PartitionProblem)
  selection = empty(p)

  for ii = 1:get_num_agents(p)
    solution_element = solve_block(p, ii, selection)

    push!(selection, solution_element)
  end

  evaluate_solution(p, selection)
end

############
# dag solver
############

export DAGSolver, solve_dag, sequence, in_neighbors, deleted_edge_weight

abstract type DAGSolver end
# in_neighbors(d::DAGSolver, agent_index) = <agent in neighbors>
# sequence(d::DAGSolver) = <sequence of agent ids>

function solve_dag(d::DAGSolver, p::PartitionProblem)
  indices = get_element_indices(p.partition_matroid)

  # selection is a mapping from agent indices to block_indices
  selection = Dict{Int64, Int64}()

  for agent_index in sequence(d)
    neighbor_selection::ElementArray(p) =
      map(x->(x, selection[x]), in_neighbors(d, agent_index))

    # Index of agent and its solution
    solution_element = solve_block(p, agent_index, neighbor_selection)

    # Store the index of agent's solution
    selection[agent_index] = solution_element[2]
  end

  selection_tuples::ElementArray(p) =
    map(x->(x, selection[x]), sequence(d))

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

    weight += reduce(deleted_edges; init=0.0) do w, edge
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
struct PartitionSolver <: DAGSolver
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
  n = convert(Int64, ceil(total_weight(p)
                          / (get_num_agents(p)*desired_suboptimality)))
  max(1, n)
end

# compute nominal local number of partitions
function compute_local_num_partitions(desired_suboptimality,
                                      p::PartitionProblem)
  W  = compute_weight_matrix(p)

  map(1:length(p.partition_matroid)) do ii
    n = convert(Int64, ceil(sum(W[ii,:]) / (2 * desired_suboptimality)))
    max(1, n)
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

struct RangeSolver <: DAGSolver
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
