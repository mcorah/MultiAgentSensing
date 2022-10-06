module SubmodularMaximization

#
# Hack in python imports for now
#
using PyCall

const tikzplotlib = PyNULL()
const ag = PyNULL()

function __init__()
  try
    copy!(tikzplotlib, pyimport("tikzplotlib"))
  catch e
    println("Could not import tikzplotlib, trying matploblib2tikz instead")
    copy!(tikzplotlib, pyimport("matplotlib2tikz"))
  end

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
export ExplicitPartitionProblem, get_element, get_agent, get_center

# General methods that probably need updates
export marginal_gain, compute_weight, compute_weight_matrix, mean_weight,
total_weight, extract_triangle, visualize_solution

# Agent specification for coverage problems
export Agent, generate_agents,
  generate_colors,
  visualize_agents,
  get_element_indices,
  independent

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
abstract type PartitionProblem{PartitionElement} end

# Define dependent types
PartitionElement(::PartitionProblem{T}) where T = T
# Subtypes should redefine this method
PartitionElement(::Type{<:PartitionProblem{T}}) where T = T

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
# Define the solution for individual solution elements
objective(p::PartitionProblem{T}, X::T) where T = objective(p, [X])

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

# Solution elemnts for concrete partition matroids are indices within the array
# of blocks
#
# (agent_index, block_index)
const ExplicitSolutionElement = Tuple{Int64,Int64}
struct ExplicitPartitionProblem <: PartitionProblem{ExplicitSolutionElement}
  objective::Function
  partition_matroid::Vector
end

get_element(problem::ExplicitPartitionProblem, x) =
  get_element(problem.partition_matroid, x)
get_element(partition_matroid, x) =
  get_block(partition_matroid[x[1]])[x[2]]

get_agent(problem::ExplicitPartitionProblem, x) =
  get_agent(problem.partition_matroid, x)
get_agent(partition_matroid::Vector, x) = partition_matroid[x]

get_center(p::PartitionProblem, x) = get_center(get_agent(p, x))

objective(p::ExplicitPartitionProblem, X::Vector{ExplicitSolutionElement}) =
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

function get_element_indices(agents::Vector, block::Integer)
  map(1:length(get_block(agents[block]))) do block_index
    (block, block_index)
  end
end
function get_element_indices(agents::Vector)
  map(agents, 1:length(agents)) do agent, agent_index
    get_element_indices(agents, agent_index)
  end
end

# A set is independent if it contains at most one assignment to each agent
# Tuple consist of agent and agent-index
independent(x) = length(x) == length(Set(map(first, x)))

# Returns true if any element of Y can be added to X
can_augment(x, Y::Vector) = any(y->independent(vcat(x, y)), Y)
# Can agent index add solution elements to x in a partition matroid
can_augment(x, index::Int64) = !in(index, map(first, x))
# Can a given solution be added to x
can_augment(x, e::ExplicitSolutionElement) = can_augment(x, first(e))

include("src/utils.jl")
include("src/visualization.jl")
include("src/normal_lookup_table.jl")
include("src/solvers.jl")

# Support for solvers with range-based communication graphs
export make_adjacency_matrix, plot_adjacency, plot_shortest_path,
make_hop_adjacency, neighbors, shortest_path, path_distance, is_connected,
generate_connected_problem, solve_multi_hop

export communication_span, communication_messages, communication_volume

export MultiHopSolver, SequentialCommunicationSolver, AuctionSolver,
LocalAuctionSolver

include("src/communications_graphs_solvers.jl")

include("src/coverage/coverage.jl")
include("src/coverage/probabilistic_coverage.jl")

include("src/target_tracking/target_tracking_top.jl")

end
