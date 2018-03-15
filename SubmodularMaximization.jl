module SubmodularMaximization

using Iterators

import Base.<

export  PartitionProblem, PartitionElement, ElementArray, Solution, empty,
  get_element, objective, evaluate_solution, marginal_gain, compute_weight,
  compute_weight_matrix, mean_weight, total_weight,
  get_element_indices,
  visualize_solution

export Agent, generate_agents,
  generate_colors,
  visualize_agents

export DAGSolver, PartitionSolver,
  sequence, in_neighbors

export solve_optimal, solve_worst, solve_myopic, solve_random, solve_sequential,
  solve_dag, solve_n_partitions

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

# dag solver
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

# Generic partition solver

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

# fixed number of partitions
function solve_n_partitions(num_partitions, p::PartitionProblem)
  num_agents = length(p.partition_matroid)

  partition_numbers = map(x->rand(1:num_partitions), 1:num_agents)

  partition_solver = PartitionSolver(partition_numbers)

  solve_dag(partition_solver, p)
end

# global adaptive number of partitions
# local adaptive number of partitions

end
