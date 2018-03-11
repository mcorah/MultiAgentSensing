module SubmodularMaximization

using Iterators

import Base.<

export  PartitionProblem, PartitionElement, ElementArray, Solution, empty,
  objective, evaluate_solution, marginal_gain, compute_weight,
  get_element_indices,
  visualize_solution

export solve_optimal

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

marginal_gain(f, x, Y) = f(vcat([x], y)) - f(y)

compute_weight(f, x, y) = f(x) - marginal_gain(f, x, [y])

compute_weight(f, X::Array, Y::Array) =
  maximum([compute_weight(f, x, y) for x in X, y in Y])

# indexing and solver tools

# construct index set to ease manipulation of the matroid
get_element_indices(agents) = map(agents, 1:length(agents)) do agent, agent_index
  map(1:length(get_block(agent))) do block_index
    (agent_index, block_index)
  end
end

include("coverage.jl")

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

# generic visualization
function visualize_solution(p::PartitionProblem, X::ElementArray, agent_colors)
  elements = map(X) do x
    get_element(p.partition_matroid, x)
  end

  colors = [agent_colors[x[1]] for x in X]
  visualize_solution(elements, colors)
end

visualize_solution(p::PartitionProblem, s::Solution, agent_colors) =
  visualize_solution(p, s.elements, agent_colors)

end
