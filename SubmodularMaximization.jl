module SubmodularMaximization

export marginal_gain, compute_weight

# f({x} | Y)
type PartitionProblem
  objective
  partion_matroid::Array
end

marginal_gain(f, x, Y) = f(vcat([x], y)) - f(y)

compute_weight(f, x, y) = f(x) - marginal_gain(f, x, [y])

compute_weight(f, X::Array, Y::Array) =
  maximum([compute_weight(f, x, y) for x in X, y in Y])

include("coverage.jl")

# solvers

# sequential solver

# dag solver

# optimal solver

# anti-optimal solver

end
