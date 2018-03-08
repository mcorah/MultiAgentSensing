module SubmodularMaximization

export marginal_gain, compute_weight

# f({x} | Y)
marginal_gain(f, x, Y) = f(vcat([x], y)) - f(y)

compute_weight(f, x, y) = f(x) - marginal_gain(f, x, [y])

function compute_weight(f, X::Array, Y::Array) =

  for x in X, y in Y
    compute_weight(f, X, Y)
  end
end

end
