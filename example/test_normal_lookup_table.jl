using SubmodularMaximization
using Distributions
using Statistics

table = SubmodularMaximization.NormalLookup(increment=0.0001, max=4.0)

maxerror = maximum(0.0:0.00001:5) do x
  trueval = pdf(Normal(0.0, 1.0), -x)
  lookupval = SubmodularMaximization.evaluate(table, error= -x, stddev=1.0)

  abs(trueval - lookupval)
end
println("Max error: ", maxerror)

let sigma = 3.0, error = 5.0
  @show trueval = pdf(Normal(0.0, sigma), error)
  @show lookupval = SubmodularMaximization.evaluate(table, error=error, stddev=sigma)
  @show abs(trueval - lookupval)
end
