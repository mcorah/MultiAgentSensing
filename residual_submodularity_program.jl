using Convex
using SCS

# for submodular functions of the form
# f = g(|X|)

a = Variable()
b = Variable()
c = Variable()

g = -3*a + 3*b - c

monotonicity = [a >= 0; b >= a; c >= b]
submodularity = [b - a <= a; c - b <= b - a]
scaling = [a == 1]

println("Maximized redundancy")
problem = maximize(g, [monotonicity; submodularity; scaling])
solver = SCSSolver(verbose = 0)
solve!(problem, solver)

println("Opt=$(problem.optval)")
println("a=$(evaluate(a))")
println("b=$(evaluate(b))")
println("c=$(evaluate(c))")

println()
println("Minimized redundancy")
problem = minimize(g, [monotonicity; submodularity; scaling])
solver = SCSSolver(verbose = 0)
solve!(problem, solver)
println("Opt=$(problem.optval)")
println("a=$(evaluate(a))")
println("b=$(evaluate(b))")
println("c=$(evaluate(c))")
