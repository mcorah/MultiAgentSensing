using SubmodularMaximization
using PyPlot

function evaluate_pdf(p, name)
  println("Evaluating $name")

  println("PDF at center")
  @show pdf(p, [0.5, 0.5])

  figure()
  visualize_pdf(g)
  title("$name")
end

g = Gaussian([0.5, 0.5], diagm([0.2, 1.0]))
evaluate_pdf(g, "Gaussian")
