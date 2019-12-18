# Some simple tests on the entropy/information code

using SubmodularMaximization

grid_size = 10

grid = Grid(grid_size, grid_size)

sensor = RangingSensor(0.5^2, 0.1^2)
histogram_filter = Filter(grid)

# Test entropy on uniform prior
@show grid_entropy = entropy(histogram_filter)
real_entropy = -log(2, 1/num_states(grid))
if !isapprox(grid_entropy, real_entropy; rtol=1e-3)
  println("Incorrect entropy for uniform prior")
end
