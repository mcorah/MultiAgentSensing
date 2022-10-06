# run simple tests on the filter and updates

using SubmodularMaximization
using LinearAlgebra

grid_size = 2
grid = Grid(grid_size, grid_size)

robot_state = div.((grid_size, grid_size), 2)
sensor = RangingSensor()

grid_states = get_states(grid)
histogram_filter = Filter(grid)

@show prior = HistogramFilters.get_data(histogram_filter)

if !all(prior .== 1/grid_size^2)
  println("Problem with uniform prior")
end

# Test neighbors
@show ns = neighbors(grid, (1,1))
real_ns = Set([(1,1), (1,2), (2,1)])
if real_ns != Set(ns)
  println("Neighbors do not match")
end

@show transition = Matrix(transition_matrix(grid))

# true 4x4 transition matrix for a 2x2 grid noting that the indices are
# [1, 3]
# [2, 4]
real_transition = vcat([1/3  1/3  1/3  0  ],
                       [1/3  1/3    0  1/3],
                       [1/3    0  1/3  1/3],
                       [  0  1/3  1/3  1/3])

if transition != real_transition
  println("Problem with transition matrix")
end
