# Some simple tests to verify the sparse filtering code

using SubmodularMaximization

g = Grid(10, 10)

thresh = 1e-3

sensor = RangingSensor()

target_state = (5,5)

histogram_filter = SparseFilter(g, target_state, threshold=thresh)
