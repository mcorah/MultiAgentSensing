# Some simple tests to verify the sparse filtering code

using SubmodularMaximization
using HistogramFilters
using SparseArrays

g = Grid(10, 10)

thresh = 1e-3

sensor = RangingSensor()

robot_state = (2,2)
target_state = (5,5)

target_filter = SparseFilter(g, target_state, threshold=thresh)

@assert nnz(get_values(target_filter)) == 1

@assert get_states(target_filter) == [target_state]

process_update!(target_filter, transition_matrix(g))

@assert nnz(get_values(target_filter)) == 5
@assert length(get_states(g, target_filter)) == 5


range_observation = generate_observation(g, sensor, robot_state,
                                         target_state)
measurement_update!(target_filter, robot_state,
                    get_states(g, target_filter), sensor, g,
                    range_observation)

@assert nnz(get_values(target_filter)) == 5
@assert length(get_states(g, target_filter)) == 5
