# Plots an random example of a prior and the resulting information distribution

using SubmodularMaximization
using Profile

steps = 10000
grid_size = 10
num_observations = 3
horizon = 5
num_information_samples = 10

grid = Grid(grid_size, grid_size)
sensor = RangingSensor(0.5^2, 0.1^2)

target_state = (2,2)

# precomputation
histogram_filter = Filter(grid, target_state)

# Create some fake observations and update the filter
for ii = 1:num_observations
  robot_state = random_state(grid)
  range_observation = generate_observation(sensor, robot_state, target_state)
  measurement_update!(histogram_filter, robot_state, get_states(grid), sensor,
                      range_observation)
end

function run_test(steps; print=true)
  ifprint(xs...) = if print println(xs...) end

  time = @elapsed for ii = 1:steps
    robot_state = random_state(grid)
    trajectory = fill(robot_state, horizon)

    finite_horizon_information(grid, histogram_filter, sensor,
                               trajectory,
                               num_samples = num_information_samples).reward
  end

  ifprint("Test duration: ", time, " seconds")
  ifprint("  ", time / steps, " seconds per information estimate")
  ifprint("  ", time / (steps * num_information_samples),
          " per information sample")
  ifprint("Note: horizon length is ", horizon)
end

run_test(1)
run_test(steps)

# Run the profiler
Profile.init(n=1000000)
@profile run_test(1, print=false)

Profile.clear()
Profile.init(n=1000000)
@profile run_test(steps, print=false)

println()
Profile.print(mincount=100)
