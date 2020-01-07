using SubmodularMaximization
using POMDPs
using MCTS

using Statistics
using Profile

default_steps = 100
grid_size = 10
horizon = 5
iterations = 1000
information_samples = 1

function run_test(steps; print=true)
  ifprint(xs...) = if print println(xs...) end

  ifprint("Running test: ", steps, " steps")
  grid = Grid(grid_size, grid_size)
  sensor = RangingSensor(0.5^2, 0.1^2)

  robot_state = random_state(grid)
  target_state = random_state(grid)

  target_states = Array{State}(undef, steps)
  target_states[1] = target_state

  robot_states = Array{State}(undef, steps)
  robot_states[1] = robot_state

  histogram_filter = Filter(grid, target_state)

  problem = SingleRobotTargetTrackingProblem(grid, sensor, horizon,
                                             [histogram_filter],
                                             num_information_samples =
                                               information_samples)

  time = @elapsed for ii = 2:steps
    # Before the target moves and the robot receives a measurement, execute robot
    # dynamics
    robot_state = solve_single_robot(problem, robot_state,
                                     n_iterations = iterations).action
    robot_states[ii] = robot_state

    # Then update the target and sample the observation
    target_state = target_dynamics(grid, target_states[ii-1])
    target_states[ii] = target_state

    range_observation = generate_observation(grid, sensor, robot_state,
                                             target_state)

    # After updating states, update the filter (in place for now)
    process_update!(histogram_filter, transition_matrix(grid))
    measurement_update!(histogram_filter, robot_state, get_states(grid), sensor,
                        grid, range_observation)
  end

  ifprint("Test duration: ", time, " seconds")
  ifprint("  ", time / (steps - 1), " seconds per step")
  ifprint("  ", time / ((steps - 1) * iterations), " seconds per MCTS sample")
  ifprint("  ", time / ((steps - 1) * iterations * information_samples),
          " seconds per information sample")
  ifprint("Note: horizon length is ", horizon)
end

# Show timing without profiling
run_test(2)
run_test(default_steps)

# Run the profiler
Profile.init(n=1000000)
@profile run_test(2, print=false)

Profile.clear()
Profile.init(n=1000000)
@profile run_test(default_steps, print=false)

println()
Profile.print(mincount=100)

nothing
