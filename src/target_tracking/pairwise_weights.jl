# This file contains necessary code to compute weights according to a sum
# decomposition

export reachable_set, propagate_reachable_set, channel_capacities_by_target,
  channel_capacities_by_target_time, channel_capacities_mcts


# Compute reachable set on the grid for n steps as a set of states
function reachable_set(grid::Grid, state::State, steps::Integer)
  states = Set([state])

  for ii = 1:steps
    states = propagate_reachable_set(grid, states)
  end

  states
end
# Helper function for computation of reachable sets
function propagate_reachable_set(grid::Grid, states)
  union(map(x->Set(neighbors(grid, x)), collect(states))...)
end

# Computes reachable sets for each step in a horizon and returns the result in a
# named tuple to avoid ambiguity with the form
# (step=<step>, states=Set{State})
struct ReachableAtTime
  step::Int64
  states::Vector{State}
end
function reachable_sets(grid::Grid, state::State, horizon::Integer)
  sets = Vector{ReachableAtTime}(undef, horizon)

  for step in 1:horizon
    local states

    if step == 1
      states = propagate_reachable_set(grid, Set([state]))
    else
      states = propagate_reachable_set(grid, sets[step-1].states)
    end

    sets[step] = ReachableAtTime(step, collect(states))
  end

  sets
end

# We will define a partition matroid for an array of ReachableAtTime
#
# To do so, we provide access to an iterable object that represents the
# collection of set actions in a given block of the partition matroid
get_block(x::ReachableAtTime) = x.states
# get element will determine what goes into the objective
function get_element(partition_matroid::Vector{ReachableAtTime}, x)
  set = partition_matroid[x[1]]

  TimedObservation(set.step, set.states[x[2]])
end

#
# Channel capacity bounds
#
# The use of the sum decomposition of the objective relies on computation of
# upper-bounds of the individual summands. In the case of mutual information
# objectives, these bounds are channel capacities
#

#
# Compute an individual channel capacity bound between a robot and a target,
# optionally at a specific time-step
#

function channel_capacity_by_target_time(p::MultiRobotTargetTrackingProblem,
                                         filter::Filter;
                                         step, robot_index)
  configs = p.configs

  # Note: observations is a vector of TimedObservation
  function objective(observations)
    finite_horizon_information(configs.grid, filter, configs.sensor,
                               observations, step;
                               num_samples=
                                configs.objective_information_samples,
                               entropy_only_at_end=true
                              ).reward
  end

  # Due to the objective design, states after the given time-step do not matter
  partition_matroid = reachable_sets(configs.grid,
                                     p.partition_matroid[robot_index],
                                     step)

  # Construct and solve the optimization problem
  problem = ExplicitPartitionProblem(objective, partition_matroid)
  solution = solve_sequential(problem)

  # Assuming that the sampling of the objective is adequate
  # opt <= 2 * (solution.value + epsilon)
  # and epsilon ~= 0
  #
  # Hence we treat the following as an upper bound on the weight.
  #
  # Note: Other, tighter bounds are possible. These include:
  #   * Online bounds based on submodularity (Sum of greatest marginals in each
  #     block)
  #   * Possibly some channel capacity bounds, but that is a different story
  2 * solution.value
end

function channel_capacity_by_target(p::MultiRobotTargetTrackingProblem,
                                    filter::Filter; robot_index)
  configs = p.configs

  function objective(observations)
    finite_horizon_information(configs.grid, filter, configs.sensor,
                               observations, configs.horizon;
                               num_samples=
                                configs.objective_information_samples,
                               entropy_only_at_end=false
                              ).reward
  end

  partition_matroid = reachable_sets(configs.grid,
                                     p.partition_matroid[robot_index],
                                     configs.horizon)

  # Construct and solve the optimization problem
  problem = ExplicitPartitionProblem(objective, partition_matroid)
  solution = solve_sequential(problem)

  # See discussion in the target/time implementation
  2 * solution.value
end

# Partition matroids turned out to be super slow as I probably could have
# guessed. Using MCTS is a bit tenuous, but I should be able to get results in a
# small fraction of the time I would require otherwise
#
# On the plus side, this should be a relatively easy problem for MCTS to solve
function channel_capacity_mcts(p::MultiRobotTargetTrackingProblem,
                               filter::AnyFilter; robot_index)
  configs=p.configs

  # Construct a solver for the individual filter
  problem = SingleRobotTargetTrackingProblem(configs.grid, configs.sensor,
                                           configs.horizon,
                                           [filter],
                                           num_information_samples=
                                             configs.solver_information_samples)
  function objective(observations)
    finite_horizon_information(configs.grid, filter, configs.sensor,
                               observations, configs.horizon;
                               num_samples=
                                configs.objective_information_samples,
                               entropy_only_at_end=false
                              ).reward
  end

  state = p.partition_matroid[robot_index]

  trajectory = solve_single_robot(problem, state,
                                  n_iterations=
                                    configs.solver_iterations).trajectory

  # Sample again to get an accurate estimate of the reward
  #
  # The tuple is a hack to put this in a format tha the multi-robot code can
  # recognize
  objective([trajectory])
end

#
# Compute all channel capacity bounds (by target and or time-step) for a given
# robot and store in an array
#

function channel_capacities_by_target_time(p::MultiRobotTargetTrackingProblem,
                                           robot_index::Integer,
                                           target_in_range =
                                           (robot_index, target_index) -> true
                                          )
  configs = p.configs

  map(product(enumerate(p.target_filters), 1:configs.horizon)) do
    ((target_index, filter), step)

    if target_in_range(robot_index, target_index)
      channel_capacity_by_target_time(p, filter, step=step,
                                      robot_index=robot_index)
    else
      0.0
    end
  end
end

function channel_capacities_by_target(p::MultiRobotTargetTrackingProblem,
                                      robot_index::Integer,
                                      target_in_range = (robot_index,
                                                         target_index) -> true
                                     )
  map(enumerate(p.target_filters)) do (target_index, filter)
    if target_in_range(robot_index, target_index)
      channel_capacity_by_target(p, filter,robot_index=robot_index)
    else
      0.0
    end
  end
end

function channel_capacities_mcts(p::MultiRobotTargetTrackingProblem,
                                 robot_index::Integer,
                                 target_in_range = (robot_index,
                                                    target_index) -> true
                                )
  map(enumerate(p.target_filters)) do (target_index, filter)
    if target_in_range(robot_index, target_index)
      channel_capacity_mcts(p, filter, robot_index=robot_index)
    else
      0.0
    end
  end
end

# Compute a weight based on channel capacities from the sum decomposition
#
# Note: here the weight corresponds to the maximum reduction of one robot's
# marginal gain given the other
#
# Here, we take advantage of the fact that the marginal gain with respect to a
# target must always be positive. As such, marginal can be reduced by at most
# the lesser channel capacity. Summing produces a bound over all components of
# the decomposition
compute_weight(a::Array{Float64}, b::Array{Float64}) = sum(min.(a, b))

function compute_weight_matrix(p::MultiRobotTargetTrackingProblem;
                               channel_capacity_method=channel_capacities_mcts,
                               robot_target_range_limit=Inf,
                               threaded = false
                              )
  local capacities

  n = get_num_agents(p)

  function range_test(robot_index, target_index)
    target_mean_distance(p, robot_index,
                         target_index) < robot_target_range_limit
  end

  # Compute channel capacities for each robot
  channel_capacity(x) = channel_capacity_method(p, x, range_test)

  if threaded
    capacities = thread_map(channel_capacity, 1:n, Vector{Float64})
  else
    capacities = map(channel_capacity, 1:n)
  end

  weights = zeros(n,n)

  for ii in 2:n, jj in 1:ii-1
    w = compute_weight(capacities[ii], capacities[jj])

    weights[ii, jj] = w
    weights[jj, ii] = w
  end

  weights
end
