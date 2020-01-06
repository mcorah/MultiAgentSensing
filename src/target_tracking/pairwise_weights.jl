# This file contains necessary code to compute weights according to a sum
# decomposition

export reachable_set, propagate_reachable_set, channel_capacities_by_target,
  channel_capacities_by_target_time


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
  # Note: X is a vector of TimedObservation
  function objective(observations)
    finite_horizon_information(p.grid, filter, p.sensor,
                               observations, step;
                               num_samples=p.objective_information_samples,
                               entropy_only_at_end=true
                              ).reward
  end

  # Due to the objective design, states after the given time-step do not matter
  partition_matroid = reachable_sets(p.grid, p.partition_matroid[robot_index],
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

  function objective(observations)
    finite_horizon_information(p.grid, filter, p.sensor,
                               observations, p.horizon;
                               num_samples=p.objective_information_samples,
                               entropy_only_at_end=false
                              ).reward
  end

  partition_matroid = reachable_sets(p.grid, p.partition_matroid[robot_index],
                                     p.horizon)

  # Construct and solve the optimization problem
  problem = ExplicitPartitionProblem(objective, partition_matroid)
  solution = solve_sequential(problem)

  # See discussion in the target/time implementation
  2 * solution.value
end


#
# Compute all channel capacity bounds (by target and or time-step) for a given
# robot and store in an array
#

function channel_capacities_by_target_time(p::MultiRobotTargetTrackingProblem,
                                           robot_index::Integer)
  map(product(p.target_filters, 1:p.horizon)) do (filter, step)
    channel_capacity_by_target_time(p, filter, step=step,
                                    robot_index=robot_index)
  end
end

function channel_capacities_by_target(p::MultiRobotTargetTrackingProblem,
                                      robot_index::Integer)
  map(p.target_filters) do filter
    channel_capacity_by_target(p, filter,robot_index=robot_index)
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
                               channel_capacity_method)
  n = get_num_agents(p)

  # Compute channel capacities for each robot
  capacities = map(x->channel_capacity_method(p, x), 1:n)

  weights = zeros(n,n)

  for ii in 2:n, jj in 1:ii-1
    w = compute_weight(capacities[ii], capacities[jj])

    weights[ii, jj] = w
    weights[jj, ii] = w
  end

  weights
end
