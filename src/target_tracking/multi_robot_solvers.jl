using LinearAlgebra

abstract type AbstractTargetProblem <: PartitionProblem{Tuple{Int64,
                                                              Trajectory}}
end

function get_state(p::AbstractTargetProblem, index)
  p.partition_matroid[index]
end

# Return agent center (which is just the state)
# The original coverage code involved more complex agent representations.
get_center(x::Tuple{Int64,Int64}) = [x...]

function sample_block(p::AbstractTargetProblem, block::Integer)
  horizon = p.configs.horizon
  grid = p.configs.grid

  trajectory = Array{State}(undef, horizon)

  # (target dynamics are the same as random rollouts for tracking robots)
  current_state = p.partition_matroid[block]
  for ii in 1:horizon
    current_state = target_dynamics(grid, current_state)
    trajectory[ii] = current_state
  end

  (block, trajectory)
end
