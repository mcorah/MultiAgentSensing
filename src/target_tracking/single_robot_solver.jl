# Single robot solver using Monte Carlo tree search

using LinearAlgebra

using MCTS
using POMDPs
import POMDPs.isterminal
import POMDPs.actions

export SingleRobotTargetTrackingProblem, MDPState, generate_solver, isterminal

const default_num_iterations = 1000
const default_exploration_constant = 10.0

#
# Define the MDP model for POMDPs.jl
#

# Simple struct for the state
# (defining the state trajectory in place probably ends up being faster than
# not)
#
# We will use the usual 0 indexing for depth (i.e. depth represents a number of
# actions taken)
struct MDPState
  state::State
  depth::Integer
  prev::Union{MDPState, Nothing}
end
# Constructor for initial states
MDPState(state) = MDPState(state, 0, nothing)
# Construct a new state in a trajectory
MDPState(m::MDPState, s::State) = MDPState(s, m.depth + 1, m)

# The target tracking problem is an mdp with MDPStates for states
# And target tracking States for actions
struct SingleRobotTargetTrackingProblem <: MDP{MDPState, State}
  grid::Grid
  sensor::RangingSensor
  horizon::Integer
  target_filters::Vector{Filter{Int64}}
  trajectories::Vector{Vector{State}}
end
function SingleRobotTargetTrackingProblem(grid::Grid, sensor::RangingSensor,
                                          horizon::Integer,
                                          filters::Vector{Filter{Int64}})
  SingleRobotTargetTrackingProblem(grid, sensor, horizon, filters,
                                   Vector{State}[])
end

function generate_solver(depth;
                         n_iterations = default_num_iterations,
                         exploration_constant = default_exploration_constant)

  solver = MCTSSolver(n_iterations = n_iterations,
                      depth = depth,
                      exploration_constant = exploration_constant,
                      enable_tree_vis = false
                     )
end

horizon(x::SingleRobotTargetTrackingProblem) = x.horizon

# I probably dont need to define this, but it's good to have
function isterminal(model::SingleRobotTargetTrackingProblem, state)
  state.depth == model.horizon
end

function actions(model::SingleRobotTargetTrackingProblem, state::MDPState)
  neighbors(model.grid, state.state)
end
# I hope that the set of valid actions is not constant
function actions(model::SingleRobotTargetTrackingProblem)
  println("providing all actions")
  get_states(model.grid)
end
POMDPs.discount(model::SingleRobotTargetTrackingProblem) = 1.0

# This tree implementation should be fine since everything should already be in
# cache once we reach the end.
# The alternative probably involves messy equality operators
#
# Note that the trajectory will *not* include the initial position
function state_trajectory(state::MDPState)
  ret = Array{State}(undef, state.depth)

  ret[1] = state.state

  curr_state = state
  for ii = 2:state.depth
    curr_state = curr_state.prev
    ret[ii] = curr_state.state
  end

  ret
end

# sample reward for all targets and sum
function sample_reward(model::SingleRobotTargetTrackingProblem,
                       trajectory::Vector{State}; kwargs...)
  if !isempty(model.trajectories)
    println(stderr, "Planning for multiple robots has not yet been implemented")
  end

  trajectories = vcat(model.trajectories, [trajectory])
  sum(model.target_filters) do filter
    finite_horizon_information(model.grid, filter, model.sensor,
                               trajectories; kwargs...).reward
  end
end

# Generative model for the single-robot MDP
#
# We will sample the reward once we reach the end of the trajectory
function POMDPs.gen(model::SingleRobotTargetTrackingProblem,
                    s::MDPState, a::State, rng)
  new_state = MDPState(s, a)
  reward = 0.0

  if isterminal(model, new_state)
    reward = sample_reward(model, state_trajectory(new_state); rng = rng)
  end

  (sp=new_state , r=reward)
end
