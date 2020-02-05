# Single robot solver using Monte Carlo tree search

using LinearAlgebra

using MCTS
using POMDPs
using Random
import POMDPs.isterminal
import POMDPs.actions

export SingleRobotTargetTrackingProblem, MDPState, generate_solver,
  solve_single_robot, isterminal

const default_solver_information_samples = 1

const default_horizon = 2

# This approximates the (half) the expected reward as prior tests indicated
# that MCTS performs best in this regime
#
# The exact values were tuned based on the default configuration (number of
# targets, sensor noise) for 4 and 8 robots and consideration of typical
# objective values per robot as obtained from the multi-robot example script
exploration_constant(horizon) = 0.18 * horizon^2 + 0.31

# We map the horizon length to the number of samples according to approximately
# where the average return crossed 97% in a prior test
#
# Note: the last test indicated slow convergence for a horizon length of 2.
const default_num_iterations = Dict(1=>512,
                                    2=>512,
                                    3=>1024,
                                    4=>1024,
                                    5=>2048,
                                    6=>4000)

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
  depth::Int64
  prev::Union{MDPState, Nothing}
end
# Constructor for initial states
MDPState(state) = MDPState(state, 0, nothing)
# Construct a new state in a trajectory
MDPState(m::MDPState, s::State) = MDPState(s, m.depth + 1, m)

# The target tracking problem is an mdp with MDPStates for states
# And target tracking States for actions
struct SingleRobotTargetTrackingProblem{F<:AnyFilter} <: MDP{MDPState, State}
  grid::Grid
  sensor::RangingSensor
  horizon::Int64
  target_filters::Vector{F}
  prior_trajectories::Vector{Vector{State}}

  num_information_samples::Int64

  rng::MersenneTwister

  function SingleRobotTargetTrackingProblem(grid::Grid, sensor::RangingSensor,
                                            horizon::Integer,
                                            filters::Vector{T};
                                            prior_trajectories = Trajectory[],
                                            num_information_samples =
                                              default_solver_information_samples,
                                            rng = Random.GLOBAL_RNG
                                           ) where T <: AnyFilter
    new{T}(grid, sensor, horizon, filters, prior_trajectories,
        num_information_samples, rng)
  end
end

function generate_solver(depth;
                         n_iterations = default_num_iterations[depth],
                         exploration_constant = exploration_constant(depth),
                         enable_tree_vis = false)

  solver = MCTSSolver(n_iterations = n_iterations,
                      depth = depth,
                      exploration_constant = exploration_constant,
                      enable_tree_vis = enable_tree_vis
                     )
end

# Return the next state for the robot
function solve_single_robot(problem::SingleRobotTargetTrackingProblem,
                            state::State;
                            n_iterations =
                              default_num_iterations[problem.horizon],
                            exploration_constant =
                              exploration_constant(problem.horizon))

  solver = generate_solver(horizon(problem), n_iterations = n_iterations,
                           exploration_constant = exploration_constant)
  policy = solve(solver, problem)

  action, info = action_info(policy, MDPState(state))
  tree = info[:tree]
  trajectory = extract_trajectory(problem, tree, state)

  (
   action=action,
   trajectory=trajectory,
   tree=tree
  )
end

# Pull the tree out of the info object
function extract_trajectory(problem::SingleRobotTargetTrackingProblem,
                            tree::MCTS.MCTSTree,
                            initial_state::State)

  # Recal that our State objects are the actions in the MDP
  states = Vector{State}(undef, horizon(problem))

  mdp_state = MDPState(initial_state)

  # Allow for cases where the tree might not all be stored by sampling the rest
  # of the rollout
  for ii in 1:horizon(problem)
    state = try
      action(MCTS.best_sanode_Q(MCTS.StateNode(tree, mdp_state)))
    catch e
      target_dynamics(problem.grid, mdp_state.state; rng=problem.rng)
    end

    states[ii] = state

    # Propagate the trajectory
    mdp_state = MDPState(mdp_state, state)
  end

  states
end

horizon(x::SingleRobotTargetTrackingProblem) = x.horizon

# I probably dont need to define this, but it's good to have
function isterminal(model::SingleRobotTargetTrackingProblem, state)
  state.depth == model.horizon
end

function actions(model::SingleRobotTargetTrackingProblem, state::MDPState)
  neighbors(model.grid, state.state)
end
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
                       trajectory::Trajectory; kwargs...)

  # Compute reward conditional on prior selections (trajectories)
  sum(model.target_filters) do filter
    finite_horizon_information(model.grid, filter, model.sensor,
                               trajectory;
                               prior_observations=model.prior_trajectories,
                               num_samples=model.num_information_samples,
                               kwargs...).reward
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

  if isnan(reward)
    error("Information reward (", reward, ") is NaN")
  end

  (sp=new_state , r=reward)
end
