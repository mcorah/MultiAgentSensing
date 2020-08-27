# This file provides versions of solvers based on communication graphs

using Statistics

function make_adjacency_matrix(problem::ExplicitPartitionProblem, range)
  num_agents = get_num_agents(problem)

  adjacency = zeros(num_agents, num_agents)

  for ii in 1:num_agents, jj in 1:ii-1
    center_ii = get_center(get_agent(problem, ii))
    center_jj = get_center(get_agent(problem, jj))

    if norm(center_ii - center_jj) <= range
      adjacency[ii,jj] = 1
      adjacency[jj,ii] = 1
    end
  end

  adjacency
end

function plot_adjacency(problem::ExplicitPartitionProblem, range::Real)
  adjacency = make_adjacency_matrix(problem, range)

  plot_adjacency(problem, adjacency)
end
function plot_adjacency(problem::ExplicitPartitionProblem, adjacency::Array)
  num_agents = get_num_agents(problem)

  for ii in 1:num_agents, jj in 1:ii-1
    @show center_ii = get_center(get_agent(problem, ii))
    @show center_jj = get_center(get_agent(problem, jj))

    if adjacency[ii,jj] > 0
      x = map(first, [center_ii, center_jj])
      y = map(last, [center_ii, center_jj])
      plot(x, y, color="black")
    end
  end
end
