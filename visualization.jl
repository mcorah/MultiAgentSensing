using PyPlot
using Colors

###############
# visualization
###############
rgb_tuple(color::RGB) = (red(color), green(color), blue(color))

generate_colors{T <: Agent}(agents::Array{T,1}) =
  distinguishable_colors(length(agents) + 1)[2:end]

function visualize_agents{T <: Agent}(agents::Array{T,1}, colors)
  centers = map(get_center, agents);

  map(agents, colors) do agent, color
    center = get_center(agent)

    scatter([center[1]], [center[2]], color = rgb_tuple(color), marker = (5, 2, 0))

    map(get_block(agent)) do element
      plot_element(element, color = rgb_tuple(color))
    end
  end

  Void
end

function visualize_solution(p::PartitionProblem, X::ElementArray, agent_colors)
  elements = map(X) do x
    get_element(p.partition_matroid, x)
  end

  colors = [agent_colors[x[1]] for x in X]
  visualize_solution(elements, colors)
end

visualize_solution(p::PartitionProblem, s::Solution, agent_colors) =
  visualize_solution(p, s.elements, agent_colors)
