using SubmodularMaximization
using PyPlot

close()

num_robots = 4
steps = 10
horizon = SubmodularMaximization.default_horizon

configs = MultiRobotTargetTrackingConfigs(num_robots,
                                          horizon=horizon)

data = target_tracking_experiment(steps=steps, num_robots=num_robots,
                                  configs=configs)

target_states = cat(map(x->x.target_states, data)..., dims=2)
robot_states = cat(map(x->x.robot_states, data)..., dims=2)

plot_state_space(configs.grid)
xlim([0, configs.grid.width+1])
ylim([0, configs.grid.height+1])

for ii = 1:size(robot_states, 1)
  let robot_states = robot_states[ii,:]
    if !continuous(configs.grid, robot_states)
      @show robot_states
      error("Robot states are not continuous")
    end

    plot_trajectory(robot_states, color=:blue)
  end
end

for ii = 1:size(target_states, 1)
  let target_states = target_states[ii,:]
    if !continuous(configs.grid, target_states)
      @show target_states
      error("Target states are not continuous")
    end

    plot_trajectory(target_states)
  end
end
