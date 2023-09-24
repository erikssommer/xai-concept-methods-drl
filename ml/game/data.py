import gym

# Set the logging level
gym.logger.set_level(40)

go_env = gym.make('gym_go:go-v0', size=0)
GoVars = go_env.govars
GoGame = go_env.gogame