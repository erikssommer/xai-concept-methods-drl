import gym

# Create the go environment
env = gym.make('gym_go:go-v0', size=5, komi=0, reward_method='heuristic')

# Reset the environment
env.reset()

# Play a random move
env.step(env.action_space.sample())

# Render the board
env.render()

# Play a random game
done = False
while not done:
    # Get valid moves
    valid_moves = env.valid_moves()
    valid = False

    while not valid:
        action = env.action_space.sample()
        if valid_moves[action] != 0:
            _, _, done, _ = env.step(action)
            valid = True
    
    env.render()

# Close the environment
env.close()