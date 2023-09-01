import gym

# Create the go environment
env = gym.make('gym_go:go-v0', size=2, komi=0, reward_method='real')

# Reset the environment
env.reset()

print(env.canonical_state())

# Play a random game
terminated = False
while not terminated:
    # Get valid moves
    valid_moves = env.valid_moves()
    valid = False

    while not valid:
        action = env.action_space.sample()
        if valid_moves[action] != 0:
            observation, reward, terminated, info = env.step(action)

            # Render the board
            env.render()

            # Print the results
            print("Observation: \n {}".format(observation))
            print("Info: {}".format(info))
            print("Reward: {}, Terminated: {}".format(reward, terminated))

            # Switch valid to True
            valid = True

# Close the environment
env.close()