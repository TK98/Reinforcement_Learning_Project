def test_episodes(env, policy, num_episodes):
    # policy = policy(network, 0.0)
    policy.test()
    network = policy.network

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []
    episode_rewards = []
    for i in range(num_episodes):
        network.start_episode(env, policy)

        steps = 0
        rewards = 0
        while True:
            experience, done = network.step_episode(env, policy)

            rewards += experience[2]
            global_steps += 1
            steps += 1

            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps obtaining {3} reward"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m', rewards))

                episode_durations.append(steps)
                episode_rewards.append(rewards)

                break

    return episode_durations, episode_rewards