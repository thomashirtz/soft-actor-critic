import soft_actor_critic


if __name__ == '__main__':
    environment_id = 'LunarLanderContinuous-v2'
    soft_actor_critic.train(environment_id)
