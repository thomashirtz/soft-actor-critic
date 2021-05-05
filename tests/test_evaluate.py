import soft_actor_critic


if __name__ == '__main__':
    environment_id = 'LunarLanderContinuous-v2'
    name = 'SAC_20210504_152750_LunarLanderContinuous-v2_lr0.0003'
    soft_actor_critic.evaluate(environment_id, run_name=name)


