import soft_actor_critic


if __name__ == '__main__':
    environment_id = 'LunarLanderContinuous-v2'
    name = 'name_of_a_previous_run'
    soft_actor_critic.evaluate(environment_id, run_name=name)
