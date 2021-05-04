import soft_actor_critic


if __name__ == '__main__':
    environment_id = 'Pendulum-v0'
    environment_kwargs = {}
    soft_actor_critic.train(environment_id, env_kwargs=environment_kwargs, load_models=False)
