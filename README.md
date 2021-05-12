# soft-actor-critic
<img align="right" width="400"  src="lunar_lander.gif"> 
Implementation of the soft actor critic algorithm using Pytorch. Code kept as lean and clean
as possible on purpose. 

## Usage

### Train

```
python3 soft_actor_critic train
```

With some arguments:
```
python3 soft_actor_critic train --env-name MountainCarContinuous-v0 --learning-rate 0.001
```

### Eval

```
python3 soft_actor_critic eval --run-name name_of_my_last_run
```

With some arguments:
```
python3 soft_actor_critic eval --run-name name_of_my_last_run --hidden-units 512 512 --seed 2
```
(The environment name, and the hidden units need to correspond to the arguments in the run that is loaded)


## Help

```
python soft_actor_critic --help
```
Output:
```
usage: Use "python soft_actor_critic --help" for more information

PyTorch Soft Actor-Critic

positional arguments:
  {train,eval}  Selection of the mode to perform
    train       Train an agent
    eval        Evaluate the performance of an already trained agent

optional arguments:
  -h, --help    show this help message and exit
```

### Train help

```
python soft_actor_critic train --help
```
Output:
```
usage: Use "python soft_actor_critic --help" for more information train [-h] [--env-name] [--hidden-units  [...]]
                                                                        [--directory] [--seed] [--run-name]
                                                                        [--batch-size] [--memory-size]
                                                                        [--learning-rate] [--gamma] [--tau]
                                                                        [--num-steps] [--start-step] [--alpha]

optional arguments:
  -h, --help            show this help message and exit
  --env-name            Gym environment to train on (default: LunarLanderContinuous-v2)
  --hidden-units  [ ...]
                        List of networks' hidden units (default: [256, 256])
  --directory           Root directory in which the run folder will be created (default: ../runs/)
  --seed                Seed used for pytorch, numpy and the environment (default: 1)
  --run-name            Name used for saving the weights and the logs (default: generated using the "get_run_name"
                        function)
  --batch-size          Batch size used by the agent during the learning phase (default: 256)
  --memory-size         Size of the replay buffer (default: 1000000)
  --learning-rate       Learning rate used for the networks and entropy optimization (default: 0.0003)
  --gamma               Discount rate used by the agent (default: 0.99)
  --tau                 Value used for the progressive update of the target networks (default: 0.005)
  --num-steps           Number training steps (default: 1000000)
  --start-step          Step after which the agent starts to learn (default: 1000)
  --alpha               Starting value of the entropy (alpha) (default: 0.2)
```

### Evaluate help

```
python soft_actor_critic eval --help
```
Output:
```
usage: Use "python soft_actor_critic --help" for more information eval [-h] [--env-name] [--hidden-units  [...]]
                                                                       [--directory] [--seed] [--run-name]
                                                                       [--num-episodes] [--deterministic] [--render]
                                                                       [--record]

optional arguments:
  -h, --help            show this help message and exit
  --env-name            Gym environment to train on (default: LunarLanderContinuous-v2)
  --hidden-units  [ ...]
                        List of networks' hidden units (default: [256, 256])
  --directory           Root directory in which the run folder will be created (default: ../runs/)
  --seed                Seed used for pytorch, numpy and the environment (default: 1)
  --run-name            Run name of an already trained agent located in the "--directory" directory
  --num-episodes        Number of episodes to run (default: 3)
  --deterministic       Toggle deterministic behavior of the agent when interacting with the environment
  --render              Toggle the rendering of the episodes
  --record              Toggle the recording of the episodes (toggling "record" would also toggle "render")
```

## Equations

This is an attempt to show the equation of the paper, and their correspondence in the source code:

### Critic Optimization

[Equation 6:](https://arxiv.org/pdf/1812.05905v2.pdf)  

<img src="https://render.githubusercontent.com/render/math?math=%5Chat%7B%5Cnabla%7D_%7B%5Ctheta%7D%20J_%7BQ%7D(%5Ctheta)%3D%5Cnabla_%7B%5Ctheta%7D%20Q_%7B%5Ctheta%7D%5Cleft(%5Cmathbf%7Ba%7D_%7Bt%7D%2C%20%5Cmathbf%7Bs%7D_%7Bt%7D%5Cright)%5Cleft(Q_%7B%5Ctheta%7D%5Cleft(%5Cmathbf%7Bs%7D_%7Bt%7D%2C%20%5Cmathbf%7Ba%7D_%7Bt%7D%5Cright)-%5Cleft(r%5Cleft(%5Cmathbf%7Bs%7D_%7Bt%7D%2C%20%5Cmathbf%7Ba%7D_%7Bt%7D%5Cright)%2B%5Cgamma%5Cleft(Q_%7B%5Cbar%7B%5Ctheta%7D%7D%5Cleft(%5Cmathbf%7Bs%7D_%7Bt%2B1%7D%2C%20%5Cmathbf%7Ba%7D_%7Bt%2B1%7D%5Cright)-%5Calpha%20%5Clog%20%5Cleft(%5Cpi_%7B%5Cphi%7D%5Cleft(%5Cmathbf%7Ba%7D_%7Bt%2B1%7D%20%5Cmid%20%5Cmathbf%7Bs%7D_%7Bt%2B1%7D%5Cright)%5Cright)%5Cright)%5Cright)%5Cright.">

```python
def _critic_optimization(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
                         next_state: torch.Tensor, done: torch.Tensor) -> Tuple[float, float]:
    with torch.no_grad():
        next_action, next_log_pi = self.policy.evaluate(next_state)
        next_q_target_1, next_q_target_2 = self.target_critic.forward(next_state, next_action)
        min_next_q_target = torch.min(next_q_target_1, next_q_target_2)
        next_q = reward + (1 - done) * self.gamma * (min_next_q_target - self.alpha * next_log_pi)

    q_1, q_2 = self.critic.forward(state, action)
    q_network_1_loss = F.mse_loss(q_1, next_q)
    q_network_2_loss = F.mse_loss(q_2, next_q)
    q_loss = (q_network_1_loss + q_network_2_loss) / 2

    self.critic_optimizer.zero_grad()
    q_loss.backward()
    self.critic_optimizer.step()
    return q_network_1_loss.item(), q_network_2_loss.item()

```

### Policy Optimization 

[Equation 10:](https://arxiv.org/pdf/1812.05905v2.pdf)  

<img src="https://render.githubusercontent.com/render/math?math=%5Chat%7B%5Cnabla%7D_%7B%5Cphi%7D%20J_%7B%5Cpi%7D(%5Cphi)%3D%5Cnabla_%7B%5Cphi%7D%20%5Calpha%20%5Clog%20%5Cleft(%5Cpi_%7B%5Cphi%7D%5Cleft(%5Cmathbf%7Ba%7D_%7Bt%7D%20%5Cmid%20%5Cmathbf%7Bs%7D_%7Bt%7D%5Cright)%5Cright)%2B%5Cleft(%5Cnabla_%7B%5Cmathbf%7Ba%7D_%7Bt%7D%7D%20%5Calpha%20%5Clog%20%5Cleft(%5Cpi_%7B%5Cphi%7D%5Cleft(%5Cmathbf%7Ba%7D_%7Bt%7D%20%5Cmid%20%5Cmathbf%7Bs%7D_%7Bt%7D%5Cright)%5Cright)-%5Cnabla_%7B%5Cmathbf%7Ba%7D_%7Bt%7D%7D%20Q%5Cleft(%5Cmathbf%7Bs%7D_%7Bt%7D%2C%20%5Cmathbf%7Ba%7D_%7Bt%7D%5Cright)%5Cright)%20%5Cnabla_%7B%5Cphi%7D%20f_%7B%5Cphi%7D%5Cleft(%5Cepsilon_%7Bt%7D%20%3B%20%5Cmathbf%7Bs%7D_%7Bt%7D%5Cright)">

```python
def _policy_optimization(self, state: torch.Tensor) -> float:
    with eval_mode(self.critic):
        predicted_action, log_probabilities = self.policy.evaluate(state)
        q_1, q_2 = self.critic(state, predicted_action)
        min_q = torch.min(q_1, q_2)

        policy_loss = ((self.alpha * log_probabilities) - min_q).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        return policy_loss.item()

```

### Entropy Optimization 

[Equation 17:](https://arxiv.org/pdf/1812.05905v2.pdf)  

<img src="https://render.githubusercontent.com/render/math?math=%5Calpha_%7Bt%7D%5E%7B*%7D%3D%5Carg%20%5Cmin%20_%7B%5Calpha_%7Bt%7D%7D%20%5Cmathbb%7BE%7D_%7B%5Cmathbf%7Ba%7D_%7Bt%7D%20%5Csim%20%5Cpi_%7Bt%7D%5E%7B*%7D%7D%5Cleft%5B-%5Calpha_%7Bt%7D%20%5Clog%20%5Cpi_%7Bt%7D%5E%7B*%7D%5Cleft(%5Cmathbf%7Ba%7D_%7Bt%7D%20%5Cmid%20%5Cmathbf%7Bs%7D_%7Bt%7D%20%3B%20%5Calpha_%7Bt%7D%5Cright)-%5Calpha_%7Bt%7D%20%5Coverline%7B%5Cmathcal%7BH%7D%7D%5Cright%5D">

```python
def _entropy_optimization(self, state: torch.Tensor) -> float:
    with eval_mode(self.policy):
        _, log_pi = self.policy.evaluate(state)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()
        return alpha_loss.item()

```

## Resources

List of repository that helped me to solve technical issues:
- [pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic) 
- [Soft-Actor-Critic-and-Extensions](https://github.com/BY571/Soft-Actor-Critic-and-Extensions) 
- [spinningup](https://github.com/openai/spinningup) 

Equations made with [this](https://jsfiddle.net/8ndx694g/) tool, taken from [this](https://gist.github.com/VictorNS69/1c952045825eac1b5e4d9fc84ad9d384) thread.

## License
```
Copyright 2021 Thomas Hirtz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Original papers:

Soft Actor-Critic Algorithms and Applications, Haarnoja et al. [[arxiv]](https://arxiv.org/abs/1812.05905v2) (January 2019)   
Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, Haarnoja et al. [[arxiv]](https://arxiv.org/abs/1801.01290) (January 2018)
