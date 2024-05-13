# PPO-swarm
[SKRL implementation of MAPPO](https://skrl.readthedocs.io/en/latest/api/multi_agents/mappo.html) applied to [MPE Simple Spread](https://pettingzoo.farama.org/environments/mpe/simple_spread/)

Code runs, I think agents are having a hard time learning either because 1 of a bajillion hyperparameters aren't set appropriately, or not enough runtime, or because the value network shouldn't be gaussian, or because the agents aren't given the information to figure out which one of the NUM_AGENTS they are individually.
