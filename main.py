from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.utils.model_instantiators.torch import deterministic_model
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint


from pettingzoo.mpe import simple_spread_v3

env = simple_spread_v3.parallel_env(
    render_mode="human", N=5, local_ratio=0.5, max_cycles=25, continuous_actions=True
)

env = wrap_env(env)
env.reset()

device = torch.device("mps")


# define the policy model

# instantiate the agent's models
models = {}
for agent_name in env.possible_agents:
    # Then, when defining each model:
    models[agent_name] = {}
    models[agent_name]["policy"] = deterministic_model(
        observation_space=env.observation_spaces[agent_name],
        action_space=env.action_spaces[agent_name],
        device=env.device,
    )
    print(env.observation_spaces[agent_name])

    models[agent_name]["value"] = deterministic_model(
        observation_space=env.observation_spaces[agent_name],
        action_space=env.action_spaces[agent_name],
        device=env.device,
    )

# adjust some configuration if necessary
cfg_agent = MAPPO_DEFAULT_CONFIG.copy()
memories = {
    agent_name: RandomMemory(memory_size=1000) for agent_name in env.possible_agents
}

agent = MAPPO(
    possible_agents=env.possible_agents,
    models=models,
    memories=memories,  # only required during training
    cfg=cfg_agent,
    observation_spaces=env.observation_spaces,
    action_spaces=env.action_spaces,
    device=env.device,
    shared_observation_spaces=env.shared_observation_spaces,
)

agent.init()


observations, infos = env.reset()


while True:
    observations, infos = env.reset()
    max_timesteps = 100
    timestep = 0
    agent.init()
    old_observations = None
    while env.agents:
        print(f"timestep: {timestep}")
        # agent.pre_interaction(timestep, max_timesteps)
        actions = agent.act(observations, infos, timestep)

        det_actions = {
            agent_name: action.detach() for agent_name, action in actions[0].items()
        }

        observations, rewards, terminations, truncations, infos = env.step(det_actions)
        # if old_observations:
        #     agent.record_transition(
        #         old_observations,
        #         actions,
        #         rewards,
        #         observations,
        #         terminations,
        #         truncations,
        #         infos,
        #         timestep,
        #         max_timesteps,
        #     )
        # old_observations = observations
        # agent.post_interaction(timestep, max_timesteps)
        timestep += 1

print(env.agents)
env.close()
