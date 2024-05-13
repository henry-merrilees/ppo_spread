from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.utils.model_instantiators.torch import multivariate_gaussian_model
from pettingzoo.mpe import simple_spread_v3
import torch

hidden_size = 32
num_agents = 3
max_timesteps = 30

env = simple_spread_v3.parallel_env(
    render_mode="human", # render_mode="rgb_array" if you want to drop visualization for sake of speed
    N=num_agents,
    local_ratio=0.5,
    max_cycles=max_timesteps,
    continuous_actions=True,
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
    models[agent_name]["policy"] = multivariate_gaussian_model(
        observation_space=env.observation_spaces[agent_name],
        action_space=env.action_spaces[agent_name],
        device=env.device,
    )
    models[agent_name]["value"] = multivariate_gaussian_model(
        observation_space=env.observation_spaces[agent_name].shape[0]*num_agents,
        action_space=1,
        device=env.device,
    )
# adjust some configuration if necessary
cfg_agent = MAPPO_DEFAULT_CONFIG.copy()
cfg_agent["learning_rate"] = 1e-3
memories = {
    agent_name: RandomMemory(memory_size=10000) for agent_name in env.possible_agents
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


agent.init()
pooled_rewards = []
while True:
    observations, infos = env.reset()
    timestep = 0
    prev_data = None
    rewards = []
    while env.agents:
        states = agent.act(observations, infos, timestep)

        det_actions = {
            agent_name: torch.clamp(action.detach(), 0.0, 1.0)
            for agent_name, action in states[0].items()
        }

        data = env.step(det_actions)
        _, reward, *_ = data
        avg_reward = (sum(reward.values())/len(reward))

        # print info keys
        if prev_data and agent._current_log_prob is not None:
            (
                old_observations,
                old_rewards,
                old_terminations,
                old_truncations,
                old_infos,
            ) = prev_data

            # Handle the absence of 'shared_next_states'. This could be setting it to None or an appropriate default
            old_infos["shared_next_states"] = infos["shared_states"]

            agent.record_transition(
                states=old_observations,
                actions=det_actions,
                rewards=old_rewards,
                next_states=observations,
                terminated={
                    agent_name: old_terminations[agent_name]
                    for agent_name in old_terminations
                },
                truncated={
                    agent_name: old_truncations[agent_name]
                    for agent_name in old_truncations
                },
                infos=old_infos,
                timestep=timestep-1,
                timesteps=max_timesteps,
            )
        prev_data = data
    print("reward", avg_reward.item())
env.close()
