from utils.provider import Provider
from collections import deque
from utils.model import ActorCritic
import torch
import torch.nn as nn
import torch.optim as optim

class LLMAgent():
    def __init__(self, setting_file, agent_name):

        self.setting = setting_file
        self.agent_name = agent_name

        self.provider_name = self.setting['provider']
        self.model = self.setting['model']
        self.temperature = self.setting['temperature']
        self.api_key = self.setting['api_key']

        self.provider = Provider(
            provider=self.provider_name,
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
        )

        self.memory_length = self.setting['memory_length']
        self.memory = {
            'prompt': deque(maxlen=self.memory_length),
            'llm_actions': deque(maxlen=self.memory_length)
        }
            
    def answer(self, prompt):
        # Update memory
        # self.memory['prompt'].append(prompt)
        # Query llm
        llm_actions = self.provider.query_llm(prompt)

        # self.memory['llm_actions'].append(llm_actions.content)

        return llm_actions.content


class A2C_manager():
    def __init__(self, state_dim, action_dim, actor_lr=0.001, critic_lr=0.005, gamma=0.99):
        self.gamma = gamma
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer_actor = optim.Adam(self.actor_critic.actor.parameters(), lr=actor_lr)
        self.optimizer_critic = optim.Adam(self.actor_critic.critic.parameters(), lr=critic_lr)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.actor_critic(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def train(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.LongTensor([action])
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])

        _, state_value = self.actor_critic(state)
        _, next_state_value = self.actor_critic(next_state)

        expected_value = reward + (1 - done) * self.gamma * next_state_value
        advantage = expected_value - state_value

        action_probs, _ = self.actor_critic(state)
        action_log_probs = torch.log(action_probs.squeeze(0)[action])

        actor_loss = -action_log_probs * advantage.detach()
        critic_loss = self.loss_fn(state_value, expected_value.detach())

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

    def save_model(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def load_model(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.actor_critic.eval()

