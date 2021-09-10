#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import parl

from smac.env import StarCraft2Env
from env_wrapper import SC2EnvWrapper
from copy import deepcopy
from replay_buffer import EpisodeExperience, EpisodeReplayBuffer

from transformer_model import TransformerModel
from qmixer_model import QMixerModel
from qmix import QMIX
from qmix_agent import QMixAgent
from actor import Actor
from parl.utils import logger

class Learner(object):
    def __init__(self, config):
        self.config = config
        #=== Create Agent ===
        env = StarCraft2Env(
            map_name=config['scenario'], difficulty=config['difficulty'])
        env = SC2EnvWrapper(env)
        config['episode_limit'] = env.episode_limit
        config['obs_shape'] = env.obs_shape
        config['state_shape'] = env.state_shape
        config['n_agents'] = env.n_agents
        config['n_actions'] = env.n_actions
        config = deepcopy(config)
        # 改成Transformer
        agent_model = TransformerModel(config['obs_shape'], config['n_actions'],
                            config['rnn_hidden_dim'])
        qmixer_model = QMixerModel(
                    config['n_agents'], config['state_shape'], config['mixing_embed_dim'],
                    config['hypernet_layers'], config['hypernet_embed_dim'])
        algorithm = QMIX(agent_model, qmixer_model, config['double_q'],
                    config['gamma'], config['lr'], config['clip_grad_norm'])
        qmix_agent = QMixAgent(
                    algorithm, config['exploration_start'], config['min_exploration'],
                    config['exploration_decay'], config['update_target_interval'])

        #=== Learner ===
        #self.total_steps = 0
        self.central_steps = 0
        self.learn_steps = 0
        self.target_update_count = 0
        self.rpm = EpisodeReplayBuffer(config['replay_buffer_size'])

        parl.connect(self.config['master_address'])
        #=== Remote Actor ===
        self.create_actors()

    def create_actors(self):
        self.remote_actors = [
            Actor(self.config) for _ in range(self.config['actor_num'])
        ]
        logger.info('Creating {} remote actors to connect.'.format(
            self.config['actor_num']))

    def step(self):
        pass

    def should_stop(self):
        #return self.total_steps >= self.config['training_steps']
        return self.central_steps >= self.config['training_steps']

    def run_evaluate_episode(self):
        self.qmix_agent.reset_agent()
        episode_reward = 0.0
        episode_step = 0
        terminated = False
        state, obs = self.env.reset()

        while not terminated:
            available_actions = self.env.get_available_actions()
            actions = self.qmix_agent.predict(obs, available_actions)
            state, obs, reward, terminated = self.env.step(actions)
            episode_step += 1
            episode_reward += reward

        is_win = self.env.win_counted
        return episode_reward, episode_step, is_win

    def save(self):
        self.env.save()