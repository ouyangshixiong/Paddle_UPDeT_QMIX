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

import time
import parl
import numpy as np

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
        self.__create_agent()

        #=== init Learner params ===
        #self.total_steps = 0
        self.central_steps = 0
        self.learn_steps = 0
        self.target_update_count = 0
        self.rpm = EpisodeReplayBuffer(config['replay_buffer_size'])

        #=== Remote Actor ===
        parl.connect(self.config['master_address'])
        self.__create_actors()

    def __create_agent(self):
        # init config from env
        env = StarCraft2Env(map_name=self.config['scenario'], difficulty=self.config['difficulty'])
        env = SC2EnvWrapper(env)
        self.config['episode_limit'] = env.episode_limit
        self.config['obs_shape'] = env.obs_shape
        self.config['state_shape'] = env.state_shape
        self.config['n_agents'] = env.n_agents
        self.config['n_actions'] = env.n_actions
        
        # 从RNN改成Transformer
        agent_model = TransformerModel(self.config['obs_shape'], self.config['n_actions'],
                            self.config['rnn_hidden_dim'])
        qmixer_model = QMixerModel(
                    self.config['n_agents'], self.config['state_shape'], self.config['mixing_embed_dim'],
                    self.config['hypernet_layers'], self.config['hypernet_embed_dim'])
        algorithm = QMIX(agent_model, qmixer_model, self.config['double_q'],
                    self.config['gamma'], self.config['lr'], self.config['clip_grad_norm'])
        self.qmix_agent = QMixAgent(
                    algorithm, self.config['exploration_start'], self.config['min_exploration'],
                    self.config['exploration_decay'], self.config['update_target_interval'])

    def __create_actors(self):
        self.remote_actors = [
            Actor(self.config) for _ in range(self.config['actor_num'])
        ]
        logger.info('Creating {} remote actors to connect.'.format(
            self.config['actor_num']))

    def step(self):
        self.central_steps += 1
        # get all data from remote actors.
        c1 = time.time()
        sample_data_object_ids = [
            remote_actor.sample() for remote_actor in self.remote_actors
        ]
        sample_datas = [
            future_object.get() for future_object in sample_data_object_ids
        ]

        for sample_data in sample_datas:
            for data in sample_data:
                #if 'steps' == data:
                    #for steps in sample_data[data]:
                        #self.total_steps += steps
                #elif 'episode_experience' == data:
                if 'episode_experience' == data:
                    for episode_experience in sample_data[data]:
                        self.rpm.add(episode_experience)
        print("collect data time cost:{}".format( time.time()-c1 ))

        mean_loss = []
        mean_td_error = []
        for _ in range(self.config['calc_num']):
            s_batch, a_batch, r_batch, t_batch, obs_batch, available_actions_batch,\
                    filled_batch = self.rpm.sample_batch(self.config['batch_size'])
            # center learn
            loss, td_error = self.qmix_agent.learn(s_batch, a_batch, r_batch, t_batch,
                                        obs_batch, available_actions_batch,
                                        filled_batch)
            mean_loss.append(loss)
            mean_td_error.append(td_error)

            # TODO confirm whether UPDeT framework need to update remote network params
            self.__update_remote_network()
        
        mean_loss = np.mean(mean_loss) if mean_loss else None
        mean_td_error = np.mean(mean_td_error) if mean_td_error else None
        return mean_loss, mean_td_error

    def __update_remote_network(self):
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