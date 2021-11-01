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

from smac.env import StarCraft2Env
from env_wrapper import SC2EnvWrapper
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from config import Config

from learner import Learner
from copy import deepcopy
from parl.utils import summary
import numpy as np

from runner import Runner

def run_sequential(config):
    # init config from env
    env_info = StarCraft2Env(map_name=config['scenario'], difficulty=config['difficulty'])
    env_info = SC2EnvWrapper(env_info)
    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info.state_shape},
        "obs": {"vshape": env_info.obs_shape, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": np.longlong},
        "avail_actions": {"vshape": (env_info.n_actions,), "group": "agents", "dtype": np.int32},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": np.uint8},
    }
    groups = {
        "agents": env_info.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=env_info.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, config['buffer_size'], env_info.episode_limit + 1,
                          preprocess=preprocess,
                          device="cpu" if config['buffer_cpu_only'] else config['device'])

    # UPDet mac --> PARL actor ???这里不能耦合到actor

    # runner select action --> send to remote --> data back
    runner = Runner(config, scheme=scheme, groups=groups, preprocess=preprocess)
    episode_batch = runner.run()
    buffer.insert_episode_batch(episode_batch)


    if buffer.can_sample(config['batch_size']):
        episode_sample = buffer.sample(config['batch_size'])

        # Truncate batch to only filled timesteps
        max_ep_t = episode_sample.max_t_filled()
        episode_sample = episode_sample[:, :max_ep_t]

        #learn


    while True:
        
        s,r = train_episode(env_info)

        if True:
            #learn
            pass

        # evaluate

        # save_model

def train_episode(env):
    # init
    env.reset()

    while True:
        # pre-update
        # select_action --> PARL actor
        env.step()
        # post-update

    #last update
    #last action data update
    





if __name__ == '__main__':
    config = deepcopy(Config)
    run_sequential(config)
