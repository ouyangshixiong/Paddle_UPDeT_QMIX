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

from learner import Learner
from config import QMixConfig as config
from parl.utils import summary
import numpy as np

if __name__ == '__main__':
    # center learning
    learner = Learner(config)
    while not learner.should_stop():
        loss, td_error = learner.step()
        summary.add_scalar('train_loss', loss, learner.central_steps)
        summary.add_scalar('train_td_error:', td_error, learner.central_steps)

        if learner.central_steps % config['test_steps'] == 0:
            eval_reward_buffer = []
            eval_steps_buffer = []
            eval_is_win_buffer = []
            for _ in range(3):
                eval_reward, eval_step, eval_is_win = learner.run_evaluate_episode()
                eval_reward_buffer.append(eval_reward)
                eval_steps_buffer.append(eval_step)
                eval_is_win_buffer.append(eval_is_win)
            summary.add_scalar('eval_reward', np.mean(eval_reward_buffer),
                            learner.central_steps)
            summary.add_scalar('eval_steps', np.mean(eval_steps_buffer),
                            learner.central_steps)
            mean_win_rate = np.mean(eval_is_win_buffer)
            summary.add_scalar('eval_win_rate', mean_win_rate,
                            learner.central_steps)
            summary.add_scalar('target_update_count',
                            learner.target_update_count, learner.central_steps)
