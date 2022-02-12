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

Config = {
    'master_address': 'localhost:8010',
    'actor_num': 1,
    'env_num': 1,
    'sample_batch_episode': 2,
    'log_metrics_interval_s': 1,
    'calc_num': 2,
    'scenario': '8m_vs_9m',
    'replay_buffer_size': 5000,
    'mixing_embed_dim': 32,
    'rnn_hidden_dim': 64,
    'lr': 0.0005,
    'memory_warmup_size': 16,
    'gamma': 0.99,
    'exploration_start': 1.0,
    'min_exploration': 0.1,
    'exploration_decay': 2e-6,
    'update_target_interval': 50,
    'batch_size': 16,
    'training_steps': 5000,
    'test_steps': 100,
    'clip_grad_norm': 10,
    'hypernet_layers': 2,
    'hypernet_embed_dim': 64,
    'double_q': True,
    'difficulty': '7',
    'buffer_size': 5000, # Size of the replay buffer
    'buffer_cpu_only': True,
    'obs_agent_id': False, # Include the agent's one_hot id in the observation
    'obs_last_action': False, # Include the agent's last action (one_hot) in the observation
    # --- Transformer parameters. Should be set manually. ---
    'token_dim': 5, # Marines. For other unit type (e.g. Zeolot) this number can be different (6).
    'emb': 32, # embedding dimension of transformer
    'heads': 3, # head number of transformer
    'depth': 2, # block number of transformer
    'ally_num': 8, # number of ally (5m_vs_6m)
    'enemy_num': 9, # number of enemy (5m_vs_6m)
}