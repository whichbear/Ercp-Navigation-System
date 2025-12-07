import gymnasium as gym
import numpy as np
from gymnasium import spaces
from cosserat_rod import CosseratRod
import torch  # 添加缺失的torch引用

class ERCPEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # 物理系统初始化
        self.rod = CosseratRod(num_nodes=20, length=0.3)  # 30cm导管
        self.wall_z = 0.0  # 模拟十二指肠壁位置
        
        # 目标位置
        self.target_pos = np.array([0.25, 0.02, -0.1], dtype=np.float32)  # 3D目标点
        
        # 空间定义
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.rod.num_nodes*13 + 3,),  # 状态+目标位置
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(3,),  # 施加在导管尖端的力[x,y,z]
            dtype=np.float32
        )
        self.max_steps = 200
        self.current_step = 0
        self.dt = 0.02  # 默认时间步长（可按需修改)

    def step(self, action, dt=None):
        """兼容 gym 的 step：接收可选 dt 参数"""
        if dt is None:
            dt = self.dt

        # 将动作转换为物理力（确保张量类型一致）
        applied_force = action * 0.1  # 缩放系数
        
        # 施加外力到导管尖端（使用正确索引）
        self.rod.external_forces[-1] += torch.tensor(applied_force, dtype=torch.float32)
        
        # 接触检测与处理
        penetrations, contact_mask = self.rod.detect_contact(self.wall_z)
        self.rod.apply_contact_forces(contact_mask, penetrations)
        
        # 使用时间步长推进物理系统
        self.rod.step(dt=dt)
        
        # 观测与奖励计算
        obs = self._get_obs()
        reward = self._compute_reward(penetrations)
        
        # 终止条件判断
        self.current_step += 1
        terminated = self._check_done()
        truncated = self.current_step >= self.max_steps

        info = {}
        return obs, reward, terminated, truncated, info

    # 以下方法保持不变
    def _get_obs(self):
        rod_state = self.rod.state.numpy().flatten()
        return np.concatenate([rod_state, self.target_pos]).astype(np.float32)

    def _compute_reward(self, penetrations):
        tip_pos = self.rod.state[-1, :3].numpy()
        position_error = -10.0 * np.linalg.norm(tip_pos - self.target_pos)
        contact_penalty = -0.5 * np.sum(np.abs(penetrations))
        smoothness_penalty = -0.1 * np.linalg.norm(self.rod.state[-1, 7:10].numpy())
        return position_error + contact_penalty + smoothness_penalty

    def _check_done(self):
        tip_pos = self.rod.state[-1, :3].numpy()
        return (
            np.linalg.norm(tip_pos - self.target_pos) < 0.005 or 
            self.current_step >= self.max_steps
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rod = CosseratRod(num_nodes=20, length=0.3)
        self.current_step = 0
        return self._get_obs(), {}