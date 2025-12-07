# contact_model.py
import numpy as np
import torch

class HertzContact:
    """赫兹接触力计算模块（返回 torch.Tensor）"""
    def __init__(self):
        self.tissue_params = {
            'normal': {'E': 35e3, 'mu': 0.45},
            'tumor': {'E': 250e3, 'mu': 0.5}
        }

    def calculate_force(self, penetration_depth, rod_radius, tissue_type='normal', device=None):
        """
        返回 torch.tensor([Fx, Fy, Fz], dtype=float32, device=device)
        :param penetration_depth: 接触穿透深度 (m)（<=0 表示无接触）
        :param rod_radius: 导管半径 (m)
        :param tissue_type: 组织类型
        :param device: torch.device 或 None
        """
        if device is None:
            device = torch.device('cpu')
        if penetration_depth <= 0:
            return torch.zeros(3, dtype=torch.float32, device=device)

        E_tissue = self.tissue_params[tissue_type]['E']
        mu_tissue = self.tissue_params[tissue_type]['mu']
        E_rod = 193e9  # 导管弹性模量 (Pa)
        mu_rod = 0.3

        E_star = 1.0 / ((1 - mu_rod**2) / E_rod + (1 - mu_tissue**2) / E_tissue)
        R_star = rod_radius

        pd = abs(penetration_depth)
        F_normal = (4.0 / 3.0) * E_star * (R_star ** 0.5) * (pd ** 1.5)
        F_tangent = 0.1 * F_normal

        return torch.tensor([F_tangent, 0.0, F_normal], dtype=torch.float32, device=device)
