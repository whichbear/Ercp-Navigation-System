import torch
import numpy as np
from contact_model import HertzContact

class CosseratRod:
    def __init__(self, num_nodes=20, length=1.0, density=7900):
        self.num_nodes = num_nodes
        self.length = length
        self.ds = length / (num_nodes - 1)
        self.density = density
        self.rod_radius = 2.54e-3 / 2  # 导管半径 (单位:m)
        
        # 材料本构参数 (SS304不锈钢修正值)
        self.material = {
            'elastic_modulus': 193e9,    # 弹性模量 E (Pa)
            'poisson_ratio': 0.295,      # 泊松比 ν
            'yield_stress': 215e6,       # 初始屈服应力 σ_y0 (Pa)
            'hardening_modulus': 1.05e9, # 塑性硬化模量 H (Pa)
            'beta': 0.65                 # 混合硬化系数 (0=各向同性,1=运动)
        }
        self.E = self.material['elastic_modulus']  # 保持旧版本API兼容
        self.nu = self.material['poisson_ratio']
        self.sigma_y0 = self.material['yield_stress']
        
        # 状态变量 [x, y, z, qw, qx, qy, qz, vx, vy, vz, ωx, ωy, ωz]
        self.state = torch.zeros((num_nodes, 13), dtype=torch.float32)
        self.state[:, 0] = torch.linspace(0, length, num_nodes, 
                                        dtype=torch.float64).float()  # X轴初始位置
        self.state[:, 3] = 1.0 + 1e-9  # 四元数初始化避免奇异
        
        # 新增塑性状态变量
        self.ep_eq = torch.zeros(num_nodes)          # 等效塑性应变 (标量/节点)
        self.back_stress = torch.zeros((num_nodes, 3,3))  # 背应力张量 (矢量/节点)
        
        # 外力系统
        self.external_forces = torch.zeros((num_nodes, 3), dtype=torch.float32)
        self.contact_model = HertzContact()   # 接触力模型
        self.damping_factor = 0.98            # 数值阻尼系数
        # 新增参考构型坐标
        self.reference_positions = torch.linspace(0, length, num_nodes, dtype=torch.float32)
        self.state[:, 0] = self.reference_positions.clone() 
        # 修改为真实单元长度计算
        self.element_lengths = torch.diff(self.reference_positions)
        # 新增旋转矩阵缓存优化
        self._rotation_matrices = torch.zeros((num_nodes, 3, 3), dtype=torch.float32)
        element_volume = torch.pi * (self.rod_radius**2) * self.element_lengths
        self.element_mass = self.density * torch.pi * (self.rod_radius**2) * self.element_lengths
        self.mass_per_node = torch.zeros(num_nodes, dtype=torch.float32)
        if num_nodes > 1:
            self.mass_per_node[0] = 0.5 * self.element_mass[0]
            self.mass_per_node[-1] = 0.5 * self.element_mass[-1]
            self.mass_per_node[1:-1] = 0.5*(self.element_mass[:-1] + self.element_mass[1:])
            if num_nodes > 2:
                self.mass_per_node[1:-1] = 0.5 * (self.element_mass[:-1] + self.element_mass[1:])
    def _compute_rotation_matrices(self):
        """修正四元数转旋转矩阵的数学公式"""
        q = self.state[:, 3:7]
        q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-8)  # 严格归一化
    
        # 分离四元数分量 (w, x, y, z)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
        # 使用标准旋转矩阵公式
        self._rotation_matrices = torch.stack([torch.stack([1 - 2*(y**2+z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
        torch.stack([2*(x*y+w*z), 1 - 2*(x**2+z**2), 2*(y*z - w*x)], dim=-1),
        torch.stack([2*(x*z-w*y), 2*(y*z + w*x), 1 - 2*(x**2+y**2)], dim=-1)],dim=2)
        return self._rotation_matrices
    
    def quaternion_to_matrix(self, q):
        """四元数转旋转矩阵（优化GPU支持）"""
        q = self.state[:, 3:7] / (torch.norm(self.state[:, 3:7], dim=1, keepdim=True) + 1e-8)
        w, x, y, z = q.unbind(-1)
        return torch.stack([
            torch.stack([1-2*(y**2+z**2),2*(x*y-y*w),2*(x*z+y*w)], dim=-1),
            torch.stack([2*(x*y+z**2),1-2*(x**2+z**2),2*(y*z-x*w)], dim=-1),
            torch.stack([2*(x*z-z*y),2*(y*z+x*w),1-2*(x**2+y**2)], dim=-1)
        ], dim=1)
    def _compute_internal_forces(self):
        forces = torch.zeros_like(self.state[:, :3])
        R_matrices = self._compute_rotation_matrices()

        for i in range(self.num_nodes - 1):
            # 获取正确的单元参数
            dX = self.element_lengths[i].item()  # 原始单元长度（单位：米）
            dx = (self.state[i+1, :3] - self.state[i, :3])  # 位移差（单位：米）
        
            # 转换到材料坐标系
            R = R_matrices[i]
            dx_local = R.T @ dx
        
            # 计算工程应变（无量纲）
            strain_axial = (dx_local[0] - dX) / (dX + 1e-8)
        
            # 计算轴向应力（单位：Pa）
            stress_axial = self.E * strain_axial
        
            # 计算截面积（单位：m²）
            A = torch.pi * (self.rod_radius**2)
        
            # 计算轴向力（单位：牛顿）
            force_magnitude = stress_axial * A
            force_global = R @ torch.tensor([force_magnitude, 0.0, 0.0], dtype=torch.float32)
            print(f"\n单元 {i} 力计算:")
            print(f"节点 {i} 坐标: {self.state[i, :3].numpy()}")
            print(f"节点 {i+1} 坐标: {self.state[i+1, :3].numpy()}")
            print(f"位移差(全局): {dx.numpy()}")
            print(f"位移差(局部): {dx_local.numpy()}")
            print(f"轴向应变: {strain_axial:.4f}")
            print(f"轴向应力: {stress_axial:.2f} Pa")
            print(f"局部力: [{force_magnitude:.2f}, 0.0, 0.0] N")
            print(f"全局力: {force_global.numpy()}")
            # 分配节点力（牛顿第三定律）
            forces[i] -= force_global
            forces[i+1] += force_global

        return forces

    def _compute_green_strain(self, i, R_matrices):
        """修正变形梯度方向"""
        # 参考构型坐标差（固定为初始配置）
        dX = self.reference_positions[i] - self.reference_positions[i-1]  # 单元左端长度
    
        # 当前构型坐标差（全局坐标系）
        dx_backward = self.state[i, :3] - self.state[i-1, :3]  # 左单元变形
    
        # 转换到材料坐标系（使用节点i的旋转矩阵）
        R = R_matrices[i]
        F_backward = (R.T @ dx_backward) / dX  # 左单元变形梯度
    
        return 0.5 * (F_backward.T @ F_backward - torch.eye(3))  # 格林应变

    def j2_plasticity(self, strain, node_idx):
        """修正后的J2混合硬化模型（返回映射算法)"""
        # ================== 参数准备 ==================
        E = self.material['elastic_modulus']
        nu = self.material['poisson_ratio']
        H = self.material['hardening_modulus']
        beta = self.material['beta']
        sigma_y0 = self.material['yield_stress']
    
        mu = E / (2*(1 + nu))  # 剪切模量
        K = E / (3*(1 - 2*nu)) # 体积模量
        tolerance = 1e-6        # 数值容差

        # ================== 弹性预测 ==================
        # 分解体积应变和偏应变
        strain_vol = torch.trace(strain) / 3 * torch.eye(3)
        strain_dev = strain - strain_vol
    
        # 弹性预测应力（考虑历史背应力）
        stress_vol = 3 * K * strain_vol
        stress_dev_pred = 2 * mu * (strain_dev - self.ep_eq[node_idx] * np.sqrt(2/3))
        stress_pred = stress_vol + stress_dev_pred - self.back_stress[node_idx]

        # ================== 屈服判断 ==================
        # 计算等效偏应力
        deviatoric_stress = stress_pred - torch.trace(stress_pred)/3 * torch.eye(3)
        sigma_eq = torch.sqrt(1.5*torch.sum(deviatoric_stress**2))
    
        # 当前屈服应力（各向同性硬化部分）
        sigma_y = sigma_y0 + (1 - beta) * H * self.ep_eq[node_idx]
    
        if sigma_eq <= sigma_y + tolerance:
            return stress_pred  # 保持弹性预测

        # ================== 塑性修正 ==================
        # 计算流动方向
        n = deviatoric_stress / (sigma_eq + tolerance)
    
        # 迭代求解塑性乘子（改进的收敛算法）
        delta_gamma = 0.0
        for _ in range(5):  # 3-5次迭代即可收敛
            residual = sigma_eq - 3*mu*delta_gamma - (sigma_y + H*delta_gamma)
            delta_gamma += residual / (3*mu + H + tolerance)
            delta_gamma = torch.clamp(delta_gamma, min=0)

        # ================== 状态更新 ==================
        # 更新等效塑性应变
        self.ep_eq[node_idx] += delta_gamma
    
        # 更新背应力（运动硬化部分）
        delta_back_stress = beta * H * delta_gamma * n
        self.back_stress[node_idx] += delta_back_stress
    
        # ================== 应力更新 ==================
        # 偏应力回拉
        stress_dev_corrected = stress_dev_pred - 2*mu*delta_gamma*n
    
        # 最终应力（含背应力调整）
        stress_corrected = stress_vol + stress_dev_corrected - self.back_stress[node_idx]
    
        return stress_corrected

    def stress_to_force(self, stress, i):
        """基于材料方向的应力积分"""
        # 材料坐标系中的力（X方向）
        A = torch.pi * (self.rod_radius**2)
        local_force = torch.tensor([stress[0,0] * A, 0.0, 0.0], dtype=torch.float32)
        
        # 转换到全局坐标系
        R = self._compute_rotation_matrices()[i]
        return R @ local_force



    def detect_contact(self, wall_z):
        """精确接触检测"""
        # 计算导管底部位置（z轴向上）
        bottom_z = self.state[:, 2] - self.rod_radius
        penetrations = wall_z - bottom_z
        contact_mask = penetrations > 0
        return penetrations * contact_mask, contact_mask


    def apply_contact_forces(self, contact_mask, penetrations):
        """使用张量索引高效应用接触力"""
        contact_indices = torch.where(contact_mask)[0]  # 获取接触节点索引
    
        for i in contact_indices:
            force = self.contact_model.calculate_force(
                penetration_depth=penetrations[i].item(),  # 转换标量值
                rod_radius=self.rod_radius,
                tissue_type='normal'
            )
            self.external_forces[i] += torch.tensor(force, dtype=torch.float32)

    def step(self, dt):
        """执行物理模拟步"""
        # 内力计算
        internal_forces = self._compute_internal_forces()
    
        # 总力计算
        total_forces = internal_forces + self.external_forces
    
        # 质量计算（确保张量类型）
        mass_per_node = torch.tensor(self.density*torch.pi*(self.rod_radius**2)*self.ds,dtype=torch.float32).unsqueeze(-1)
    
        # 加速度计算
        acceleration = total_forces / mass_per_node
        acceleration = torch.clamp(total_forces / mass_per_node, -1e4, 1e4)
    
        new_velocity = self.state[:,7:10] * self.damping_factor + acceleration * dt
        self.state[:,7:10] = new_velocity  # 限制速度范围
        
        # 位置更新（半隐式欧拉）
        self.state[:,:3] += self.state[:,7:10] * dt
         # 强制固定左端节点
        self.state[0, :3] = self.reference_positions[0]  # 位置重置为初始值
        self.state[0, 7:10] = torch.zeros(3)              # 速度清零
        # 清空外力
        self.external_forces.zero_()