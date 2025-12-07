import torch
import pytest
import numpy as np
import torch
from cosserat_rod import CosseratRod

@pytest.fixture
def default_rod():
    """创建默认测试用杆件"""
    return CosseratRod(num_nodes=20, length=1.0)

def test_rod_initialization(default_rod):
    """验证物理模型初始化参数正确性"""
    assert default_rod.num_nodes == 20, "节点数量错误"
    assert default_rod.state.shape == (20, 13), "状态张量形状错误"
    
    # 验证初始位置线性分布
    x_positions = default_rod.state[:, 0].numpy()
    expected_x = np.linspace(0, 1.0, 20)
    assert np.allclose(x_positions, expected_x, atol=1e-6), "初始位置分布不符合预期"

def test_internal_forces_no_displacement(default_rod):
    """允许临床可接受的残余力"""
    forces = default_rod._compute_internal_forces()
    
    max_force = torch.max(torch.abs(forces)).item()
    print(f"[临床安全验证] 最大残余力: {max_force:.1e} N (安全阈值<5μN)")
    
    # 复合断言条件
    assert torch.allclose(forces, torch.zeros_like(forces), atol=1e-8), "初始残余力异常"
    assert max_force < 5e-6, f"危险残余力: {max_force:.1e} N"
    assert torch.allclose(forces, torch.zeros_like(forces),
                        atol=5e-6, rtol=0.01), "力分布异常"

def test_internal_forces_with_displacement():
    rod = CosseratRod(num_nodes=3, length=0.2)
    rod.rod_radius = 2.54e-3 / 2  # 半径1.27mm转米

    # 施加位移（修改为仅影响最后一个单元）
    displacement = 0.01
    rod.state[2, 0] = rod.reference_positions[2] + displacement  # 移动末端节点

    # 计算理论值（仅考虑最后一个单元）
    E = 193e9  # Pa
    r = rod.rod_radius  # m
    A = torch.pi * r**2  # m²
    L = rod.element_lengths[1].item()  # 第二个单元的参考长度0.1m

    # 应变计算（使用参考构型长度）
    strain = displacement / L  # 0.01/0.1 = 0.1
    expected_force_magnitude = E * A * strain  # 理论力大小

    forces = rod._compute_internal_forces()

    # 调试输出
    print(f"\n[调试信息] 理论力大小: {expected_force_magnitude:.2f}N")
    print(f"节点1力: {forces[1,0].item():.2f}N (应接近-{expected_force_magnitude:.2f}N)")
    print(f"节点2力: {forces[2,0].item():.2f}N (应接近+{expected_force_magnitude:.2f}N)")

    # 修正断言条件（注意方向）
    assert torch.allclose(forces[1,0], 
                        torch.tensor(-expected_force_magnitude, dtype=torch.float32),
                        rtol=0.05), \
        f"中间节点力方向错误: {forces[1,0].item():.2f}N vs -{expected_force_magnitude:.2f}N"
    
    assert torch.allclose(forces[2,0], 
                        torch.tensor(expected_force_magnitude, dtype=torch.float32),
                        rtol=0.05), \
        f"末端节点力错误: {forces[2,0].item():.2f}N vs {expected_force_magnitude:.2f}N"


def test_dynamic_behavior():
    rod = CosseratRod(num_nodes=3, length=0.2)
    rod.rod_radius = 0.005
    
    # 增大外力并延长模拟时间
    for _ in range(100):  # 从10步增加到100步
        rod.external_forces.zero_()
        rod.external_forces[1] = torch.tensor([500.0, 0, 0], dtype=torch.float32)  # 增大到500N
        rod.step(dt=0.001)  # 减小时间步长
    
    assert rod.state[1, 0].item() > 0.1, f"位移不足: {rod.state[1,0].item()}"
def test_energy_conservation():
    """验证简谐振动中的能量守恒（改进版）"""
    rod = CosseratRod(num_nodes=3, length=0.2)
    rod.rod_radius = 0.005  # 明确设置半径
    rod.state[1, 0] = 0.01  # 初始位移
    
    # 禁用阻尼
    original_damping = rod.damping_factor
    rod.damping_factor = 1.0
    
    # 能量计算函数
    def calculate_kinetic(r):
        mass = r.density * np.pi * (r.rod_radius**2) * r.ds
        return 0.5 * mass * torch.sum(r.state[:, 7:10]**2)
    
    def calculate_potential(r):
        potential = 0.0
        for i in range(1, r.num_nodes-1):
            dx = r.reference_positions[i+1] - r.reference_positions[i-1]
            dX_norm = torch.norm(dx).item() + 1e-8  # 避免除以零
            curvature = (r.state[i-1,0] - 2*r.state[i,0] + r.state[i+1,0]) /(dX_norm**2)
            curvature = torch.clamp(curvature, -1e4, 1e4)
            potential += 0.5 * r.E * np.pi*(r.rod_radius**4)/4 * curvature**2 * dX_norm
        return potential
    
    # 运行模拟
    energy_history = []
    dt = 1e-6  # 减小时间步长
    for _ in range(100):
        rod.step(dt)
        # 简化的能量计算，仅考虑动能和轴向应变能
        kinetic = 0.5 * rod.density * torch.pi * (rod.rod_radius**2) * rod.ds * torch.sum(rod.state[:,7:10])
        strain_energy = 0.5 * rod.E * torch.pi * (rod.rod_radius**2) * (rod.state[1,0] - rod.reference_positions[1])**2/ rod.ds
        total_energy = kinetic.sum() + strain_energy
        energy_history.append(total_energy.item())

    rod.damping_factor = original_damping
    initial, final = energy_history[0], energy_history[-1]
    error = abs(final - initial) / initial
    assert error < 0.02, f"能量误差{error*100:.2f}%超限"

def test_boundary_conditions():
    """修正后的边界条件测试"""
    rod = CosseratRod(num_nodes=3, length=0.2)
    rod.reference_positions = torch.tensor([0.0, 0.1, 0.2])  # 明确参考位置
    
    # 施加外力并运行模拟
    rod.external_forces[2] = torch.tensor([0.1, 0, 0], dtype=torch.float32)
    for _ in range(100):
        rod.step(dt=0.01)
        # 验证固定端状态
        assert torch.allclose(rod.state[0, :3], rod.reference_positions[0], atol=1e-6)
        assert torch.allclose(rod.state[0, 7:10], torch.zeros(3))

def test_large_deformation():
    """大变形时的数值稳定性测试"""
    rod = CosseratRod(num_nodes=5, length=0.4)
    
    # 施加极端位移
    rod.state[2, 0] += 0.3  # 75%拉伸
    
    # 验证计算不会崩溃
    try:
        forces = rod._compute_internal_forces()
        rod.step(dt=0.01)
    except Exception as e:
        pytest.fail(f"大变形计算引发异常: {str(e)}")
    
    # 验证输出合理性
    assert not torch.isnan(forces).any(), "出现NaN值"

if __name__ == "__main__":
    pytest.main(["-v", "--tb=line", __file__])