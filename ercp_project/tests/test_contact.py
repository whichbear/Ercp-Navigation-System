import torch
import pytest
import numpy as np
from cosserat_rod import CosseratRod
from contact_model import HertzContact

@pytest.fixture
def contact_model():
    return HertzContact()

# ---------------------------
# 基础功能验证
# ---------------------------
def test_hertz_force_basic(contact_model):
    """验证基础赫兹力计算"""
    force = contact_model.calculate_force(
        penetration_depth=0.001,  # 1mm穿透
        rod_radius=0.002,         # 2mm导管
        tissue_type='normal'
    )
    
    # 理论值计算
    E_star = 1 / ((1-0.3**2)/193e9 + (1-0.45**2)/35e3)
    R_star = 0.002
    expected = (4/3) * E_star * np.sqrt(R_star) * (0.001)**1.5
    
    # 验证法向分量
    assert np.isclose(force[2], expected, rtol=0.01), "法向力计算错误"
    # 验证切向分量
    assert force[0] == pytest.approx(0.1 * force[2], rel=0.1), "切向力比例错误"

# ---------------------------
# 边界条件测试
# ---------------------------
@pytest.mark.parametrize("penetration, expected_force", [
    (0.0, 0.0),        # 零穿透
    (-0.001, 0.0),     # 负穿透（无接触）
    (1e-6, 0.26e-5),    # 微小穿透
    (0.01, 2.6)       # 大穿透
])
def test_penetration_edge_cases(contact_model, penetration, expected_force):
    """边界穿透值测试"""
    force = contact_model.calculate_force(
        penetration_depth=penetration,
        rod_radius=0.002,
        tissue_type='normal'
    )
    assert np.isclose(force[2], expected_force, rtol=0.5), f"穿透深度{penetration}m时力值异常"

# ---------------------------
# 组织类型验证
# --------------------------
def test_tissue_type_variation(contact_model):
    """验证不同组织类型的刚度影响"""
    force_normal = contact_model.calculate_force(0.001, 0.002, 'normal')
    force_tumor = contact_model.calculate_force(0.001, 0.002, 'tumor')
    
    # 肿瘤组织刚度更大应产生更大接触力
    assert force_tumor[2] > force_normal[2] * 5, "组织类型影响不符合预期"

# ---------------------------
# 多节点接触场景
# ---------------------------
def test_multi_node_contact():
    """验证多个节点同时接触时的力分配（z轴向上坐标系）"""
    rod = CosseratRod(num_nodes=5)
    rod.rod_radius = 0.002  # 导管半径2mm
    
    # 设置节点z坐标（单位：米）
    rod.state[:, 2] = torch.tensor(
        [0.0010, 0.0015, 0.0018, 0.0030, 0.0025],  # 修改节点2的z坐标以确保接触
        dtype=torch.float32
    )
    
    # 接触检测（地面在z=0）
    penetrations, contact_mask = rod.detect_contact(wall_z=0.0)
    
    # --- 断言1：接触节点数量 ---
    assert sum(contact_mask) == 3, f"应检测到3个接触节点，实际{sum(contact_mask)}个"
    
    # --- 断言2：接触节点索引 ---
    expected_mask = [True, True, True, False, False]  # 节点0、1、2、4接触（共4个，需进一步调整）
    # 若需要严格3个接触，调整坐标为[0.0010, 0.0015, 0.0018, 0.0030, 0.0025]
    assert contact_mask.numpy().tolist() == expected_mask, "接触节点索引错误"
    
    # --- 应用接触力 ---
    rod.apply_contact_forces(contact_mask, penetrations)
    
    # --- 断言3：法向力方向验证 ---
    contact_nodes = torch.where(contact_mask)[0]
    assert torch.all(rod.external_forces[contact_nodes, 2] > 0), "法向力方向错误"
    
    # --- 断言4：切向力验证 ---
    assert torch.any(rod.external_forces[contact_nodes, 0] != 0), "切向力未正确计算"

# ---------------------------
# 动态行为验证
# ---------------------------
def test_contact_dynamics():
    """验证接触力对物理模拟的影响"""
    rod = CosseratRod(num_nodes=3)
    rod.rod_radius = 0.005
    
    # 初始穿透
    rod.state[:, 2] = torch.tensor([-0.001, -0.002, -0.003])
    
    # 记录初始位置
    initial_z = rod.state[:, 2].clone()
    
    # 模拟10步
    for _ in range(10):
        penetrations, contact_mask = rod.detect_contact(0.0)
        rod.apply_contact_forces(contact_mask, penetrations)
        rod.step(dt=0.01)
    
    # 验证位置变化 (应被接触力推回)
    assert torch.all(rod.state[:, 2] > initial_z), "接触未影响运动轨迹"

# ---------------------------
# 异常处理测试
# ---------------------------
def test_invalid_tissue_type(contact_model):
    """验证无效组织类型的错误处理"""
    with pytest.raises(KeyError):
        contact_model.calculate_force(0.001, 0.002, 'invalid_tissue')

# ---------------------------
# 数值稳定性测试
# ---------------------------
def test_numerical_stability(contact_model):
    """极端穿透深度下的数值稳定性"""
    extreme_penetration = 1.0  # 1米（非生理范围）
    force = contact_model.calculate_force(extreme_penetration, 0.002, 'normal')
    
    assert not np.isnan(force).any(), "出现数值不稳定"
    assert np.isfinite(force).all(), "计算结果非有限值"

if __name__ == "__main__":
    pytest.main(["-v", "--tb=line", __file__])