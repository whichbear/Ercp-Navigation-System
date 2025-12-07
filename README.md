# ERCP 下插管导航系统

本仓库包含用于 ERCP 下插管导航研究的仿真与测试代码（`ercp_project/`），以及一个 Ultralytics YOLOv8 的代码副本（`ultralytics-8.3.94目标识别/ultralytics-8.3.94/`）。




## 目录概览
- `ercp_project/`：物理模型与环境
	- `contact_model.py`：赫兹接触力模型，基于 `torch`，返回接触力张量。
	- `cosserat_rod.py`：Cosserat 杆（导管）离散化与力学求解，包含内力、接触检测、塑性模型与时间积分步 `step()`。
	- `ercp_env.py`：基于 `gymnasium` 的仿真环境 `ERCPEnv`，封装 `CosseratRod` 并实现 `step`/`reset`。
	- `tests/`：单元测试（`pytest`）覆盖接触模型与杆件动力学。
- `ultralytics-8.3.94目标识别/ultralytics-8.3.94/`：Ultralytics 示例代码与脚本（用于目标检测、训练等）。

## 要点 / 依赖
- 推荐 Python 版本：3.8+
- 主要依赖（项目已自动生成 `requirements.txt`，见仓库根目录）：
	- `torch`（CPU 或 CUDA 版本，依据你环境）
	- `numpy`
	- `gymnasium`
	- `pytest`

安装示例（PowerShell）：

```powershell
cd 'd:\ercp下插管导航系统'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

如果你需要 GPU 支持，请先安装相应的 `torch` 包（含 CUDA）。

## 核心贡献与开发进展

**核心贡献（概要）**

- 我在本项目的主要贡献在于：
	1. 开发满足医学要求的物理仿真模块（见 `ercp_project/`），用于模拟导管-组织交互与接触力学；
	2. 基于 Ultralytics YOLO 实现对手术乳头及相关解剖结构的目标检测，并将视觉检测结果用于导管导航决策。

	更多验证细节请参考 `ercp_project/tests/` 中的单元测试（`test_rod.py`, `test_contact.py`），测试覆盖接触力学、数值稳定性与功能性断言，说明物理仿真已满足相关医学/数值要求。

**已实现（可验证 / 已开发但部分未上传）**

- 物理仿真模块：`ercp_project/cosserat_rod.py`, `ercp_project/contact_model.py`, `ercp_project/ercp_env.py`；实现内力计算、塑性模型、接触检测与时间积分，已通过单元测试验证。
- 视觉→物理 参数传递接口：已完成初步接口层实现（vision→physics），用于将 YOLO 输出转换为物理仿真可用的位姿/位置参数
- 视觉能力增强：实现乳头空间位置识别、肠腔轴线识别与导管姿态估计，提升检测在导航场景中的可用性。
- 风险评估与混合决策模块：已搭建初步模块，用于评估视觉与物理子模型输出并实现简单的混合决策策略（风险评分 + 决策融合）。

> 说明：上述部分功能中，若包含大模型或大数据集，部分实现已完成但尚未上传仓库。

**在研 / 后续计划**

- 引入 MasFusion 进行多尺度信息融合：融合图像、几何与时间尺度特征以提高关键交互信息的鲁棒性与识别率。
- 动态校准特征权重：在线根据观测置信度与历史表现调整融合特征权重，确保系统聚焦于对导航决策最关键的信息。
- 构建专家系统：在运行过程中收集交互数据，用于训练/构建专家规则集，以增强物理模型在异常或边界情形下的稳健性与安全性。



## 运行单元测试

在仓库根目录执行：

```powershell
pytest -q
```

测试涵盖 `ercp_project` 中的接触模型与杆件动力学核心函数。

## 快速使用示例（仿真）

下面的 Python 代码展示如何在交互式环境或脚本中运行 `ERCPEnv`：

```python
from ercp_project.ercp_env import ERCPEnv

env = ERCPEnv()
obs, _ = env.reset()
for _ in range(env.max_steps):
		action = env.action_space.sample()  # 随机策略示例
		obs, reward, terminated, truncated, info = env.step(action)
		if terminated or truncated:
				break

print('仿真结束, reward=', reward)
```

或者直接使用 `CosseratRod` 进行小范围测试（快速脚本）：

```python
from ercp_project.cosserat_rod import CosseratRod
rod = CosseratRod(num_nodes=5, length=0.2)
rod.state[2,0] += 0.01
forces = rod._compute_internal_forces()
print(forces)
```

## Ultralytics 部分

Ultralytics 子目录包含 `detect.py`、训练脚本和数据集组织示例。若要运行检测示例：

```powershell
cd "ultralytics-8.3.94目标识别\ultralytics-8.3.94"
python detect.py --source path/to/images --weights path/to/weights.pt
```
 
## 如何发布到 GitHub（示例）

```powershell
# 进入项目目录
cd 'd:\ercp下插管导航系统'

#（可选）先检查本地最大的文件，避免意外把大模型/数据提交上来
Get-ChildItem -Recurse -File | Sort-Object Length -Descending |
	Select-Object FullName,@{Name='SizeMB';Expression={[math]::Round($_.Length/1MB,2)}} -First 30

# 初始化仓库（若已经是 git 仓库可跳过）
git init

# 确保 .gitignore 存在且规则正确
git add .gitignore

# 添加 README 和其他小文件（先不要 add 整个目录，如果不确定是否含大文件）
git add README.md requirements.txt

# 或者一次性添加所有（在你确认没有已跟踪的大文件后）
# git add .

# 提交
git commit -m "Initial: add README, .gitignore, requirements"

# 添加远程(使用 HTTPS)
git remote add origin https://github.com/whichbear/Ercp-Navigation-System.git

# 将分支改名为 main（如需要）
git branch -M main

# 推送到远程（首次推送设置 upstream）
git push -u origin main
```

如果历史中已包含大文件，请在推送前使用 `git filter-repo` 或 BFG 清理历史，我可以协助。


## 贡献与许可证
- 欢迎提交 issue 或 PR。若准备公开，请补充 `LICENSE`（例如 MIT 或 Apache-2.0）。

