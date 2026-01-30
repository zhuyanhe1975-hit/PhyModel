# PhyModel (ER15 World/Physics Model)

目标：为 ER15-1400 建立足够真实的物理特性与刚柔动力学模型（MuJoCo），用于后续控制/优化/数字孪生。

本项目从 PerfOpt_v0 迁移而来，仅保留“世界模型建立”必需内容（模型文件 + MJCF 校验 + 基础 FK 校验），避免携带优化器与控制器。

## 目录

- `models/`：ER15 MJCF/URDF/STL
- `phymodel/mjcf/validator.py`：MJCF 校验与摘要输出
- `phymodel/kinematics/`：最小 FK 校验（确认 TCP/site 刷新正确）
- `scripts/validate_mjcf.py`：命令行校验入口

## 快速开始

```bash
python3 scripts/validate_mjcf.py --mjcf models/er15-1400.mjcf.xml
```

## 统一参数文件（MJCF + 可调参）

本项目将“机器人物理/动力学参数”集中在一个文件里：`params/er15-1400.params.json`，其中：

- `mjcf_snapshot`：从 `models/er15-1400.mjcf.xml` 自动提取的结构/惯量/关节等快照
- `tunable`：可调参（当前重点是摩擦；也包含仿真 timestep override）

修改 MJCF 后保持同步：

```bash
python3 scripts/sync_params.py --mjcf models/er15-1400.mjcf.xml --out params/er15-1400.params.json
python3 scripts/validate_mjcf.py --mjcf models/er15-1400.mjcf.xml --params params/er15-1400.params.json
```

## 摩擦力模型：快速看到效应

无 GUI（导出曲线与数据到 `artifacts/`）：

```bash
python3 scripts/run_friction_demo.py --mjcf models/er15-1400.mjcf.xml --model lugre --compare
```

一个脚本覆盖多种“最能体现摩擦”的情形（通过 `--excitation` 切换）：

```bash
# 1) 起动摩擦/静摩擦（breakaway）：力矩双向缓慢爬升
python3 scripts/run_friction_demo.py --model lugre --compare --excitation ramp_torque --amp 20 --period 4 --hold 0.5

# 2) 零速附近往复：三角速度参考（过零时滞回最明显）
python3 scripts/run_friction_demo.py --model lugre --compare --excitation tri_vel --vmax 0.3 --freq 0.5

# 额外：隔离单关节（锁定 2~6 关节为 0 位），只观察关节 1 的摩擦效应
python3 scripts/run_friction_demo.py --model lugre --compare --excitation tri_vel --joint 1 --lock-others

# 3) 速度扫描（类 Stribeck）：速度参考分段扫到 +vmax 再到 -vmax
python3 scripts/run_friction_demo.py --model lugre --compare --excitation vel_sweep --vmax 0.6 --duration 6
```

有 GUI（MuJoCo viewer，适合直观看差异）：

```bash
python3 scripts/view_friction.py --mjcf models/er15-1400.mjcf.xml --model lugre
```

## 开发环境（本机约定）

本仓库对应的 MuJoCo Python 环境已在 Miniconda 中准备好：`mjwarp_env`。

```bash
conda activate mjwarp_env
python3 scripts/validate_mjcf.py --mjcf models/er15-1400.mjcf.xml
```

## 下一步（建议）

1) 接触/自碰/相邻碰撞过滤：从“可控无自碰”开始，逐步回归真实。
2) 摩擦模型：粘滑、库伦+粘性、Stribeck（按关节/减速器特性）。
3) 柔顺/结构振动：关节弹性、传动间隙、末端柔性（必要时做降阶模型）。
4) 载荷与惯量：工具/工件参数化（便于工况切换与辨识）。
