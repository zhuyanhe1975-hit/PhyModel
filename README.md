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
