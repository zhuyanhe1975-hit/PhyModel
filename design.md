# PhyModel / ER15-1400 — Design

## 1. 设计原则

1. **逐步逼真（Progressive realism）**：从“可控、可解释、可调参”开始，逐步打开碰撞/接触/柔顺/柔性等复杂因素。
2. **模型与参数解耦**：结构几何（MJCF/网格）与可辨识参数（摩擦、阻尼、接触、载荷）分层管理，避免每次调参都改 MJCF。
3. **快速校验优先**：每个阶段都应有秒级/分钟级脚本验证基本正确性，避免“仿真看起来怪但不知道哪错了”。
4. **接口稳定**：上层（控制/优化/辨识）只依赖稳定 API；内部可迭代替换参数化与子模型实现。

## 2. 当前仓库骨架（As-is）

### 2.1 目录结构

- `models/`
  - `er15-1400.mjcf.xml`：主线 MuJoCo 模型；**当前将所有 geom 设置为 `contype=0/conaffinity=0`，并添加邻接 body exclude**，用于排除碰撞干扰。
  - `ER15-1400.urdf`：备用/对照。
  - `*.STL`：网格。
- `phymodel/`
  - `mjcf/validator.py`：MJCF 静态校验（XML、mesh 文件存在、joint range 合法性、惯量缺失提示等）。
  - `kinematics/mujoco_fk.py`：最小 FK（给定 `q` 输出 `body_name` 的位置）。
- `scripts/validate_mjcf.py`：命令行入口，输出 JSON 形式校验结果。

### 2.2 当前数据流

`scripts/validate_mjcf.py` → `phymodel.mjcf.validate_mjcf()` → 解析 `models/*.mjcf.xml` → 输出错误/警告/摘要。

FK（现为库函数）预期数据流：

加载 MJCF → `mujoco.MjModel/MjData` → `MujocoFK.tcp_from_q(q)` → `mj_forward()` → 读取 `xpos`。

## 3. 目标架构（To-be）

> 下面模块是建议的演进方向；不要求一次性实现，可按 `tasks.md` 分阶段落地。

### 3.1 核心包分层

- `phymodel/model/`：模型加载与组装
  - 从 MJCF 加载 MuJoCo 模型（`MjModel/MjData`）
  - 应用参数集（见下）
  - 提供统一的访问点（关节/执行器/site 名称映射）
- `phymodel/params/`：参数集与版本管理
  - 参数结构定义：关节摩擦/阻尼、接触参数、载荷、柔顺等
  - 参数存储：建议 `params/*.yaml` 或 `params/*.json`（可与模型版本绑定）
  - 参数注入：修改 MJCF（生成临时 MJCF）或使用 MuJoCo API 在运行时写入 model 字段
- `phymodel/kinematics/`：运动学
  - FK：body/site/geom frame
  - （可选）IK：用于标定与工况设置（短期可不做）
- `phymodel/dynamics/`：动力学与执行器接口
  - 支持 torque 控制与（可选）位置/速度伺服器（通过 actuator 或外部控制律）
  - 提供可复现实验的摩擦/阻尼/armature 等项
- `phymodel/contact/`：接触/碰撞策略
  - 调试模式（无碰撞）与真实模式（选择性碰撞）
  - 自碰/邻接碰撞过滤清单与可视化/日志输出
  - 接触参数的场景化配置（地面、工件、link 碰撞）
- `phymodel/validation/`：一致性与回归测试
  - MJCF 静态检查增强
  - 动力学 sanity check（自由落体、能量衰减、关节限位等）
  - 标定实验回放与误差度量（轨迹/力矩/末端误差）

### 3.2 参数集（Parameter Sets）

建议引入 “参数集” 概念，至少包含：

- `mode`：`debug_no_contact` / `contact_realistic` / `id_calibration` 等
- `robot`：关节阻尼、摩擦（coulomb/viscous）、armature、frictionloss 等
- `contact`：摩擦系数、solref/solimp、bounce/restitution、接触对（pair）覆盖
- `payload`：工具/工件质量、质心、惯量、安装位姿
- `compliance`：关节弹簧-阻尼、间隙近似参数（阶段性启用）

注：MuJoCo 对许多参数既可在 MJCF 里声明，也可在加载后通过 `MjModel` 字段直接写入；推荐优先选择**可回归与可追踪**的方式（例如生成临时 MJCF 并记录差异），并为每次仿真输出参数快照。

### 3.3 模型文件组织（建议）

- `models/er15-1400/`
  - `base.mjcf.xml`（结构/几何/惯量）
  - `variants/`（可选：碰撞 geom、不同工具法兰等）
  - `assets/`（STL 与纹理等）
- 根目录保留 `models/er15-1400.mjcf.xml` 作为入口文件，内部通过 `<include>` 或构建脚本拼装（视项目偏好）。

## 4. 关键约定（Conventions）

- 单位：m、kg、s、rad、N、N·m。
- 命名：
  - link：`link_1..link_6`（已使用）
  - joint：`joint_1..joint_6`（已使用）
  - site：建议增加 `tcp`、`flange`、`base` 等（后续用于 FK/标定/控制引用）
- 模式开关：碰撞/接触/柔顺应可在不改上层代码的前提下切换（通过参数集与构建流程实现）。

## 5. 脚本入口（CLI）

建议最终形成一组“短路径”入口，覆盖日常迭代：

- `scripts/validate_mjcf.py`：静态校验（已存在）
- `scripts/check_fk.py`：给定 `--q` 输出 `--body/--site` 的位姿
- `scripts/sim_step.py`：加载参数集并跑短仿真（用于检查稳定性/能量/接触）
- `scripts/replay_log.py`：回放标定/实机日志并输出误差指标（后续）

