# gmon 耦合器控制脉冲黑箱优化说明文档

**文件清单**

| 文件 | 作用 |
|---|---|
| `bbo_gmon.py` | 核心优化代码（仿真 + CMA-ES） |
| `pulse_visualizer.html` | 脉冲波形与硬件畸变交互可视化 |
| `README.md` | 本文档 |

---

## 1. 参考文献

> **[REF]** *Supplementary information of  <Violating Bell’s inequality with remotely connected superconducting qubits>*


主要使用到的方程：

| 编号 | 物理含义 |
|---|---|
| S10 | 内磁通 $\delta$ → 耦合强度 $g_1(\delta)$ |
| S17 | 耦合强度 → AC Stark 频移 $\delta\omega$ |
| S23 | 耦合强度 → Purcell 衰减率 $\Gamma$ |
| S24 | 外磁通 $\delta_\text{ext}$ → 内磁通 $\delta$（隐式方程求逆） |
| S26 | 参数化控制脉冲波形（erf 方波） |

**CMA-ES 算法原始文献：**
> Hansen, N. (2016). *The CMA Evolution Strategy: A Tutorial*. arXiv:1604.00772.

---

## 2. 物理模型

### 2.1 系统结构

gmon 耦合器通过可调外磁通 $\delta_\text{ext}$ 控制超导量子比特之间的耦合强度。控制目标是：在耦合器 **ON 态**时，使 Purcell 渠道尽快将激发态布居数衰减至零（即最小化 $T_1$），以提升静态开关比。

```
比特 ──── gmon 耦合器 ────
              ↑
        外磁通控制线
          δ_ext(t)
```

### 2.2 耦合器物理参数（§1，由论文附录给出）

| 参数 | 符号 | 值 | 单位 | 物理含义 |
|---|---|---|---|---|
| 自由光谱区 | $\omega_\text{FSR}$ | $2\pi \times 0.079$ | GHz | 腔模间距 |
| 几何电感 | $L_g$ | 0.2 | nH | |
| 线路电感 | $L_w$ | 0.1 | nH | |
| 总电感 | $L_T$ | 0.566 | nH | |
| Josephson 电感 | $L_J$ | 8.34 | nH | |
| 谐振器电感 | $L_n$ | $0.5 \times 402 \times 0.78$ | nH | |
| 量子比特频率 | $\omega_q$ | $2\pi \times 5.809$ | GHz | |
| 谐振腔频率 | $\omega_n$ | $70 \times \omega_\text{FSR}$ | GHz | |
| 非线性系数 | $\beta$ | $(2L_g+L_w)/L_T$ | — | |
| 耦合系数 | $g_c$ | $-\frac{1}{2}\sqrt{\frac{\omega_q\omega_n}{(L_g+L_J)(L_g+L_n)}}$ | GHz | |
| Stark 系数 | $\alpha_\text{shift}$ | $-\sqrt{(L_g+L_n)/(L_g+L_J)}$ | — | |

### 2.3 关键方程链

#### Purcell 方程的形式

本代码采用的演化方程来源于**非马尔可夫 Purcell 效应**在时域的积分形式（等价于Eq. S20不考虑反射光子的改写版本）。其推导出发点是将量子比特–耦合器系统在 Born–Markov 近似下约化，得到比特激发态振幅 $\sigma_1(t)$ 满足的一阶方程：

$$\dot{\sigma}_1(t) = \left[-i \delta\omega(t) - \frac{\Gamma(t)}{2}\right]\sigma_1(t)$$

其中两个时变系数均由瞬时耦合强度 $g_1(t)$ 给出：

$$\delta\omega(t) = \alpha_\text{shift} \cdot g_1(t) \qquad \text{（AC Stark 频移，Eq. S17）}$$

$$\Gamma(t) = \frac{2\pi g_1(t)^2}{\omega_\text{FSR}} \qquad \text{（Purcell 衰减率，Eq. S23）}$$

**注：** $\Gamma(t)$ 的这一形式是 Fermi 黄金定则在密集腔模（自由光谱区 $\omega_\text{FSR}$）条件下的结果——比特通过耦合强度 $g_1$ 向腔模辐射，总速率正比于 $g_1^2$ 与态密度 $1/\omega_\text{FSR}$ 的乘积。当控制脉冲将耦合器切换至 ON 态时， $g_1(t)$ 显著增大， $\Gamma(t)$ 随之急剧上升，驱动快速 Purcell 衰减。

对上述一阶线性 ODE 直接积分，得到解析解：

$$\sigma_1(t) = \sigma_1(0)\exp \left(-i\int_0^t \delta\omega(t') dt' - \frac{1}{2}\int_0^t \Gamma(t') dt'\right)$$

取初始条件 $\sigma_1(0) = 1$（比特从 $|1\rangle$ 态出发），末态布居数为 $P_1(T) = |\sigma_1(T)|^2$。代码中用**累积求和（`np.cumsum`）乘以时间步 $\Delta t$** 近似两个积分，精度为 $O(\Delta t)$，在 $\Delta t = 0.1 \text{ns}$ 下对 ns 量级脉冲已足够准确。

**与常数 $T_1$ 的关系：** 若 $g_1$ 为常数（方波），则 $\Gamma$ 为常数，$P_1(T) = e^{-\Gamma T}$，对应指数衰减 $T_1 = 1/\Gamma$。对任意脉冲形状，$T_1$ 由 $\ln P_1(T)$ 对 $T$ 的线性拟合斜率提取（见 §2.4）。

---

控制信号 $\delta_\text{ext}(t)$ 经如下链路映射为 Purcell 衰减率：

```
pulse(t) [amp, σ_rise, σ_fall]
    ↓  (硬件畸变: LPF + Bias tee)
distorted(t)
    ↓  × (δ_ext,on − δ_ext,off) + δ_ext,off
δ_ext(t)                        [外磁通]
    ↓  eq_S24: 求逆 δ_ext = δ + β sin δ
δ(t)                            [内磁通]
    ↓  eq_S10
g1(t)                           [瞬时耦合强度]
    ↓  eq_S17          ↓  eq_S23
δω(t) [AC Stark 频移]  Γ(t) [Purcell 衰减率]
```

量子比特 $|1\rangle$ 态的相干演化（忽略纯退相干）：

$$\sigma_1(t) = \exp \left(-i\int_0^t \delta\omega(t') dt' - \frac{1}{2}\int_0^t \Gamma(t') dt'\right)$$

末态布居数： $P_1 = |\sigma_1(T_\text{end})|^2$

### 2.4 T1 提取方法

对一系列脉冲宽度 $T \in T_\text{sweep}$，分别测量末态 $P_1(T)$，对指数衰减拟合：

$$\ln P_1(T) \approx -\frac{T}{T_1} + C \implies T_1 = -\frac{1}{m}, \quad m = \text{polyfit slope}$$

$T_1$ 越小，表示耦合器 ON 态越强，开关比越高。

---

## 3. 控制脉冲设计

### 3.1 波形定义（Eq. S26，含零点修正）

理想参数化波形为高斯平滑方波（erf 卷积）：

$$\Phi_\text{ideal}(t) = \frac{A}{2}\left[\text{erf} \left(\frac{t-t_s}{\sigma_r}\right) - \text{erf} \left(\frac{t-t_e}{\sigma_f}\right)\right]$$

但直接使用时 $\Phi(0) \neq 0$（erf 函数在有限 $t_s$ 处有残余值）。**解析零点修正**后：

$$\Phi(t) = \frac{A}{2} \left[\left(\text{erf}\frac{t-t_s}{\sigma_r}+\text{erf}\frac{t_s}{\sigma_r}\right) - \left(\text{erf}\frac{t-t_e}{\sigma_f}+\text{erf}\frac{t_e}{\sigma_f}\right)\right]$$

严格保证 $\Phi(0) = 0$（无直流偏置跳变）。根据方程Eq.S26，在理想条件下，有
$$\delta_{\mathrm{ext}}(t)=(\pi-\delta_{\mathrm{off}})\Phi(t)+\delta_{\mathrm{off}}$$
若存在波形畸变（见3.3节），则 $\Phi(t)\to \tilde{\Phi}(t)$，相应的真实外加磁通也会畸变。

### 3.2 波形参数

| 参数 | 含义 | 当前设置 |
|---|---|---|
| $\sigma_r$ (`sig_rise`) | 上升沿 Gaussian 宽度 | 优化变量，范围 1–15 ns |
| $\sigma_f$ (`sig_fall`) | 下降沿 Gaussian 宽度 | 优化变量，范围 1–15 ns |
| $A$ (`amp`) | 脉冲幅度（归一化） | 优化变量，范围 0.7–1.3 |
| $t_s = N_\sigma \cdot \sigma_r$ | 上升沿开始时刻 | $N_\sigma = 3$（代码中 `_N_SIGMA`） |
| $t_e = 2t_s + W + N_\sigma \cdot \sigma_f$ | 下降沿结束时刻 | $W$ = 脉冲平顶宽度 |

选取 $N_\sigma = 3$ 保证 $t=0$ 处波形导数 $\propto e^{-9} \approx 10^{-4}$，可视为零。

### 3.3 硬件畸变模型

在仿真实验中，考虑了控制线上存在两种主要畸变（代码 `apply_hardware_distortion`）：

**① Gaussian 低通滤波器（250 MHz，主要效应）**

$$\tilde{\Phi}(\omega) = \Phi(\omega) \cdot \exp \left(-\frac{\omega^2}{2(2\pi \cdot 250 \text{MHz})^2}\right)$$

时域卷积宽度 $\sigma_t = 1/(2\pi \times 250 \text{MHz}) \approx 0.64 \text{ns}$。
该滤波器解释了实验中适度平滑脉冲优于极端尖锐脉冲的现象：过于尖锐的上升沿在经过 LPF 后幅度大幅衰减，实际到达样品的磁通偏小，耦合器无法充分打开。

**② Bias tee RC 高通（次要效应，时间常数 $\tau \approx 500 \text{ns}$）**

离散递推公式（因果滤波）：

$$h_i = (1-\alpha)(h_{i-1} + x_i - x_{i-1}), \quad \alpha = \frac{\Delta t}{\tau + \Delta t}$$

在脉冲宽度远小于 $\tau$ 时可忽略，但对 $T > 100 \text{ns}$ 的宽脉冲有拖尾效应。

注意在实际实验中，只需直接调用CMA-ES的黑箱优化算法，不需要知道实际噪声模型。

---

## 4. 代码架构

```
bbo_gmon.py
│
├── § 1  系统参数          全局物理常量
├── § 2  物理关系映射      eq_S10/S17/S23/S24 → 四个纯函数
├── § 3  控制脉冲          eq_S26_control_pulse()
├── § 4  硬件畸变模型      apply_hardware_distortion() ←── 仅仿真用
├── § 5  仿真演化          simulate_single_T1_point()  ←── 仅仿真用
│                          
│
├── § 6  Oracle 接口       ←── 仿真 / 实验解耦点
│        ├── SimulationOracle.measure()   调用 §5 仿真
│        └── ExperimentOracle             ←── 实验组实现此处
│               ├── make_pulse()          生成脉冲数组供 AWG 上传
│               └── measure()             [TODO] 仪器控制 + 返回 T1
│
├── § 7  CMA-ES 优化器     run_optimizer(measure_fn, ...)
│                          与 oracle 完全解耦，只调用 measure_fn
│
└── § 8  主程序            选择 oracle，调用 run_optimizer，可视化结果
```

**关键设计原则：** `run_optimizer` 只接受 `measure_fn(params) → T1`，不关心背后是仿真还是真实测量。切换模式只需修改主程序中的一行：

```python
oracle = SimulationOracle(T_sweep)       # 仿真模式
# oracle = MyExperimentOracle(T_sweep)   # 实验模式（取消注释）
```

---

## 5. CMA-ES 算法说明

### 5.1 基本原理

协方差矩阵自适应进化策略（CMA-ES）是一种无梯度黑箱优化算法，适用于：
- 目标函数不可微或噪声较大（实验测量）
- 参数维度较低（本问题为 3 维）
- 存在参数耦合关系

每轮迭代流程：

```
1. 从当前高斯分布采样 λ 个候选参数（种群大小 popsize=6）
2. 对每个候选参数调用 measure_fn，获取 T1
3. 取 T1 最小的前若干个更新分布均值
4. 根据成功方向更新协方差矩阵（自适应调节搜索形状）
5. 重复直到收敛或达到 maxiter
```

### 5.2 关键超参数

| 参数 | 默认值 | 含义 | 调整建议 |
|---|---|---|---|
| `x0` | `[1.0, 2.0, 2.0]` | 初始参数 `[amp, σ_rise, σ_fall]` | 填入实验中已知较优值 |
| `sigma0` | `0.2` | 初始搜索半径 | 约为参数范围的 10–20% |
| `bounds` | `[[0.7,1,1],[1.3,15,15]]` | 参数边界 | 根据系统实际范围调整 |
| `popsize` | `6` | 每轮评估次数 | 噪声大时增大（8–12） |
| `maxiter` | `100` | 最大迭代轮数 | 收敛慢时增大 |

### 5.3 评估次数估算

总实验测量次数 ≈ `popsize × maxiter × len(T_sweep)`
默认设置：$6 \times 100 \times 50 = 30{,}000$ 次单点测量。

实验时间紧张时可减小 `len(T_sweep)` 或 `maxiter`。若仅需粗优化，`maxiter=30` 通常已足够。

---

## 6. HTML 可视化工具（`pulse_visualizer.html`）

直接用浏览器打开 `pulse_visualizer.html`，无需安装任何依赖。

### 6.1 界面布局

```
┌─────────────────────────────────────────────────────┐
│  标题 + 脉冲公式显示                                  │
├────────────────────────────┬────────────────────────┤
│  画布 1：原始脉冲           │  脉冲参数控制面板        │
│  （erf 方波 + 零点修正）    │  ① amp    ∈ [-2, 2]    │
│                            │  ② σ_rise ∈ [0.1,15]ns │
│                            │  ③ σ_fall ∈ [0.1,15]ns │
│                            │  ④ N_σ    ∈ [1, 8]     │
│                            │  （显示派生量 t_start）  │
├────────────────────────────┴────────────────────────┤
│  [ 恢复初始值 ] 按钮                                  │
├────────────────────────────┬────────────────────────┤
│  画布 2：畸变后波形对比      │  畸变参数控制面板        │
│  — 原始（虚线）             │  ⑤ LPF 带宽 50–1000MHz │
│  — 经 Gaussian LPF（橙）   │  ⑥ Bias tee τ 10–2000ns│
│  — 经 LPF+Bias tee（绿）   │  （显示 σ_t 和 f_c）    │
└────────────────────────────┴────────────────────────┘
```

### 6.2 参数含义

| 滑块 | 范围 | 对应代码参数 | 说明 |
|---|---|---|---|
| amp | −2 ~ +2 | `amp` | 脉冲幅度；正值向上，负值向下 |
| σ_rise | 0.1–15 ns | `sig_rise` | 上升沿平滑度；越大越慢 |
| σ_fall | 0.1–15 ns | `sig_fall` | 下降沿平滑度；越大越慢 |
| N_σ | 1–8 | `_N_SIGMA` | t_start = N_σ × σ_rise；越大零点越干净 |
| BW_LPF | 50–1000 MHz | `lpf_bw_ghz` | 控制线低通带宽；250 MHz 为论文值 |
| τ_HP | 10–2000 ns | `bias_tee_tau_ns` | Bias tee 时间常数；越大高通截止越低 |

### 6.3 典型使用场景

- **调参前预览**：在正式优化前，拖动滑块直观感受各参数对波形形状的影响
- **畸变评估**：调节 BW_LPF 和 τ_HP，比对三条曲线，判断硬件畸变对脉冲形状的实际影响程度
- **边界设定参考**：通过可视化确认合理的 `sigma_rise`/`sigma_fall` 搜索范围，再填入 `bounds`

---

## 7. 实验组接入指南

实验组同学只需完成以下两步，无需修改其他任何代码。

### 步骤1：实现 ExperimentOracle

在 `bbo_gmon.py` 末尾（或单独文件中）继承 `ExperimentOracle`，实现 `measure()` 方法：

```python
class MyExperimentOracle(ExperimentOracle):
    """
    将 CMA-ES 建议的参数发送给实际仪器，返回实测 T1。
    """
    def measure(self, params):
        amp, sig_rise, sig_fall = params
        P1_list = []

        for T_width in self.T_sweep:
            # 1. 生成当前宽度的脉冲波形
            t_arr, pulse = self.make_pulse(T_width, params)

            # 2. 将波形上传至 AWG（根据你们的仪器驱动替换）
            #    awg.upload(t_arr, pulse)
            #    awg.run()

            # 3. 触发 T1 测量序列，读取末态布居数
            #    p1 = instruments.measure_p1()
            #    P1_list.append(p1)

            pass  # ← 删除此行，替换为真实仪器调用

        # 4. 线性拟合 log(P1) ~ T，提取 T1
        P1 = np.array(P1_list)
        m, _ = np.polyfit(self.T_sweep, np.log(P1 + 1e-12), 1)
        T1 = -1.0 / m if m < 0 else 1e6
        print(f"  params={np.round(params,3)}, T1={T1:.1f} ns")
        return T1
```

### 步骤 2：修改主程序中的 oracle 选择

在 `if __name__ == "__main__":` 中，注释掉仿真 oracle，启用实验 oracle：

```python
T_sweep = np.linspace(0, 15, 50)   # 调整测量点数以控制单轮耗时

# oracle = SimulationOracle(T_sweep)   # 仿真模式（关闭）
oracle = MyExperimentOracle(T_sweep)   # 实验模式（开启）

x0 = [1.0, 2.0, 2.0]   # 填入实验中已知较优的初始值

best_params = run_optimizer(
    oracle.measure,
    x0      = x0,
    sigma0  = 0.2,          # 初始搜索半径（参数范围的 ~10%）
    bounds  = [[0.7, 1.0, 1.0], [1.3, 15.0, 15.0]],
    popsize = 6,            # 每轮并行评估数；若噪声大可增至 8–10
    maxiter = 50,           # 实验中先用较少迭代验证流程
)
```

### 常见问题

**Q: `T_sweep` 应该怎么设置？**
A: 选取能覆盖 T1 量级的范围，通常 `np.linspace(0, T1_估计值 × 3, 30)` 即可。点数越少，单轮优化越快，但 T1 拟合精度越低。建议先用 20–30 点验证流程，稳定后增至 50 点。

**Q: CMA-ES 收敛太慢怎么办？**
A: (1) 检查 `sigma0` 是否合适（应约为参数搜索范围的 10–20%）；(2) 提供一个更接近最优的初始值 `x0`；(3) 减小搜索范围 `bounds`。

**Q: 测量噪声很大，优化不稳定？**
A: 增大 `popsize`（如从 6 增至 10），或在 `measure()` 中对每个参数点做 2–3 次重复测量取平均后再返回 T1。

**Q: 能否固定某个参数只优化其余两个？**
A: 将该参数从 `params` 中移出，在 `measure()` 内硬编码其值；同时相应缩减 `x0`、`bounds` 的维度。

**Q: 如何在优化过程中实时保存中间结果？**
A: 在 `measure()` 中将每次调用的 `params` 和返回的 `T1` 追加写入文件：
```python
with open("opt_log.csv", "a") as f:
    f.write(f"{amp},{sig_rise},{sig_fall},{T1}\n")
```

---

## 8. 快速上手检查清单

- [ ] 安装依赖：`pip install numpy scipy matplotlib cma`
- [ ] 用 `pulse_visualizer.html` 预览脉冲形状，确认边界范围合理
- [ ] 仿真模式运行一次（`SimulationOracle`），确认代码无误
- [ ] 实现 `MyExperimentOracle.measure()`，先用短 `maxiter=5` 做端到端测试
- [ ] 正式运行，监控输出的 `T1` 是否随迭代下降
- [ ] 优化结束后，用最优参数 `best_params` 在实验中验证开关比提升
