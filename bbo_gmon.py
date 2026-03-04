#!/usr/bin/env python3
"""
bbo_gmon.py — CMA-ES 黑箱优化 gmon 耦合器静态开关比

优化变量：控制脉冲参数  params = [amp, sig_rise, sig_fall]
目标函数：coupler ON 态下的 T1（最小化 → 提高开关比）

切换仿真 / 实验模式：
    仿真模式  →  oracle = SimulationOracle(T_sweep)
    实验模式  →  oracle = ExperimentOracle(T_sweep)，实现 measure() 方法
"""

import numpy as np
import matplotlib.pyplot as plt
import cma
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf

# ============================================================
# § 1  系统参数（由论文附录给出）
# ============================================================
w_FSR       = 2 * np.pi * 0.079          # 自由光谱区 (GHz)
Lg, Lw, LT  = 0.2, 0.1, 0.566            # 各路电感 (nH)
LJ          = 8.34
Ln          = 0.5 * 402 * 0.78
wq          = 2 * np.pi * 5.809          # 比特频率 (GHz)
wn          = 70 * w_FSR
beta        = (2 * Lg + Lw) / LT
g_c         = -0.5 * np.sqrt((wq * wn) / ((Lg + LJ) * (Lg + Ln)))
alpha_shift = -np.sqrt((Lg + Ln) / (Lg + LJ))

# t_start = _N_SIGMA * sig_rise 保证 t=0 处导数 ∝ exp(-N²) ≈ 0
_N_SIGMA = 3

# ============================================================
# § 2  物理关系映射（论文方程 S10, S17, S23, S24）
# ============================================================
def eq_S24_solve_delta(delta_ext):
    """外磁通 → 内磁通（数值求逆 δ_ext = δ + β sin δ）"""
    delta_g = np.linspace(0, 2 * np.pi, 2000)
    return np.interp(delta_ext, delta_g + beta * np.sin(delta_g), delta_g)


def eq_S10_g1(delta):
    """内磁通 → 耦合强度 g1(δ)（方程 S10）"""
    delta     = np.atleast_1d(np.asarray(delta, dtype=float))
    cos_delta = np.cos(delta)
    safe    = np.where(cos_delta == 0, 1.0, cos_delta)   # 避免除零
    M       = np.where(cos_delta == 0, 0.0,
                       Lg**2 / (2 * Lg + Lw + LT / safe))
    return M * g_c


def eq_S17_dw(g1):
    """耦合强度 → AC Stark 频移（方程 S17）"""
    return alpha_shift * g1


def eq_S23_Gamma(g1):
    """耦合强度 → Purcell 衰减率（方程 S23）"""
    return 2 * np.pi * g1**2 / w_FSR


# ============================================================
# § 3  控制脉冲（论文方程 S26）
# ============================================================
def eq_S26_control_pulse(t_arr, amp, sig_rise, sig_fall, width, t_start=None):
    """
    erf 方波 + 解析零点修正：pulse(t=0) = 0。

    修正项使 erf 在 t=0 处的剩余值被精确抵消：
        (amp/2) * [(erf((t-ts)/sr) + erf(ts/sr))
                 - (erf((t-te)/sf) + erf(te/sf))]
    """
    if t_start is None:
        t_start = _N_SIGMA * sig_rise
    t_end = 2 * t_start + width + _N_SIGMA * sig_fall
    return (amp / 2.0) * (
        (erf((t_arr - t_start) / sig_rise) + erf(t_start / sig_rise)) -
        (erf((t_arr - t_end)   / sig_fall) + erf(t_end   / sig_fall))
    )


# ============================================================
# § 4  硬件畸变模型（250 MHz Gaussian LPF + Bias tee HP）
# ============================================================
def apply_hardware_distortion(signal, dt=0.1,
                               lpf_bw_ghz=0.250,
                               bias_tee_tau_ns=500.0):
    """
    ① Gaussian LPF（论文 Supplementary II，250 MHz 带宽）
       sigma_t = 1/(2π·BW) ≈ 0.637 ns，解释了适当平滑比极端尖锐更优的实验现象。
    ② Bias tee RC 高通（tau_ns → inf 时退化为全通，可关闭）
    """
    sigma_t  = 1.0 / (2 * np.pi * lpf_bw_ghz)
    filtered = gaussian_filter1d(signal.astype(float), sigma=sigma_t / dt)

    if np.isfinite(bias_tee_tau_ns) and bias_tee_tau_ns > 0:
        alpha = dt / (bias_tee_tau_ns + dt)
        hp    = np.zeros_like(filtered)
        for i in range(1, len(filtered)):
            hp[i] = (1 - alpha) * (hp[i - 1] + filtered[i] - filtered[i - 1])
        return hp
    return filtered

# ============================================================
# § 5  仿真：单点 Purcell 演化
# ============================================================
def simulate_single_T1_point(T_width, params, dt=0.1, debug=False):
    """
    给定脉冲宽度 T_width 和参数 [amp, sig_rise, sig_fall]，
    积分 Purcell 演化方程，返回 (P1_final, Gamma_t, t_arr)。

    debug=True 时绘制各中间量的时域图。
    """
    amp, sig_rise, sig_fall = params
    t_start = _N_SIGMA * sig_rise
    t_tail  = _N_SIGMA * sig_fall
    # 增加20ns余量，从而在仿真环境中确保所有动态都被捕获（尤其是长T1时的尾部衰减）
    t_arr   = np.arange(0, 2 * t_start + T_width + 2 * t_tail + 20, dt)

    pulse     = eq_S26_control_pulse(t_arr, amp, sig_rise, sig_fall, T_width,
                                     t_start=t_start)
    distorted = apply_hardware_distortion(pulse, dt=dt)

    delta_ext_off = np.pi / 2.0 + beta * np.sin(np.pi / 2.0) 
    delta_ext_on  = np.pi       + beta * np.sin(np.pi)
    delta_t       = eq_S24_solve_delta(delta_ext_off + distorted * (delta_ext_on - delta_ext_off)) # solve delta according to Eq.S24

    g1_t      = eq_S10_g1(delta_t)
    dw_t = eq_S17_dw(g1_t)
    Gamma_t   = eq_S23_Gamma(g1_t)

    sigma_1 = np.exp(
        -1j * np.cumsum(dw_t) * dt
        - 0.5 * np.cumsum(Gamma_t) * dt
    )

    if debug:
        _debug_plot(t_arr, delta_t, g1_t, dw_t, Gamma_t, pulse, distorted)

    return float(np.abs(sigma_1[-1])**2), Gamma_t, t_arr


def _debug_plot(t_arr, delta_t, g1_t, dw_t, Gamma_t, pulse, distorted):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    rows = [
        (delta_t,                    r'$\delta(t)$ (rad)',            'Flux Through SQUID'),
        (g1_t / (2 * np.pi),      r'$g_1/2\pi$ (GHz)',           'Coupling Strength'),
        (dw_t / (2 * np.pi), r'$\delta\omega/2\pi$ (GHz)',   'AC Stark Shift'),
        (Gamma_t,                  r'$\Gamma(t)$ (rad/ns)',        'Purcell Decay Rate'),
        (pulse,                    'Amplitude (a.u.)',             'Original Pulse'),
        (distorted,                'Amplitude (a.u.)',             'Distorted Pulse'),
    ]
    for ax, (y, ylabel, title) in zip(axes.flat, rows):
        ax.plot(t_arr, y)
        ax.set_xlabel('Time (ns)'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(alpha=0.3)
    plt.tight_layout() 
    plt.show()


# ============================================================
# § 6  Oracle 接口（仿真 / 实验解耦点）
# ============================================================
class SimulationOracle:
    """
    仿真 oracle：用 simulate_single_T1_point 代替真实测量。
    传入 run_optimizer() 的 measure 方法签名与 ExperimentOracle 完全一致。
    """
    def __init__(self, T_sweep=None, dt=0.1):
        self.dt = dt
        self.T_sweep = T_sweep if T_sweep is not None else np.linspace(0, 15, 50) # 测量T1时的脉冲宽度扫描范围

    def measure(self, params):
        """params = [amp, sig_rise, sig_fall]  →  T1 (ns)"""
        P1 = np.array([simulate_single_T1_point(T, params, dt=self.dt)[0] for T in self.T_sweep])
        m, _ = np.polyfit(self.T_sweep, np.log(P1 + 1e-12), 1)
        return -1.0 / m if m < 0 else 1e6


class ExperimentOracle:
    """
    实验 oracle 模板。

    实验人员只需在 measure() 中实现仪器控制逻辑，无需修改优化器。
    make_pulse() 提供与仿真完全一致的脉冲生成，供直接上传 AWG。

    用法示例：
        class MyOracle(ExperimentOracle):
            def measure(self, params):
                P1 = []
                for T_width in self.T_sweep:
                    t_arr, pulse = self.make_pulse(T_width, params)
                    upload_to_awg(t_arr, pulse)
                    P1.append(run_t1_experiment())
                m, _ = np.polyfit(self.T_sweep, np.log(P1), 1)
                return -1.0 / m if m < 0 else 1e6
    """
    def __init__(self, T_sweep=None, dt=0.1):
        self.T_sweep = T_sweep if T_sweep is not None else np.linspace(0, 15, 50) # 测量T1时的脉冲宽度扫描范围
        self.dt      = dt

    def make_pulse(self, T_width, params):
        """
        生成指定宽度的控制脉冲（与仿真使用相同时间轴）。

        Returns:
            t_arr  : np.ndarray，时间轴 (ns)
            pulse  : np.ndarray，脉冲幅度 (a.u.)
        """
        amp, sig_rise, sig_fall = params
        t_start = _N_SIGMA * sig_rise
        t_tail  = _N_SIGMA * sig_fall
        t_arr   = np.arange(0, 2 * t_start + T_width + 2 * t_tail, self.dt)
        pulse   = eq_S26_control_pulse(t_arr, amp, sig_rise, sig_fall, T_width,
                                       t_start=t_start)
        return t_arr, pulse

    def measure(self, params):
        """
        params = [amp, sig_rise, sig_fall]  →  T1 (ns)

        请在此处（或子类中）填写仪器控制代码：
            for T_width in self.T_sweep:
                t_arr, pulse = self.make_pulse(T_width, params)
                upload_to_awg(t_arr, pulse)
                P1.append(run_t1_experiment())
        """
        raise NotImplementedError("请实现 ExperimentOracle.measure() 中的仪器控制逻辑")


# ============================================================
# § 7  CMA-ES 优化器（oracle 无关）
# ============================================================
def run_optimizer(measure_fn,
                  x0      = None,
                  sigma0  = 0.2,
                  bounds  = None,
                  popsize = 6,
                  maxiter = 100):
    """
    CMA-ES 优化，最小化 measure_fn(params) → T1 (ns)。

    Args:
        measure_fn : callable([amp, sig_rise, sig_fall]) → float
                     传入 SimulationOracle().measure 或
                     ExperimentOracle 子类的 measure。
        x0         : 初始参数 [amp, sig_rise, sig_fall]，默认 [1.0, 2.0, 2.0]
        sigma0     : 初始步长（搜索半径），默认 0.2
        bounds     : [[下界×3], [上界×3]]
        popsize    : CMA-ES 种群大小
        maxiter    : 最大迭代次数

    Returns:
        best_params : np.ndarray，形状 (3,)
    """
    if x0 is None:
        x0 = [1.0, 2.0, 2.0]
    if bounds is None:
        bounds = [[0.7, 1.0, 1.0], [1.3, 15.0, 15.0]]

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'bounds':    bounds,
        'popsize':   popsize,
        'maxiter':   maxiter,
        'verb_disp': 1,
    })
    es.optimize(measure_fn)
    return es.result.xbest


# ============================================================
# § 8  主程序
# ============================================================
if __name__ == "__main__":
    T_sweep = np.linspace(0, 15, 50)

    # ---- 切换注释即可切换仿真 / 实验模式 ----
    oracle = SimulationOracle(T_sweep)
    # oracle = MyExperimentOracle(T_sweep)   # 实验模式

    x0 = [1.0, 2.0, 2.0] # 初始参数 [amp, sig_rise, sig_fall]

    print("开始 CMA-ES 寻优...")
    best_params = run_optimizer(oracle.measure, x0=x0)

    print("\n=== 优化结果 ===")
    print(f"初始参数  [amp, σ_rise, σ_fall] = {x0}")
    print(f"最优参数  [amp, σ_rise, σ_fall] = "
          f"[{best_params[0]:.3f}, {best_params[1]:.3f}, {best_params[2]:.3f}]")

    # ---- 可视化优化结果，以一个15ns的方波为例 ----
    P1_init = [simulate_single_T1_point(T, x0)[0]          for T in T_sweep]
    P1_opt  = [simulate_single_T1_point(T, best_params)[0] for T in T_sweep]
    m_i, _  = np.polyfit(T_sweep, np.log(P1_init), 1)
    m_o, _  = np.polyfit(T_sweep, np.log(P1_opt),  1)

    _, Gamma_init, t_i = simulate_single_T1_point(15.0, x0)
    _, Gamma_opt,  t_o = simulate_single_T1_point(15.0, best_params)

    pulse_init = eq_S26_control_pulse(t_i, *x0, 15.0)
    pulse_opt  = eq_S26_control_pulse(t_o, *best_params, 15.0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(T_sweep, P1_init, 'ko--', label=f'Initial  (T1={-1/m_i:.1f} ns)')
    ax1.plot(T_sweep, P1_opt,  'ro-',  label=f'Optimized (T1={-1/m_o:.1f} ns)')
    ax1.set_xlabel('Pulse Width T (ns)'); ax1.set_ylabel('P1')
    ax1.set_title('T1 Decay Curve'); ax1.legend(); ax1.grid(ls='--', alpha=0.4)

    ax2.plot(t_i, Gamma_init, 'k--', label='Initial')
    ax2.plot(t_o, Gamma_opt,  'r-',  label='Optimized')
    ax2.set_xlabel('Time (ns)'); ax2.set_ylabel(r'$\Gamma(t)$ (rad/ns)')
    ax2.set_title(r'Purcell Decay Rate $\Gamma(t)$'); ax2.legend(); ax2.grid(ls='--', alpha=0.4)

    ax3.plot(t_i, pulse_init, 'k--', label='Initial')
    ax3.plot(t_o, pulse_opt,  'r-',  label='Optimized')
    ax3.set_xlabel('Time (ns)'); ax3.set_ylabel('Amplitude (a.u.)')
    ax3.set_title('Control Pulse'); ax3.legend(); ax3.grid(ls='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()
