import numpy as np
import matplotlib.pyplot as plt
import cma
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf
import time

# ==========================================
# 1. 系统参数定义 (根据附录论文参数)
# ==========================================
w_FSR = 2 * np.pi * 0.079      # 自由光谱区 (GHz)
g1_max = 2 * np.pi * 0.047     # 最大耦合强度                                 
Lg = 0.2                       
Lw = 0.1                       
LT = 0.566                   
LJ = 8.34                      
Ln = 0.5 * 402 * 0.78                    
wq = 2 * np.pi * 5.809            # 比特频率 (GHz)
wn = 70 * w_FSR 
beta = (2 * Lg + Lw)/LT
g_c = -0.5 * np.sqrt((wq * wn) / ((Lg+LJ) * (Lg+Ln)))
alpha_shift = -np.sqrt((Lg+Ln) / (Lg+LJ))

# ==========================================
# 2. 物理关系映射 
# ==========================================
def eq_S24_solve_phi(phi_ext):
    phi_guess = np.linspace(0, 2*np.pi, 2000)
    pe = phi_guess + beta * np.sin(phi_guess)
    return np.interp(phi_ext, pe, phi_guess) 


def eq_S10_g1(phi):
    if isinstance(phi, (float, int)):
        return 0 if np.cos(phi) == 0 else g_c * Lg**2 / (2*Lg + Lw + LT/np.cos(phi))
    else:
        M = np.zeros_like(phi)
        for i in range(len(phi)):
            M[i] = 0 if np.cos(phi[i]) == 0 else Lg**2 / (2*Lg + Lw + LT/np.cos(phi[i]))
        return  M * g_c


def eq_S17_delta_w(g1): 
    return alpha_shift * g1


def eq_S23_Gamma(g1): 
    return 2 * np.pi * (g1**2) / w_FSR


_N_SIGMA = 3   # t_start / t_tail 留 N_σ 倍的余量，使边沿在端点处的导数 < e^{-25} ≈ 0

def eq_S26_control_pulse(t_arr, amp, sig_rise, sig_fall, width, t_start=None):
    # t_start 默认由 sig_rise 决定，确保 t=0 处 erf 尾巴可忽略：
    #   d(pulse)/dt|_{t=0} ∝ exp(-(t_start/sig_rise)²) < e^{-N²}
    if t_start is None:
        t_start = _N_SIGMA * sig_rise

    t_end = 2 * t_start + width + _N_SIGMA * sig_fall
    return (amp / 2.0) * (
        (erf((t_arr - t_start) / sig_rise) + erf(t_start / sig_rise)) -
        (erf((t_arr - t_end)   / sig_fall) + erf(t_end   / sig_fall))
    )

# ==========================================
# 3. 硬件畸变与薛定谔演化
# ==========================================
def apply_hardware_distortion(signal, dt=0.1):
    """
    物理硬件畸变模型（参考论文 Supplementary II）：

    ① Gaussian LPF（主要效应）
       论文明确标注控制线上装有 250 MHz 带宽 Gaussian 低通滤波器。
       频域 σ_f = 250 MHz → 时域 σ_t = 1/(2π·250 MHz) ≈ 0.637 ns。
       这解释了为何实验中适当平滑的脉冲比极端尖锐的脉冲更优：
       过于尖锐的脉冲被 LPF 截断后，到达样品的实际幅度大幅衰减，
       耦合器无法充分打开 → Γ_on 反而变小。

    ② Bias tee 高通（次要效应，默认关闭）
       混合腔级 cryogenic bias tee 的 AC 路径呈高通特性。
       若脉冲宽度与时间常数 τ 相当，脉冲被微分为边沿尖刺，
       进一步惩罚太尖锐的波形。可将 tau_hp_ns 设为有限值开启。
    """
    # ① Gaussian LPF: 250 MHz 带宽（论文给出）
    sigma_t = 1.0 / (2 * np.pi * 0.250)   # ≈ 0.637 ns（时域 σ）
    sigma_samp = sigma_t / dt              # ≈ 6.4 samples（dt=0.1 ns）
    filtered = gaussian_filter1d(signal.astype(float), sigma=sigma_samp)

    # ② Bias tee RC 高通（可选，解注释后生效）
    tau_hp_ns = 500.0   # ns，典型值 100–2000 ns，视实验系统调整
    alpha = dt / (tau_hp_ns + dt)
    hp = np.zeros_like(filtered)
    for i in range(1, len(filtered)):
        hp[i] = (1 - alpha) * (hp[i-1] + filtered[i] - filtered[i-1])
    return hp

    # return filtered


def simulate_single_T1_point(T_width, params):
    amp, sig_rise, sig_fall = params
    dt = 0.1
    # t_start / t_tail 与脉冲函数保持一致，确保时间窗口两端波形已充分平息
    t_start = _N_SIGMA * sig_rise
    t_tail  = _N_SIGMA * sig_fall
    t_arr = np.arange(0, 2 *t_start + T_width + 2 * t_tail + 20, dt)

    pulse = eq_S26_control_pulse(t_arr, amp, sig_rise, sig_fall, T_width,
                                 t_start=t_start)
    distorted_pulse = apply_hardware_distortion(pulse)
    
    phi_off = np.pi / 2.0
    phi_ext_off = phi_off + beta * np.sin(phi_off)

    phi_on = np.pi
    phi_ext_on = phi_on + beta * np.sin(phi_on)
    phi_ext_t = phi_ext_off + distorted_pulse * (phi_ext_on - phi_ext_off)
    
    phi_t = eq_S24_solve_phi(phi_ext_t)
    g1_t = eq_S10_g1(phi_t)
    delta_w_t = eq_S17_delta_w(g1_t)
    Gamma_t = eq_S23_Gamma(g1_t)
    
    if False:  # 设为 True 时开启调试绘图
        fig_debug, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(t_arr, phi_t, 'b-', linewidth=1.5)
        axes[0, 0].set_xlabel('Time (ns)', fontsize=11)
        axes[0, 0].set_ylabel('$\\phi(t)$ (rad)', fontsize=11)
        axes[0, 0].set_title('Flux Through SQUID Loop', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(t_arr, g1_t / (2*np.pi), 'g-', linewidth=1.5)
        axes[0, 1].set_xlabel('Time (ns)', fontsize=11)
        axes[0, 1].set_ylabel('$g_1(t) / 2\\pi$ (GHz)', fontsize=11)
        axes[0, 1].set_title('Coupler-Qubit Coupling Strength', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(t_arr, delta_w_t / (2*np.pi), 'orange', linewidth=1.5)
        axes[1, 0].set_xlabel('Time (ns)', fontsize=11)
        axes[1, 0].set_ylabel('$\\delta\\omega(t) / 2\\pi$ (GHz)', fontsize=11)
        axes[1, 0].set_title('AC Stark Shift', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(t_arr, Gamma_t, 'r-', linewidth=1.5)
        axes[1, 1].set_xlabel('Time (ns)', fontsize=11)
        axes[1, 1].set_ylabel('$\\Gamma(t)$ (rad/ns)', fontsize=11)
        axes[1, 1].set_title('Purcell Decay Rate', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 额外两张子图：原始脉冲与硬件畸变后脉冲
        fig_wave, (axw1, axw2) = plt.subplots(1, 2, figsize=(14, 4.5))

        axw1.plot(t_arr, pulse, color='tab:blue', linewidth=1.8)
        axw1.set_xlabel('Time (ns)', fontsize=11)
        axw1.set_ylabel('Amplitude (a.u.)', fontsize=11)
        axw1.set_title('Original Control Pulse', fontsize=12)
        axw1.grid(True, alpha=0.3)

        axw2.plot(t_arr, distorted_pulse, color='tab:red', linewidth=1.8)
        axw2.set_xlabel('Time (ns)', fontsize=11)
        axw2.set_ylabel('Amplitude (a.u.)', fontsize=11)
        axw2.set_title('Distorted Pulse (Hardware Filtered)', fontsize=12)
        axw2.grid(True, alpha=0.3)

        fig_wave.tight_layout()
        plt.tight_layout()
        # plt.savefig(f'debug_dynamics_T{T_width:.1f}.png', dpi=150)
        plt.show()
        
    integral_phase = np.cumsum(delta_w_t) * dt
    integral_Gamma = np.cumsum(Gamma_t) * dt
    sigma_1 = np.exp(-1j * integral_phase - 0.5 * integral_Gamma)
    
    return np.abs(sigma_1[-1])**2, Gamma_t, t_arr

# ==========================================
# 4. 实验 T1 扫掠及 CMA-ES 适应度函数
# ==========================================
T_sweep = np.linspace(0, 15, 50) # 扫 50个不同宽度的波形

def measure_T1(params):
    P1_results = []
    for T in T_sweep:
        P1, _, _ = simulate_single_T1_point(T, params)
        P1_results.append(P1)
    # 调试：绘制当前参数下的 P1 扫谱（避免每次都画图拖慢优化）
    if not hasattr(measure_T1, "_eval_count"):
        measure_T1._eval_count = 0
    measure_T1._eval_count += 1

    debug_every = 10  # 每 10 次评估绘制一次，可按需修改
    # if measure_T1._eval_count % debug_every == 1:
    #     p1_arr = np.array(P1_results)
    #     fig, ax = plt.subplots(figsize=(6, 4))
    #     ax.plot(T_sweep, p1_arr, "o-", linewidth=1.6)
    #     ax.set_xlabel("Pulse Width T (ns)")
    #     ax.set_ylabel("P1 (log scale)")
    #     ax.set_title(f"P1 Sweep Debug #{measure_T1._eval_count}\nparams={np.round(params, 3)}")
    #     ax.grid(True, which="both", ls="--", alpha=0.4)
    #     plt.tight_layout()
    #     plt.show()

    m, c = np.polyfit(T_sweep, np.log(np.array(P1_results) + 1e-12), 1)
    T1 = -1.0 / m if m < 0 else 1000.0
    time.sleep(0.1) # 模拟测量时间延迟
    return T1

# ==========================================
# 5. 执行优化与可视化
# ==========================================
if __name__ == "__main__":
    print("开始 CMA-ES 寻优...")
    x0 = [1, 2, 2] # 初始猜测
    sigma0 = 0.2
    bounds = [[0.7, 1, 1], [1.3, 15.0, 15.0]]
    
    options = {'bounds': bounds, 
               'popsize': 6, 
               'maxiter': 100, 
               'verb_disp': 1}
    es = cma.CMAEvolutionStrategy(x0, sigma0, options)
    es.optimize(measure_T1)
    best_params = es.result.xbest
    
    print("\n=== 优化结果 ===")
    print(f"优化前参数 (Amp, Sig_rise, Sig_fall): {x0}")
    print(f"优化后参数 (Amp, Sig_rise, Sig_fall): [{best_params[0]:.2f}, {best_params[1]:.2f}, {best_params[2]:.2f}]")
    
    # ---------------- 可视化代码 ----------------
    print("\n正在生成对比图表...")
    
    # 1. 获取扫掠数据计算 T1
    P1_init, P1_opt = [], []
    for T in T_sweep:
        p_i, _, _ = simulate_single_T1_point(T, x0)
        p_o, _, _ = simulate_single_T1_point(T, best_params)
        P1_init.append(p_i)
        P1_opt.append(p_o)

    m_init, _ = np.polyfit(T_sweep, np.log(P1_init), 1)
    m_opt, _ = np.polyfit(T_sweep, np.log(P1_opt), 1)
    T1_init = -1.0 / m_init
    T1_opt = -1.0 / m_opt

    # 2. 获取单个脉冲 (T=15ns) 的微观动力学
    _, Gamma_init, t_arr_15_init = simulate_single_T1_point(15.0, x0)
    _, Gamma_opt, t_arr_15_opt = simulate_single_T1_point(15.0, best_params)

    # 3. 开始绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- 子图 1: T1 衰减曲线 ---
    ax1.plot(T_sweep, P1_init, 'ko--', label=f'Initial Guess\n($T_1$ = {T1_init:.2f} ns)')
    ax1.plot(T_sweep, P1_opt, 'ro-', linewidth=2, label=f'CMA-ES Optimized\n($T_1$ = {T1_opt:.2f} ns)')
    # ax1.set_yscale('log')
    ax1.set_xlabel('Pulse Width $T$ (ns)', fontsize=12)
    ax1.set_ylabel('Measured Qubit Population $P_1$ (Log Scale)', fontsize=12)
    ax1.set_title('Macroscopic Measurement: $T_1$ Decay Curve', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # --- 子图 2: 初始波形 vs 优化后波形 ---
    T_show = 15
    pulse_init = eq_S26_control_pulse(t_arr_15_init, x0[0], x0[1], x0[2], T_show)
    pulse_opt = eq_S26_control_pulse(t_arr_15_opt, best_params[0], best_params[1], best_params[2], T_show)

    # 如需看硬件后的实际输出波形，使用畸变后的信号
    wave_init = apply_hardware_distortion(pulse_init)
    wave_opt = apply_hardware_distortion(pulse_opt)

    ax2.plot(t_arr_15_init, pulse_init, 'k--', label='Initial Waveform')
    ax2.plot(t_arr_15_opt, pulse_opt, 'r-', linewidth=2, label='Optimized Waveform')
    ax2.set_xlabel('Time (ns)', fontsize=12)
    ax2.set_ylabel('Purcell Decay Rate $\Gamma(t)$ (rad/ns)', fontsize=12)
    ax2.set_title('Microscopic Dynamics: Effective Coupler Opening', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()