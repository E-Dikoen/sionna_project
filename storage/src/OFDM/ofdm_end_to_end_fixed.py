"""
OFDM 端到端链路实现（修复版）
验证无噪声场景下发送与接收比特一致性
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("OFDM 端到端链路实现（修复版）")
print("="*70)

# ============================================
# 参数配置
# ============================================
print("\n1. 配置系统参数...")

fft_size = 64
num_ofdm_symbols = 14
cp_length = 16
num_guard_left = 5
num_guard_right = 5
num_bits_per_symbol = 2

num_data_subcarriers = fft_size - num_guard_left - num_guard_right - 1
num_data_per_symbol = num_data_subcarriers
total_data_symbols = num_ofdm_symbols * num_data_per_symbol
total_bits = total_data_symbols * num_bits_per_symbol

print(f"   FFT大小: {fft_size}")
print(f"   OFDM符号数: {num_ofdm_symbols}")
print(f"   循环前缀长度: {cp_length}")
print(f"   有效子载波数: {num_data_subcarriers}")
print(f"   总比特数: {total_bits}")

# ============================================
# 生成随机比特
# ============================================
print("\n2. 生成随机比特...")

np.random.seed(42)
bits = np.random.randint(0, 2, total_bits)
print(f"   前20个比特: {bits[:20]}")

# ============================================
# QPSK 调制（明确映射关系）
# ============================================
print("\n3. QPSK 调制...")

def qpsk_modulate(bits_pair):
    """将2比特映射为QPSK符号"""
    b0, b1 = bits_pair[0], bits_pair[1]
    if b0 == 0 and b1 == 0:
        return 1 + 1j      # 00 → 第一象限
    elif b0 == 0 and b1 == 1:
        return -1 + 1j     # 01 → 第二象限
    elif b0 == 1 and b1 == 0:
        return 1 - 1j      # 10 → 第四象限
    else:
        return -1 - 1j     # 11 → 第三象限

# 重塑并调制
bits_reshaped = bits.reshape(-1, num_bits_per_symbol)
qam_symbols = np.array([qpsk_modulate(b) for b in bits_reshaped], dtype=complex)
print(f"   前5个QAM符号: {qam_symbols[:5]}")

# ============================================
# 资源网格映射
# ============================================
print("\n4. 资源网格映射...")

resource_grid = np.zeros((num_ofdm_symbols, fft_size), dtype=complex)
data_subcarrier_idx = list(range(num_guard_left + 1, fft_size - num_guard_right))

qam_reshaped = qam_symbols.reshape(num_ofdm_symbols, num_data_per_symbol)

for i in range(num_ofdm_symbols):
    for j, sc_idx in enumerate(data_subcarrier_idx):
        resource_grid[i, sc_idx] = qam_reshaped[i, j]

# ============================================
# IFFT
# ============================================
print("\n5. IFFT 变换...")

time_domain = np.zeros((num_ofdm_symbols, fft_size), dtype=complex)
for i in range(num_ofdm_symbols):
    time_domain[i] = np.fft.ifft(resource_grid[i]) * np.sqrt(fft_size)

# ============================================
# 添加CP
# ============================================
print("\n6. 添加循环前缀...")

tx_signal = []
for i in range(num_ofdm_symbols):
    cp = time_domain[i, -cp_length:]
    tx_signal.extend(cp)
    tx_signal.extend(time_domain[i])
tx_signal = np.array(tx_signal)

# ============================================
# 信道（无噪声）
# ============================================
print("\n7. 通过信道（无噪声）...")
rx_signal = tx_signal.copy()

# ============================================
# 去CP
# ============================================
print("\n8. 去除循环前缀...")

rx_time_domain = []
for i in range(num_ofdm_symbols):
    start = i * (fft_size + cp_length) + cp_length
    end = start + fft_size
    rx_time_domain.append(rx_signal[start:end])
rx_time_domain = np.array(rx_time_domain)

# ============================================
# FFT
# ============================================
print("\n9. FFT 变换...")

rx_freq_domain = np.zeros((num_ofdm_symbols, fft_size), dtype=complex)
for i in range(num_ofdm_symbols):
    rx_freq_domain[i] = np.fft.fft(rx_time_domain[i]) / np.sqrt(fft_size)

# ============================================
# 资源解映射
# ============================================
print("\n10. 资源网格解映射...")

rx_qam_symbols = []
for i in range(num_ofdm_symbols):
    for sc_idx in data_subcarrier_idx:
        rx_qam_symbols.append(rx_freq_domain[i, sc_idx])
rx_qam_symbols = np.array(rx_qam_symbols)

# ============================================
# QPSK 解调（与调制严格对应）
# ============================================
print("\n11. QPSK 解调...")

def qpsk_demodulate(symbol):
    """将QPSK符号解调为2比特（与调制映射严格对应）"""
    if symbol.real > 0 and symbol.imag > 0:
        return np.array([0, 0])   # 对应 1+1j
    elif symbol.real < 0 and symbol.imag > 0:
        return np.array([0, 1])   # 对应 -1+1j
    elif symbol.real > 0 and symbol.imag < 0:
        return np.array([1, 0])   # 对应 1-1j
    else:
        return np.array([1, 1])   # 对应 -1-1j

rx_bits = np.concatenate([qpsk_demodulate(sym) for sym in rx_qam_symbols])
print(f"   恢复比特数: {len(rx_bits)}")

# ============================================
# 验证
# ============================================
print("\n12. 验证比特一致性...")

errors = np.sum(bits != rx_bits[:len(bits)])
ber = errors / len(bits)

print(f"   总比特数: {len(bits)}")
print(f"   错误比特数: {errors}")
print(f"   误码率 (BER): {ber:.2e}")

if ber == 0:
    print("\n   🎉 完美！发送比特与接收比特完全一致！")
else:
    print(f"\n   ⚠️ 发现 {errors} 个错误比特")

# ============================================
# 可视化
# ============================================
print("\n13. 生成可视化结果...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 发送星座图
ax = axes[0, 0]
ax.scatter(np.real(qam_symbols), np.imag(qam_symbols), alpha=0.5, s=10, c='blue')
ax.grid(True, alpha=0.3)
ax.set_xlabel('实部')
ax.set_ylabel('虚部')
ax.set_title('发送星座图 (QPSK)')
ax.axis('equal')

# 接收星座图
ax = axes[0, 1]
ax.scatter(np.real(rx_qam_symbols), np.imag(rx_qam_symbols), alpha=0.5, s=10, c='red')
ax.grid(True, alpha=0.3)
ax.set_xlabel('实部')
ax.set_ylabel('虚部')
ax.set_title('接收星座图 (无噪声)')
ax.axis('equal')

# 资源网格
ax = axes[0, 2]
im = ax.imshow(np.abs(resource_grid), aspect='auto', cmap='hot', interpolation='nearest')
ax.set_xlabel('OFDM符号索引')
ax.set_ylabel('子载波索引')
ax.set_title('资源网格幅度')
plt.colorbar(im, ax=ax)

# 时域信号
ax = axes[1, 0]
ax.plot(np.real(tx_signal[:500]), linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.set_xlabel('采样点')
ax.set_ylabel('幅度')
ax.set_title('发送时域信号 (前500点)')

# 频域信号
ax = axes[1, 1]
freq_mag = np.abs(rx_freq_domain[0])
ax.plot(freq_mag, linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.set_xlabel('子载波索引')
ax.set_ylabel('幅度')
ax.set_title('接收频域信号')
ax.axvspan(0, num_guard_left-1, alpha=0.3, color='gray', label='保护带')
ax.axvspan(fft_size-num_guard_right, fft_size-1, alpha=0.3, color='gray')
ax.axvline(num_guard_left, color='r', linestyle='--', alpha=0.5, label='DC子载波')
ax.legend()

# 比特对比
ax = axes[1, 2]
ax.plot(bits[:200], 'b-', linewidth=0.5, label='发送比特', alpha=0.7)
ax.plot(rx_bits[:200], 'r--', linewidth=0.5, label='接收比特', alpha=0.7)
ax.grid(True, alpha=0.3)
ax.set_xlabel('比特索引')
ax.set_ylabel('比特值')
ax.set_title('比特对比 (前200个)')
ax.legend()
ax.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig('ofdm_end_to_end_result_fixed.png', dpi=150)
print("\n✅ 结果图已保存到 ofdm_end_to_end_result_fixed.png")

print("\n" + "="*70)
print("链路总结")
print("="*70)
print(f"   输入比特数: {len(bits)}")
print(f"   输出比特数: {len(rx_bits)}")
print(f"   误码率: {ber:.2e}")
print(f"   是否完全恢复: {'是' if ber == 0 else '否'}")
print("="*70)
print("\n🎉 OFDM 端到端链路验证完成！")
