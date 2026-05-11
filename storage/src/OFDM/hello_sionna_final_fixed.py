"""
Sionna "Hello, World!" 示例 - 真正的最终版
使用英文标签避免中文字体warning
"""

import tensorflow as tf
import sionna
from sionna.phy.mapping import BinarySource, Mapper, Demapper
from sionna.phy.channel import AWGN
from sionna.phy.utils import ebnodb2no
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib使用英文
plt.rcParams['font.family'] = 'DejaVu Sans'

print("🚀 Sionna Hello World Example")
print("="*50)

# 设置参数
batch_size = 1000
num_bits_per_symbol = 4  # 16QAM
coderate = 1.0
num_bits = 12000

# 创建通信组件
source = BinarySource()
mapper = Mapper("qam", num_bits_per_symbol)
channel = AWGN()
demapper = Demapper("app", "qam", num_bits_per_symbol)

# 测试不同信噪比
ebno_dbs = np.arange(0, 11, 2)
bers = []

print("\n📊 BER at different SNR:")
print("-"*50)
for ebno_db in ebno_dbs:
    # 重新生成比特
    bits = source([batch_size, num_bits])
    x = mapper(bits)
    
    # 计算噪声功率
    no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate=coderate)
    
    # 通过信道
    y = channel(x, no)
    
    # 解调得到LLR
    llr = demapper(y, no)
    
    # 硬判决：LLR > 0 表示比特1，LLR < 0 表示比特0
    bits_hat = tf.cast(tf.greater(llr, 0), tf.float32)  # 注意这里改成 greater
    
    # 计算误码率
    errors = tf.reduce_sum(tf.cast(tf.not_equal(bits, bits_hat), tf.float32))
    total_bits = tf.cast(tf.size(bits), tf.float32)
    ber = errors / total_bits
    
    bers.append(ber.numpy())
    print(f"Eb/N0 = {ebno_db:2d} dB, BER = {ber.numpy():.2e}")

print("-"*50)

# 绘制结果
plt.figure(figsize=(15, 5))

# 图1：10 dB时的接收星座
plt.subplot(131)
bits_demo = source([1, num_bits])
x_demo = mapper(bits_demo)
no_demo = ebnodb2no(10, num_bits_per_symbol, coderate=coderate)
y_demo = channel(x_demo, no_demo)
plt.scatter(y_demo[0, :500].numpy().real, y_demo[0, :500].numpy().imag, 
            alpha=0.5, s=2, c='blue')
plt.grid(True, alpha=0.3)
plt.xlabel('In-phase (I)')
plt.ylabel('Quadrature (Q)')
plt.title('Received Constellation at 10 dB')
plt.axis('equal')

# 图2：误码率曲线
plt.subplot(132)
plt.semilogy(ebno_dbs, bers, 'b-o', linewidth=2, markersize=8, label='Simulation')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('16QAM Performance in AWGN Channel')
plt.legend()
plt.ylim([1e-4, 1])

# 图3：理论值对比
plt.subplot(133)
from scipy.special import erfc
theory_bers = []
for ebno_db in ebno_dbs:
    snr_linear = 10**(ebno_db/10)
    # 16QAM的理论误码率（近似）
    theory_ber = 3/4 * erfc(np.sqrt(0.4 * snr_linear))
    theory_bers.append(theory_ber)

plt.semilogy(ebno_dbs, bers, 'b-o', label='Simulation', linewidth=2, markersize=8)
plt.semilogy(ebno_dbs, theory_bers, 'r--s', label='Theory', linewidth=2, markersize=8)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.title('Simulation vs Theory')
plt.legend()
plt.ylim([1e-4, 1])

plt.tight_layout()
plt.savefig('sionna_final_result.png', dpi=150, bbox_inches='tight')
print("\n✅ Result plot saved to sionna_final_result.png")
print("🎉 Example completed successfully!")

# 显示文件位置
import os
print(f"\n📁 Image saved at: {os.path.abspath('sionna_final_result.png')}")
