"""
5G LDPC编解码示例 - 展示信道编码的增益
"""

import tensorflow as tf
import sionna
from sionna.phy.mapping import BinarySource, Mapper, Demapper
from sionna.phy.channel import AWGN
from sionna.phy.utils import ebnodb2no
from sionna.phy.utils.metrics import compute_ber
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("5G LDPC 编解码示例")
print("="*60)

# 设置参数
batch_size = 100
k = 100          # 信息比特长度
n = 200          # 码字长度 (码率 = 1/2)
coderate = k/n   # 码率 = 0.5
num_bits_per_symbol = 2  # QPSK调制

print(f"\n📊 系统参数:")
print(f"   - 码率: {coderate}")
print(f"   - 信息比特长度: {k}")
print(f"   - 码字长度: {n}")
print(f"   - 调制方式: QPSK")

# 创建通信组件
source = BinarySource()
mapper = Mapper("qam", num_bits_per_symbol)
channel = AWGN()
demapper = Demapper("app", "qam", num_bits_per_symbol)

# 创建LDPC编解码器
encoder = LDPC5GEncoder(k, n)
decoder = LDPC5GDecoder(encoder, num_iter=20, hard_out=True)

print("\n🔐 LDPC编解码器创建成功")

# 测试不同信噪比
ebno_dbs = np.arange(0, 7, 1)
bers_coded = []
bers_uncoded = []

print("\n📊 测试不同信噪比下的误码率:")
print("-"*70)
print("Eb/N0(dB) | 未编码BER    | LDPC编码BER  | 编码增益")
print("-"*70)

for ebno_db in ebno_dbs:
    # 生成随机信息比特
    bits = source([batch_size, k])
    
    # 未编码系统（QPSK需要偶数个比特）
    if k % 2 != 0:
        bits_padded = tf.concat([bits, tf.zeros([batch_size, 1])], axis=-1)
    else:
        bits_padded = bits
    x_uncoded = mapper(bits_padded)
    
    # 编码系统
    coded_bits = encoder(bits)
    x_coded = mapper(coded_bits)
    
    # 计算噪声功率
    no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate=coderate)
    
    # 通过信道
    y_uncoded = channel(x_uncoded, no)
    y_coded = channel(x_coded, no)
    
    # 解调
    llr_uncoded = demapper(y_uncoded, no)
    llr_coded = demapper(y_coded, no)
    
    # 硬判决（未编码）
    bits_hat_uncoded_full = tf.cast(tf.greater(llr_uncoded, 0), tf.float32)
    bits_hat_uncoded = bits_hat_uncoded_full[:, :k]
    
    # LDPC解码
    bits_hat_coded = decoder(llr_coded)
    
    # 计算误码率
    ber_uncoded = compute_ber(bits, bits_hat_uncoded)
    ber_coded = compute_ber(bits, bits_hat_coded)
    
    bers_uncoded.append(ber_uncoded.numpy())
    bers_coded.append(ber_coded.numpy())
    
    gain = ber_uncoded.numpy() / ber_coded.numpy() if ber_coded.numpy() > 0 else float('inf')
    print(f"   {ebno_db:2d}     | {ber_uncoded.numpy():.2e}   | {ber_coded.numpy():.2e}     | {gain:6.1f}x")

print("-"*70)

# 绘制结果
plt.figure(figsize=(15, 5))

# 左图：BER曲线对比
plt.subplot(131)
plt.semilogy(ebno_dbs, bers_uncoded, 'r--o', linewidth=2, markersize=8, label='未编码')
plt.semilogy(ebno_dbs, bers_coded, 'b-o', linewidth=2, markersize=8, label='5G LDPC')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.title('LDPC编码 vs 未编码')
plt.legend()
plt.ylim([1e-5, 1])

# 中图：编码增益
plt.subplot(132)
gains = [u/c if c>0 else 1000 for u, c in zip(bers_uncoded, bers_coded)]
plt.bar(ebno_dbs, gains, width=0.7, color='skyblue', edgecolor='navy')
plt.grid(True, alpha=0.3)
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('编码增益 (倍数)')
plt.title('LDPC编码增益')
for i, (db, gain) in enumerate(zip(ebno_dbs, gains)):
    plt.text(db-0.2, gain+1, f'{gain:.1f}x', fontsize=9)

# 右图：理论对比
plt.subplot(133)
from scipy.special import erfc
ebno_dbs_fine = np.arange(0, 7, 0.1)
theory_bers = []
for ebno in ebno_dbs_fine:
    snr_linear = 10**(ebno/10)
    theory_ber = 0.5 * erfc(np.sqrt(snr_linear))
    theory_bers.append(theory_ber)

plt.semilogy(ebno_dbs_fine, theory_bers, 'g-', linewidth=2, label='QPSK理论值')
plt.semilogy(ebno_dbs, bers_uncoded, 'ro', markersize=6, label='未编码仿真')
plt.semilogy(ebno_dbs, bers_coded, 'bo', markersize=6, label='LDPC仿真')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.title('仿真 vs 理论')
plt.legend()
plt.ylim([1e-5, 1])

plt.tight_layout()
plt.savefig('ldpc_example_result.png', dpi=150, bbox_inches='tight')
print("\n✅ 结果图已保存到 ldpc_example_result.png")
print("🎉 第二个示例运行完成！")
