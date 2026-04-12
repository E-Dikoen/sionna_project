import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sionna as sn

# ====================
# 1. 系统参数配置
# ====================
batch_size = 10000       # 批处理大小，一次仿真大量样本
k_ldpc = 500             # LDPC 信息比特长度
n_ldpc = 1000            # LDPC 码字长度 (码率 = 0.5)
num_bits_per_symbol = 2  # QPSK 调制，每个符号 2 比特
snr_db_range = np.arange(0, 11, 2) # 信噪比范围 0dB 到 10dB

# ====================
# 2. 初始化 Sionna 组件
# ====================
binary_source = sn.utils.BinarySource() # 信源
ldpc_encoder = sn.fec.ldpc.encoding.LDPC5GEncoder(k_ldpc, n_ldpc) # 编码器
ldpc_decoder = sn.fec.ldpc.decoding.LDPC5GDecoder(ldpc_encoder, hard_out=True) # 译码器
constellation = sn.mapping.Constellation("qam", num_bits_per_symbol) # 星座图
mapper = sn.mapping.Mapper(constellation=constellation) # 映射器
demapper = sn.mapping.Demapper("app", constellation=constellation) # 解调器
channel = sn.channel.AWGN() # AWGN 信道

# ====================
# 3. 主仿真循环
# ====================
ber_results = []
print(f"开始仿真，LDPC 码率: {k_ldpc}/{n_ldpc}")

for snr_db in snr_db_range:
    # 生成数据
    info_bits = binary_source([batch_size, k_ldpc])
    # 编码
    coded_bits = ldpc_encoder(info_bits)
    # 调制
    modulated_symbols = mapper(coded_bits)
    # 计算噪声功率
    no = sn.utils.ebnodb2no(snr_db, num_bits_per_symbol, coderate=k_ldpc/n_ldpc)
    # 通过信道
    received_symbols = channel([modulated_symbols, no])
    # 解调 (计算 LLR)
    llr = demapper([received_symbols, no])
    # 译码
    decoded_bits = ldpc_decoder(llr)
    # 计算误码率
    errors = tf.reduce_sum(tf.cast(tf.not_equal(info_bits, decoded_bits), tf.int32))
    ber = errors.numpy() / (batch_size * k_ldpc)
    ber_results.append(ber)
    print(f"SNR = {snr_db:2d} dB, BER = {ber:.4e}")

# ====================
# 4. 绘图
# ====================
plt.figure(figsize=(8, 6))
plt.semilogy(snr_db_range, ber_results, 'b-o', linewidth=1.5, markersize=8, label='Sionna 仿真 (LDPC+QPSK)')
plt.grid(True, which="both", ls="-")
plt.legend()
plt.xlabel('信噪比 SNR (dB)')
plt.ylabel('误码率 BER')
plt.title('基于 Sionna 的 LDPC+QPSK 系统性能')
plt.ylim(1e-5, 1)
plt.show()