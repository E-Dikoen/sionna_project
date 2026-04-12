import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sionna as sn

# ====================
# 1. 系统参数配置
# ====================
batch_size = 10000       # 批处理大小，一次仿真大量样本
k_ldpc = 500             # LDPC 信息比特长度
n_ldpc = 1000            # LDPC 码字长度 (码率 = k/n = 0.5)
num_bits_per_symbol = 2  # QPSK 调制，每个符号承载 2 个比特
snr_db_range = np.arange(0, 11, 2) # 仿真信噪比范围 (dB)

# ====================
# 2. 初始化 Sionna 通信组件
# ====================
# 二进制信源：产生随机的 0/1 比特流
binary_source = sn.utils.BinarySource()

# LDPC 编码器：使用 5G 标准的 LDPC 码
ldpc_encoder = sn.fec.ldpc.encoding.LDPC5GEncoder(k_ldpc, n_ldpc)

# LDPC 译码器：与编码器对应
ldpc_decoder = sn.fec.ldpc.decoding.LDPC5GDecoder(ldpc_encoder, hard_out=True)

# QPSK 星座和映射器：将比特映射为复数符号
constellation = sn.mapping.Constellation("qam", num_bits_per_symbol)
mapper = sn.mapping.Mapper(constellation=constellation)

# 解调器：计算软信息（对数似然比 LLR）
demapper = sn.mapping.Demapper("app", constellation=constellation)

# AWGN 信道：加性高斯白噪声信道
channel = sn.channel.AWGN()

# ====================
# 3. 主仿真循环
# ====================
ber_results = [] # 用于存储每个信噪比下的误码率结果

print(f"开始仿真，LDPC 码率: {k_ldpc}/{n_ldpc}")

for snr_db in snr_db_range:
    # 3.1 生成随机信息比特
    info_bits = binary_source([batch_size, k_ldpc])
    
    # 3.2 LDPC 编码
    coded_bits = ldpc_encoder(info_bits)
    
    # 3.3 QPSK 调制
    modulated_symbols = mapper(coded_bits)
    
    # 3.4 通过 AWGN 信道
    no = sn.utils.ebnodb2no(snr_db, num_bits_per_symbol, coderate=k_ldpc/n_ldpc)
    received_symbols = channel([modulated_symbols, no])
    
    # 3.5 解调 (计算 LLR)
    llr = demapper([received_symbols, no])
    
    # 3.6 LDPC 译码
    decoded_bits = ldpc_decoder(llr)
    
    # 3.7 计算误码率 (BER)
    errors = tf.reduce_sum(tf.cast(tf.not_equal(info_bits, decoded_bits), tf.int32))
    ber = errors.numpy() / (batch_size * k_ldpc)
    ber_results.append(ber)
    
    print(f"SNR = {snr_db:2d} dB, BER = {ber:.4e}")

# ====================
# 4. 绘图展示结果
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