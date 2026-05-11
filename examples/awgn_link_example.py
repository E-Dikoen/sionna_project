"""
AWGN 链路示例 - 使用 BaseLink 框架
"""

import sys
sys.path.append('..')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from src.core.base_link import BaseLink
from src.utils.config import get_default_config


class AWGNLink(BaseLink):
    """AWGN 信道链路（16QAM）"""
    
    def build(self):
        """构建链路组件"""
        from sionna.phy.mapping import Mapper, Demapper
        from sionna.phy.utils import BinarySource
        from sionna.phy.channel import AWGN
        
        num_bits_per_symbol = self.config.get('modulation.num_bits_per_symbol', 4)
        
        self.source = BinarySource()
        self.mapper = Mapper("qam", num_bits_per_symbol)
        self.channel_model = AWGN()
        self.demapper = Demapper("app", "qam", num_bits_per_symbol)
    
    def tx(self, bits):
        """发送端"""
        return self.mapper(bits)
    
    def channel(self, x, no):
        """信道"""
        return self.channel_model([x, no])
    
    def rx(self, y, no):
        """接收端"""
        llr = self.demapper([y, no])
        return tf.cast(tf.less(llr, 0), tf.float32)


def main():
    # 加载配置
    config = get_default_config()
    
    # 创建链路
    link = AWGNLink(config.to_dict())
    link.build()
    
    # 设置参数
    link.batch_size = 1000
    link.num_bits = 12000
    
    # 测试不同信噪比
    ebno_dbs = [0, 2, 4, 6, 8, 10]
    bers = link.run_simulation(ebno_dbs)
    
    # 打印结果
    print("="*50)
    print("AWGN 链路仿真结果")
    print("="*50)
    for ebno, ber in zip(ebno_dbs, bers):
        print(f"Eb/N0 = {ebno:2d} dB, BER = {ber:.2e}")
    
    # 绘图
    plt.figure(figsize=(8, 6))
    plt.semilogy(ebno_dbs, bers, 'b-o', linewidth=2)
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('16QAM in AWGN Channel')
    plt.savefig('awgn_link_result.png')
    print("\n✅ 结果图已保存到 awgn_link_result.png")


if __name__ == "__main__":
    main()
