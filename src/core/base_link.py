"""
Sionna 基础仿真链路类
所有通信链路的基类，统一收发流程
"""

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod


class BaseLink(ABC):
    """通信链路基类"""
    
    def __init__(self, config=None):
        """
        初始化链路
        
        Args:
            config: 配置字典，包含所有参数
        """
        self.config = config or {}
        self._init_params()
    
    def _init_params(self):
        """初始化参数"""
        self.batch_size = self.config.get('batch_size', 100)
        self.num_bits = self.config.get('num_bits', 10000)
        self.ebno_db = self.config.get('ebno_db', 10)
        self.device = self.config.get('device', 'cpu')
    
    @abstractmethod
    def build(self):
        """构建链路（子类必须实现）"""
        pass
    
    @abstractmethod
    def tx(self, bits):
        """发送端处理（子类必须实现）"""
        pass
    
    @abstractmethod
    def channel(self, x, no):
        """信道处理（子类必须实现）"""
        pass
    
    @abstractmethod
    def rx(self, y, no):
        """接收端处理（子类必须实现）"""
        pass
    
    def forward(self, bits, no=None):
        """
        完整前向传播
        
        Args:
            bits: 输入比特
            no: 噪声功率（可选）
        
        Returns:
            bits_hat: 恢复的比特
        """
        x = self.tx(bits)
        if no is None:
            no = self._compute_noise()
        y = self.channel(x, no)
        bits_hat = self.rx(y, no)
        return bits_hat
    
    def _compute_noise(self):
        """计算噪声功率"""
        from sionna.phy.utils import ebnodb2no
        coderate = self.config.get('coderate', 1.0)
        num_bits_per_symbol = self.config.get('num_bits_per_symbol', 2)
        return ebnodb2no(self.ebno_db, num_bits_per_symbol, coderate=coderate)
    
    def compute_ber(self, bits, bits_hat):
        """计算误码率"""
        errors = tf.reduce_sum(tf.cast(tf.not_equal(bits, bits_hat), tf.float32))
        total = tf.cast(tf.size(bits), tf.float32)
        return errors / total
    
    def run_simulation(self, ebno_dbs):
        """
        运行仿真，测试不同信噪比
        
        Args:
            ebno_dbs: 信噪比列表
        
        Returns:
            bers: 对应的误码率列表
        """
        bers = []
        for ebno_db in ebno_dbs:
            self.ebno_db = ebno_db
            bits = tf.random.uniform([self.batch_size, self.num_bits], 
                                      maxval=2, dtype=tf.float32)
            bits_hat = self.forward(bits)
            ber = self.compute_ber(bits, bits_hat)
            bers.append(ber.numpy())
        return bers
