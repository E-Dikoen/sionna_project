"""
SISO Link Simulation - Full Comparison
Comparison: Uncoded BPSK, Uncoded QPSK, LDPC-coded QPSK
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sionna as sn
from scipy.special import erfc

# Set matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("SISO Link Simulation: Uncoded BPSK vs Uncoded QPSK vs LDPC-coded QPSK")
print("=" * 70)

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU available: {gpus[0]}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("Using CPU")

# ============================================
# 1. Uncoded BPSK Link
# ============================================
class UncodedBPSKLink:
    def __init__(self):
        self.num_bits_per_symbol = 1
        # BPSK uses 'pam' modulation
        self.constellation = sn.mapping.Constellation('pam', self.num_bits_per_symbol)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper('app', constellation=self.constellation)
        self.channel = sn.channel.AWGN()
    
    def call(self, batch_size, num_bits, ebno_db):
        source = sn.utils.BinarySource()
        bits = source([batch_size, num_bits])
        symbols = self.mapper(bits)
        no = sn.utils.ebnodb2no(ebno_db, 
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        received = self.channel([symbols, no])
        llr = self.demapper([received, no])
        bits_hat = tf.cast(llr > 0, tf.float32)
        return bits, bits_hat

# ============================================
# 2. Uncoded QPSK Link
# ============================================
class UncodedQPSKLink:
    def __init__(self):
        self.num_bits_per_symbol = 2
        self.mapper = sn.mapping.Mapper('qam', self.num_bits_per_symbol)
        self.demapper = sn.mapping.Demapper('app', constellation_type='qam', 
                                            num_bits_per_symbol=self.num_bits_per_symbol)
        self.channel = sn.channel.AWGN()
    
    def call(self, batch_size, num_bits, ebno_db):
        source = sn.utils.BinarySource()
        bits = source([batch_size, num_bits])
        symbols = self.mapper(bits)
        no = sn.utils.ebnodb2no(ebno_db, 
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        received = self.channel([symbols, no])
        llr = self.demapper([received, no])
        bits_hat = tf.cast(llr > 0, tf.float32)
        return bits, bits_hat

# ============================================
# 3. LDPC-coded QPSK Link
# ============================================
class LDPCCodedQPSKLink:
    def __init__(self, k=100, n=200):
        self.k = k
        self.n = n
        self.num_bits_per_symbol = 2
        self.code_rate = k / n
        self.encoder = sn.fec.ldpc.LDPC5GEncoder(k, n)
        self.decoder = sn.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True)
        self.mapper = sn.mapping.Mapper('qam', self.num_bits_per_symbol)
        self.demapper = sn.mapping.Demapper('app', constellation_type='qam', 
                                            num_bits_per_symbol=self.num_bits_per_symbol)
        self.channel = sn.channel.AWGN()
    
    def call(self, batch_size, ebno_db):
        source = sn.utils.BinarySource()
        bits = source([batch_size, self.k])
        coded_bits = self.encoder(bits)
        symbols = self.mapper(coded_bits)
        no = sn.utils.ebnodb2no(ebno_db, 
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=self.code_rate)
        received = self.channel([symbols, no])
        llr = self.demapper([received, no])
        bits_hat = self.decoder(llr)
        return bits, bits_hat

# ============================================
# BER Computation Functions
# ============================================
def compute_ber_uncoded(link, ebno_db, batch_size=2000, num_bits=1000, num_batches=10):
    total_errors = 0
    total_bits = 0
    for _ in range(num_batches):
        bits, bits_hat = link.call(batch_size, num_bits, ebno_db)
        errors = tf.reduce_sum(tf.cast(bits != bits_hat, tf.float32)).numpy()
        total_errors += errors
        total_bits += batch_size * num_bits
    return total_errors / total_bits

def compute_ber_coded(link, ebno_db, batch_size=2000, num_batches=15):
    total_errors = 0
    total_bits = 0
    for _ in range(num_batches):
        bits, bits_hat = link.call(batch_size, ebno_db)
        errors = tf.reduce_sum(tf.cast(bits != bits_hat, tf.float32)).numpy()
        total_errors += errors
        total_bits += batch_size * link.k
    return total_errors / total_bits

# ============================================
# Theoretical BER
# ============================================
def theoretical_ber_bpsk(ebno_db_range):
    ebno_linear = 10 ** (np.array(ebno_db_range) / 10)
    return 0.5 * erfc(np.sqrt(ebno_linear))

def theoretical_ber_qpsk(ebno_db_range):
    ebno_linear = 10 ** (np.array(ebno_db_range) / 10)
    return 0.5 * erfc(np.sqrt(ebno_linear))

# ============================================
# Plotting Function
# ============================================
def plot_ber_curves(ebno_range, results, theories):
    plt.figure(figsize=(12, 7))
    
    # Simulation results
    plt.semilogy(ebno_range, results['BPSK'], 'bo-', linewidth=2, 
                markersize=8, label='Uncoded BPSK (Simulation)')
    plt.semilogy(ebno_range, results['QPSK'], 'gs-', linewidth=2, 
                markersize=8, label='Uncoded QPSK (Simulation)')
    plt.semilogy(ebno_range, results['LDPC编码QPSK'], 'rd-', linewidth=2, 
                markersize=8, label='LDPC-coded QPSK (Rate=0.5)')
    
    # Theoretical curves
    plt.semilogy(ebno_range, theories['BPSK'], 'b--', linewidth=1.5, 
                label='BPSK Theory')
    plt.semilogy(ebno_range, theories['QPSK'], 'g--', linewidth=1.5, 
                label='QPSK Theory')
    
    plt.xlabel('Eb/N0 (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('SISO Link Performance Comparison', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='upper right')
    plt.ylim([1e-6, 1])
    plt.xlim([0, 10])
    
    plt.tight_layout()
    plt.savefig('ber_comparison_all.png', dpi=150)
    print("\nFigure saved: ber_comparison_all.png")
    try:
        plt.show()
    except:
        print("Note: Figure saved, can be viewed in folder")

# ============================================
# Print Results Table
# ============================================
def print_results(ebno_range, results):
    print("\n" + "=" * 80)
    print("Simulation Results Summary")
    print("=" * 80)
    print(f"{'Eb/N0 (dB)':<12} {'BPSK':>18} {'QPSK':>18} {'LDPC-coded QPSK':>20}")
    print("-" * 80)
    for i, ebno in enumerate(ebno_range):
        print(f"{ebno:<12} {results['BPSK'][i]:>18.2e} {results['QPSK'][i]:>18.2e} {results['LDPC编码QPSK'][i]:>20.2e}")
    print("=" * 80)

# ============================================
# Compute Coding Gain
# ============================================
def compute_coding_gain(ebno_range, ber_qpsk, ber_ldpc, target_ber=1e-3):
    ebno_qpsk = None
    ebno_ldpc = None
    for i, ber in enumerate(ber_qpsk):
        if ber <= target_ber and ebno_qpsk is None:
            ebno_qpsk = ebno_range[i]
    for i, ber in enumerate(ber_ldpc):
        if ber <= target_ber and ebno_ldpc is None:
            ebno_ldpc = ebno_range[i]
    
    if ebno_qpsk is not None and ebno_ldpc is not None:
        gain = ebno_qpsk - ebno_ldpc
        print(f"\nCoding Gain Analysis (BER = {target_ber:.0e}):")
        print(f"  Uncoded QPSK requires: {ebno_qpsk:.1f} dB")
        print(f"  LDPC-coded requires: {ebno_ldpc:.1f} dB")
        print(f"  → Coding Gain = {gain:.1f} dB")
        return gain
    else:
        print(f"\nCannot compute coding gain (BER = {target_ber:.0e})")
        return None

# ============================================
# Main Function
# ============================================
def main():
    ebno_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Create links
    bpsk_link = UncodedBPSKLink()
    qpsk_link = UncodedQPSKLink()
    ldpc_link = LDPCCodedQPSKLink(k=100, n=200)
    
    results = {'BPSK': [], 'QPSK': [], 'LDPC编码QPSK': []}
    
    # 1. BPSK Simulation
    print("\n[1/3] Simulating Uncoded BPSK System...")
    print("-" * 40)
    for ebno in ebno_range:
        ber = compute_ber_uncoded(bpsk_link, ebno, batch_size=2000, num_bits=1000, num_batches=10)
        results['BPSK'].append(ber)
        print(f"  Eb/N0 = {ebno:2d} dB, BER = {ber:.2e}")
    
    # 2. QPSK Simulation
    print("\n[2/3] Simulating Uncoded QPSK System...")
    print("-" * 40)
    for ebno in ebno_range:
        ber = compute_ber_uncoded(qpsk_link, ebno, batch_size=2000, num_bits=1000, num_batches=10)
        results['QPSK'].append(ber)
        print(f"  Eb/N0 = {ebno:2d} dB, BER = {ber:.2e}")
    
    # 3. LDPC-coded QPSK Simulation
    print("\n[3/3] Simulating LDPC-coded QPSK System (Rate 0.5)...")
    print("-" * 40)
    for ebno in ebno_range:
        ber = compute_ber_coded(ldpc_link, ebno, batch_size=2000, num_batches=15)
        results['LDPC编码QPSK'].append(ber)
        print(f"  Eb/N0 = {ebno:2d} dB, BER = {ber:.2e}")
    
    # Theoretical values
    theories = {
        'BPSK': theoretical_ber_bpsk(ebno_range),
        'QPSK': theoretical_ber_qpsk(ebno_range)
    }
    
    # Plot
    plot_ber_curves(ebno_range, results, theories)
    
    # Print results
    print_results(ebno_range, results)
    
    # Coding gain
    compute_coding_gain(ebno_range, results['QPSK'], results['LDPC编码QPSK'])
    
    print("\n" + "=" * 80)
    print("Conclusions:")
    print("  1. BPSK and QPSK BER curves nearly overlap (theoretically identical)")
    print("  2. LDPC coding significantly reduces SNR required to achieve the same BER")
    print("  3. Coding gain demonstrates the performance improvement from channel coding")
    print("  4. Higher-order modulation (QPSK) increases data rate without bandwidth expansion")
    print("=" * 80)
    print("\n✓ Simulation complete!")

if __name__ == "__main__":
    main()