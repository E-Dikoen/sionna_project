"""
SISO Link Simulation - Interactive Configurable Model
Users can select modulation scheme, coding rate and other parameters at runtime
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sionna as sn
from scipy.special import erfc

# Set matplotlib for English display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# Configurable Link Model Class
# ============================================
class ConfigurableLink:
    """
    Configurable communication link model
    
    Parameters:
        modulation: Modulation scheme ('bpsk', 'qpsk', '16qam', '64qam')
        code_rate: Code rate (None for uncoded, 0.5 for LDPC rate 0.5, 0.75 for rate 0.75)
        k: Number of information bits (for coded case)
    """
    
    def __init__(self, modulation='qpsk', code_rate=None, k=100):
        # Modulation parameter mapping
        mod_map = {
            'bpsk': {'type': 'pam', 'bits': 1, 'name': 'BPSK'},
            'qpsk': {'type': 'qam', 'bits': 2, 'name': 'QPSK'},
            '16qam': {'type': 'qam', 'bits': 4, 'name': '16QAM'},
            '64qam': {'type': 'qam', 'bits': 6, 'name': '64QAM'}
        }
        
        if modulation not in mod_map:
            raise ValueError(f"Unsupported modulation: {modulation}. Supported: {list(mod_map.keys())}")
        
        self.modulation = modulation
        self.modulation_name = mod_map[modulation]['name']
        self.mod_type = mod_map[modulation]['type']
        self.num_bits_per_symbol = mod_map[modulation]['bits']
        self.code_rate = code_rate
        self.k = k
        
        # Create constellation and modulator/demodulator
        if self.mod_type == 'pam':
            # BPSK uses PAM
            self.constellation = sn.mapping.Constellation(self.mod_type, self.num_bits_per_symbol)
            self.mapper = sn.mapping.Mapper(constellation=self.constellation)
            self.demapper = sn.mapping.Demapper('app', constellation=self.constellation)
        else:
            # QAM modulation
            self.mapper = sn.mapping.Mapper(self.mod_type, self.num_bits_per_symbol)
            self.demapper = sn.mapping.Demapper('app', constellation_type=self.mod_type,
                                                num_bits_per_symbol=self.num_bits_per_symbol)
        
        self.channel = sn.channel.AWGN()
        
        # Coding related
        self.encoder = None
        self.decoder = None
        self.n = None
        
        if code_rate is not None:
            # Use LDPC coding
            if code_rate == 0.5:
                self.n = 2 * k  # code_rate = k/n = 0.5
            elif code_rate == 0.75:
                self.n = int(k / 0.75)
            else:
                raise ValueError(f"Unsupported code rate: {code_rate}")
            
            self.encoder = sn.fec.ldpc.LDPC5GEncoder(k, self.n)
            self.decoder = sn.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True)
    
    def get_code_rate(self):
        return self.code_rate if self.code_rate is not None else 1.0
    
    def call(self, batch_size, ebno_db, num_bits=None):
        """Run link simulation"""
        source = sn.utils.BinarySource()
        
        # Determine number of bits
        if self.code_rate is not None:
            # Coded link
            num_info_bits = self.k
            bits = source([batch_size, num_info_bits])
            coded_bits = self.encoder(bits)
        else:
            # Uncoded link
            if num_bits is None:
                num_bits = 1000
            bits = source([batch_size, num_bits])
            coded_bits = bits
        
        # Modulation
        symbols = self.mapper(coded_bits)
        
        # Channel
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=self.get_code_rate())
        received = self.channel([symbols, no])
        
        # Demodulation
        llr = self.demapper([received, no])
        
        # Decoding
        if self.code_rate is not None:
            bits_hat = self.decoder(llr)
        else:
            bits_hat = tf.cast(llr > 0, tf.float32)
        
        return bits, bits_hat
    
    def get_config(self):
        """Return configuration string"""
        if self.code_rate is not None:
            return f"{self.modulation_name} | Coded | Rate={self.code_rate} | k={self.k}, n={self.n}"
        else:
            return f"{self.modulation_name} | Uncoded"
    
    def get_config_dict(self):
        """Return configuration dictionary"""
        return {
            'modulation': self.modulation_name,
            'num_bits_per_symbol': self.num_bits_per_symbol,
            'code_rate': self.code_rate if self.code_rate is not None else 'Uncoded',
            'k': self.k if self.code_rate is not None else 'N/A',
            'n': self.n if self.code_rate is not None else 'N/A'
        }

# ============================================
# BER Computation Functions
# ============================================
def compute_ber(link, ebno_db, batch_size=2000, num_batches=10):
    """Compute Bit Error Rate"""
    total_errors = 0
    total_bits = 0
    
    for _ in range(num_batches):
        bits, bits_hat = link.call(batch_size, ebno_db)
        errors = tf.reduce_sum(tf.cast(bits != bits_hat, tf.float32)).numpy()
        total_errors += errors
        
        if link.code_rate is not None:
            total_bits += batch_size * link.k
        else:
            total_bits += batch_size * 1000
    
    return total_errors / total_bits if total_bits > 0 else 1.0

# ============================================
# Theoretical BER
# ============================================
def theoretical_ber(ebno_db_range, modulation='qpsk'):
    """Theoretical Bit Error Rate"""
    ebno_linear = 10 ** (np.array(ebno_db_range) / 10)
    
    if modulation == 'bpsk':
        return 0.5 * erfc(np.sqrt(ebno_linear))
    elif modulation == 'qpsk':
        return 0.5 * erfc(np.sqrt(ebno_linear))
    elif modulation == '16qam':
        # 16QAM approximate theoretical BER
        return 0.375 * erfc(np.sqrt(0.8 * ebno_linear))
    else:
        return 0.5 * erfc(np.sqrt(ebno_linear))

# ============================================
# Plotting Function
# ============================================
def plot_ber_curves(ebno_range, results, title="SISO Link Performance", filename="ber_comparison.png"):
    """Plot BER curves"""
    plt.figure(figsize=(12, 7))
    
    colors = ['bo-', 'gs-', 'rd-', 'cp-', 'm^-', 'y*-']
    for i, (name, ber) in enumerate(results.items()):
        if ber is not None:
            plt.semilogy(ebno_range, ber, colors[i], linewidth=2, 
                        markersize=8, label=name)
    
    plt.xlabel('Eb/N0 (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='upper right')
    plt.ylim([1e-6, 1])
    plt.xlim([0, 10])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\nFigure saved: {filename}")
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
    
    # Header
    header = f"{'Eb/N0 (dB)':<12}"
    for name in results.keys():
        header += f"{name:>20}"
    print(header)
    print("-" * (12 + 20 * len(results)))
    
    # Data
    for i, ebno in enumerate(ebno_range):
        row = f"{ebno:<12}"
        for ber in results.values():
            row += f"{ber[i]:>20.2e}"
        print(row)
    print("=" * 80)

# ============================================
# Interactive Menu
# ============================================
def show_menu():
    """Show main menu"""
    print("\n" + "=" * 60)
    print("SISO Link Simulation System - Interactive Configuration")
    print("=" * 60)
    print("\nPlease select simulation mode:")
    print("  1. Single Link Simulation (Custom Configuration)")
    print("  2. Multi-Link Comparison (Select Multiple Configurations)")
    print("  3. Preset Comparison Scenarios")
    print("  4. Exit")
    return input("\nEnter your choice (1-4): ")

def get_modulation():
    """Interactive modulation selection"""
    print("\nAvailable modulation schemes:")
    print("  1. BPSK  (1 bit/symbol)")
    print("  2. QPSK  (2 bits/symbol)")
    print("  3. 16QAM (4 bits/symbol)")
    print("  4. 64QAM (6 bits/symbol)")
    
    choice = input("Please select (1-4): ")
    mod_map = {'1': 'bpsk', '2': 'qpsk', '3': '16qam', '4': '64qam'}
    return mod_map.get(choice, 'qpsk')

def get_coding():
    """Interactive coding selection"""
    print("\nCoding scheme:")
    print("  1. Uncoded")
    print("  2. LDPC Coding (Rate 0.5)")
    print("  3. LDPC Coding (Rate 0.75)")
    
    choice = input("Please select (1-3): ")
    if choice == '1':
        return None, None
    elif choice == '2':
        code_rate = 0.5
        k = int(input("Enter number of information bits k (default 100): ") or 100)
        return code_rate, k
    elif choice == '3':
        code_rate = 0.75
        k = int(input("Enter number of information bits k (default 150): ") or 150)
        return code_rate, k
    else:
        return None, None

def run_single_simulation():
    """Single link simulation"""
    print("\n" + "-" * 40)
    print("Single Link Simulation")
    print("-" * 40)
    
    # Get configuration
    modulation = get_modulation()
    code_rate, k = get_coding()
    
    # Create link
    link = ConfigurableLink(modulation=modulation, code_rate=code_rate, k=k if k else 100)
    print(f"\nCreated link: {link.get_config()}")
    
    # Simulation parameters
    ebno_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    batch_size = int(input("Enter batch_size (default 2000): ") or 2000)
    num_batches = int(input("Enter number of Monte Carlo batches (default 10): ") or 10)
    
    # Run simulation
    print("\nSimulating...")
    ber_list = []
    for ebno in ebno_range:
        ber = compute_ber(link, ebno, batch_size=batch_size, num_batches=num_batches)
        ber_list.append(ber)
        print(f"  Eb/N0 = {ebno:2d} dB, BER = {ber:.2e}")
    
    # Display results
    results = {link.get_config(): ber_list}
    print_results(ebno_range, results)
    
    # Plot
    plot_ber_curves(ebno_range, results, f"BER Curve - {link.get_config()}", "single_simulation.png")

def run_multi_comparison():
    """Multi-link comparison simulation"""
    print("\n" + "-" * 40)
    print("Multi-Link Comparison Simulation")
    print("-" * 40)
    
    links = []
    names = []
    
    while True:
        print(f"\nCurrently added {len(links)} link(s)")
        choice = input("Add new link? (y/n): ")
        if choice.lower() != 'y':
            break
        
        print(f"\n--- Link {len(links)+1} ---")
        modulation = get_modulation()
        code_rate, k = get_coding()
        
        link = ConfigurableLink(modulation=modulation, code_rate=code_rate, k=k if k else 100)
        links.append(link)
        names.append(link.get_config())
        print(f"Added: {link.get_config()}")
    
    if len(links) == 0:
        print("No links added, returning to main menu")
        return
    
    # Simulation parameters
    ebno_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    batch_size = int(input("Enter batch_size (default 2000): ") or 2000)
    num_batches = int(input("Enter number of Monte Carlo batches (default 10): ") or 10)
    
    # Run simulation
    results = {}
    for link, name in zip(links, names):
        print(f"\nSimulating: {name}")
        ber_list = []
        for ebno in ebno_range:
            ber = compute_ber(link, ebno, batch_size=batch_size, num_batches=num_batches)
            ber_list.append(ber)
            print(f"  Eb/N0 = {ebno:2d} dB, BER = {ber:.2e}")
        results[name] = ber_list
    
    # Display results
    print_results(ebno_range, results)
    
    # Plot
    plot_ber_curves(ebno_range, results, "Multi-Link Performance Comparison", "multi_comparison.png")

def run_preset_scenarios():
    """Preset comparison scenarios"""
    print("\n" + "-" * 40)
    print("Preset Comparison Scenarios")
    print("-" * 40)
    print("\nAvailable preset scenarios:")
    print("  1. Modulation Comparison (BPSK vs QPSK vs 16QAM)")
    print("  2. Coding Gain Comparison (QPSK vs LDPC-coded QPSK)")
    print("  3. Code Rate Comparison (Rate 0.5 vs Rate 0.75)")
    print("  4. Return to Main Menu")
    
    choice = input("\nPlease select (1-4): ")
    
    ebno_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    batch_size = 2000
    num_batches = 10
    
    results = {}
    
    if choice == '1':
        print("\nScenario 1: Modulation Comparison (Uncoded)")
        modulations = ['bpsk', 'qpsk', '16qam']
        for mod in modulations:
            link = ConfigurableLink(modulation=mod, code_rate=None)
            name = link.get_config()
            print(f"\nSimulating: {name}")
            ber_list = []
            for ebno in ebno_range:
                ber = compute_ber(link, ebno, batch_size=batch_size, num_batches=num_batches)
                ber_list.append(ber)
                print(f"  Eb/N0 = {ebno:2d} dB, BER = {ber:.2e}")
            results[name] = ber_list
        plot_ber_curves(ebno_range, results, "Modulation Comparison (Uncoded)", "modulation_comparison.png")
        
    elif choice == '2':
        print("\nScenario 2: Coding Gain Comparison (QPSK)")
        # Uncoded QPSK
        link_uncoded = ConfigurableLink(modulation='qpsk', code_rate=None)
        name_uncoded = link_uncoded.get_config()
        print(f"\nSimulating: {name_uncoded}")
        ber_uncoded = []
        for ebno in ebno_range:
            ber = compute_ber(link_uncoded, ebno, batch_size=batch_size, num_batches=num_batches)
            ber_uncoded.append(ber)
            print(f"  Eb/N0 = {ebno:2d} dB, BER = {ber:.2e}")
        results[name_uncoded] = ber_uncoded
        
        # LDPC coded QPSK
        link_coded = ConfigurableLink(modulation='qpsk', code_rate=0.5, k=100)
        name_coded = link_coded.get_config()
        print(f"\nSimulating: {name_coded}")
        ber_coded = []
        for ebno in ebno_range:
            ber = compute_ber(link_coded, ebno, batch_size=batch_size, num_batches=15)
            ber_coded.append(ber)
            print(f"  Eb/N0 = {ebno:2d} dB, BER = {ber:.2e}")
        results[name_coded] = ber_coded
        
        plot_ber_curves(ebno_range, results, "Coding Gain Analysis", "coding_gain.png")
        
    elif choice == '3':
        print("\nScenario 3: Code Rate Comparison (QPSK)")
        code_rates = [0.5, 0.75]
        for cr in code_rates:
            k = 100 if cr == 0.5 else 150
            link = ConfigurableLink(modulation='qpsk', code_rate=cr, k=k)
            name = link.get_config()
            print(f"\nSimulating: {name}")
            ber_list = []
            for ebno in ebno_range:
                ber = compute_ber(link, ebno, batch_size=batch_size, num_batches=15)
                ber_list.append(ber)
                print(f"  Eb/N0 = {ebno:2d} dB, BER = {ber:.2e}")
            results[name] = ber_list
        plot_ber_curves(ebno_range, results, "Code Rate Comparison", "code_rate_comparison.png")
    
    else:
        return

# ============================================
# Main Function
# ============================================
def main():
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU available: {gpus[0]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("Using CPU")
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            run_single_simulation()
        elif choice == '2':
            run_multi_comparison()
        elif choice == '3':
            run_preset_scenarios()
        elif choice == '4':
            print("\nThank you for using the SISO Link Simulation System!")
            break
        else:
            print("\nInvalid option, please try again")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()