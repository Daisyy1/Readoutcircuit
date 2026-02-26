from picosdk.ps2000 import ps2000
from picosdk.PicoDeviceEnums import picoEnum
from picosdk.functions import assert_pico2000_ok, adc2mV
from ctypes import byref, c_int16, c_int32, c_byte, sizeof
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal, stats
import csv
    
SAMPLES = 1000
OVERSAMPLING = 1


RPM = 500
CUTOUT = 2
DISK_RADIUS_MM = 20

V_DISK = (2 * np.pi * DISK_RADIUS_MM * RPM ) / 60
V_DISK_NM_NS = V_DISK * 1e6 / 1e9

def get_timebase(device, wanted_time_interval):
    current_timebase = 1
    time_interval = c_int32(0)
    time_units = c_int16()
    max_samples = c_int32()
    
    while ps2000.ps2000_get_timebase(
        device.handle,
        current_timebase,
        SAMPLES,
        byref(time_interval),
        byref(time_units),
        OVERSAMPLING,
        byref(max_samples)) == 0 \
        or time_interval.value < wanted_time_interval.value:
            current_timebase += 1
            
    if current_timebase.bit_length() > sizeof(c_int16) * 8:
        raise Exception('No appropriate timebase was identifiable')
        
    return current_timebase -1, current_timebase

def measure(n, v_range, name):
    with ps2000.open_unit() as device:
        print('Device Info: {}'.format(device.info))
        
        res = ps2000.ps2000_set_sig_gen_built_in(
            device.handle,
            1_000_000,
            2_000_000,
            1,# square wave
            3.18,
            3.18,
            0.0,
            0.1,
            0,
            1_000
        )
        assert_pico2000_ok(res)
        
        sleep(0.1)
        
        res = ps2000.ps2000_set_channel(
            device.handle,
            picoEnum.PICO_CHANNEL['PICO_CHANNEL_B'],
            False,
            picoEnum.PICO_COUPLING['PICO_DC'],
            v_range
        )
        assert_pico2000_ok(res)
         # Change to 4096 when measuring over 1k ohm resistor
        # Change to 8192 when measuring direct
        # Change to 12288 when measuring over op amp for 10uA current
        # This value depends on both the window range and the expected signal value.
       
       
        res = ps2000.ps2000_set_trigger(
            device.handle,
            picoEnum.PICO_CHANNEL['PICO_CHANNEL_A'],
            c_int16(8192), # This value should be chosen between +32767 and -32767, where the largest number is the max voltage of the chosen window size.
            0,
            -50,
            0
        )
        assert_pico2000_ok(res)
        
        _, timebase_a = get_timebase(device, c_int32(10))
        collection_time = c_int32()
        
        def run():
            res = ps2000.ps2000_run_block(
                device.handle,
                SAMPLES,
                timebase_a,
                OVERSAMPLING,
                byref(collection_time)
            )
            assert_pico2000_ok(res)
            
            while ps2000.ps2000_ready(device.handle) == 0:
                sleep(0.1)
                
            times = (c_int32 * SAMPLES)()
            buffer_a = (c_int16 * SAMPLES)()
            overflow = c_byte(0)
            
            res = ps2000.ps2000_get_times_and_values(
                device.handle,
                byref(times),
                buffer_a,
                None,
                None,
                None,
                byref(overflow),
                2,
                SAMPLES
            )
            assert_pico2000_ok(res)
            
            channel_a_overflow = (overflow.value & 0b0000_0001) != 0
            ps2000.ps2000_stop(device.handle)
            
            channel_a_mV = adc2mV(
                buffer_a,
                v_range,
                c_int16(32767)
            )
            return list(times), list(channel_a_mV)

        times, voltages = run()
        
        raw_fig = plt.figure()
        plt.plot(times, voltages)
        plt.xlabel("times [ns]")
        plt.ylabel("Voltages [mV]")
        raw_fig.savefig(f"Raw waveforms 2.5uA gain\\Raw_{name}.png")
        plt.close(raw_fig)
        
        labels = ['Times (ns)', 'Voltages (mV)']
        time_arr = []
        V_arr = []
        
        for i in range(n):
            time_val, V_val = run()
            if V_val[0] < V_val[-1]:
                time_arr.append(time_val)
                V_arr.append(V_val)
                
        final_t = []
        final_V = []
        vals = np.linspace(0, 0, num=SAMPLES)
        
        for i in range(len(time_arr)):
            vals = np.add(vals, time_arr[i])
        final_t = [x/n for x in vals]
        
        vals = np.linspace(0, 0, num=SAMPLES)
        for i in range(len(V_arr)):
            vals = np.add(vals, V_arr[i])
        final_V = [x/n for x in vals]
        
        float_fig = plt.figure()
        plt.plot(final_t, final_V)
        plt.xlabel("times [ns]")
        plt.ylabel("Voltages [mV]")
        float_fig.savefig(f"Float waveforms 2.5uA gain\\Float_{name}.png")
        plt.close(float_fig)
        
        return final_t, final_V, voltages

def signal_process(times, S, V, SNR_val, FWHM):
    S_corr = np.subtract(S, np.min(S))
    max_amplitude = max(S_corr)
    # Replace with peak-to-peak if the signal oscillates around zero
    low_ref = 0.1 * max_amplitude
    high_ref = 0.9 * max_amplitude
    
    t_low = None
    t_high = None
    
    for time, value in zip(times, S_corr):
        if t_low is None and value >= low_ref:
            t_low = time
        if t_low is not None and value >= high_ref:
            t_high = time
            break
            
    rise_time = None
    tau = None
    
    if t_low is not None and t_high is not None:
        rise_time = t_high - t_low

        d_eff = V_DISK_NM_NS * rise_time
        d_gauss = 0.92 * d_eff
        print("Effective Diameter:", d_eff, "[nm]")# v*tr
        print("FWHM Gaussian Diameter:", d_gauss, "[nm]")# 0.92*v*tr

        tau = rise_time / 2.2
        print("Rise Time:", rise_time, "[ns]")
        print("Tau = ", tau, "[ns]")
    else:
        print("Rise time couldn't be determined.")
        
    #SNR = (max(S) / np.max(np.subtract(S, V))) ** 2
    SNR = SNR_val
    print("SNR = ", SNR)
    
    params = [SNR, tau, rise_time, d_eff, d_gauss, FWHM]
    
    with open("5uA_gain_data.csv", "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(params)

def generate_fft(times, V, name):
    times = [x * 10 ** (-9) for x in times]
    V = [x * 10 ** (-3) for x in V]
    N = len(V)
    T = abs((times[1] - times[0]))
    print(1/T)
    
    SNf = fft(np.subtract(V, np.mean(V)))
    xf = fftfreq(N, T)[:N // 2]
    
    fig1 = plt.figure()
    plt.plot(xf, 2.0/N * np.abs(SNf[0:N // 2]))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title(f"FFT of the raw signal for {name}")
    fig1.savefig(f"FFT_of_{name}.png")

def FIR_filter(times, V, name, raw_V):
    times = [x * 10 ** (-9) for x in times]
    V = [x * 10 ** (-3) for x in V]
    N = len(V)
    T = abs(times[1] - times[0])
    
    numtaps = int(N/2) if (int(N / 2) % 2) != 0 else int(N / 2) + 1
    
    h = signal.firwin(numtaps=numtaps, cutoff=3 * 10 ** 6, window="blackman", fs=1/T)
    S = np.convolve(V, h, mode="valid")
    
    trim_size = (numtaps - 1) // 2
    
    # Adjust the V and times arrays to match the size of S
    V = V[trim_size:-(trim_size or None)]
    times = times[trim_size:-(trim_size or None)]
    raw_V = raw_V[trim_size:-(trim_size or None)]
    
    times = [x * 10 ** 9 for x in times]
    V = [x * 10 ** 3 for x in V]
    S = [x * 10 ** 3 for x in S]
    
    SNR = (max(S) / np.max(np.subtract(S, raw_V))) ** 2
    """
    fig2 = plt.figure()
    plt.plot(times, S, color="r")
    plt.xlabel("Time [s]")
    plt.ylabel("Signal [V]")
    plt.title(f"Filtered Signal for {name}")
    # fig2.savefig(f"Filter_Signal_{name}.png")
    """
    fig3 = plt.figure()
    plt.plot(times, S, color="r", label="Filtered Signal")
    plt.plot(times, V, color="b", label="Raw Signal")
    plt.xlabel("Time [ns]")
    plt.ylabel("Signal [mV]")
    plt.legend()
    plt.title(f"Filtered Signal and Raw signal for {name}")
    plt.show()
    fig3.savefig(f"FIR waveforms 2.5uA gain\\FIR_Raw_{name}.png")
    plt.close(fig3)
    
    return times, S, SNR

def calc_radius(S, resis, rotation, times):
    dI = np.zeros(len(S))
    current = np.divide(S, resis)
    distances = np.multiply(times, rotation)
    dt = times[1] - times[0]
    
    for i in range(len(S)):
        if 0 < i < len(S) - 1:
            dI[i] = (current[i+1] - current[i-1]) / (2*dt)
            
    return dI, distances

if __name__ == "__main__":
    choice = int(input("Choose Your Current: (1)5V (2)50mV (3)200mV (4)1V (5)500mV: "))
    
    if choice == 1:
        V_range = ps2000.PS2000_VOLTAGE_RANGE['PS2000_5V']
    elif choice == 2:
        V_range = ps2000.PS2000_VOLTAGE_RANGE['PS2000_50MV']
    elif choice == 3:
        V_range = ps2000.PS2000_VOLTAGE_RANGE['PS2000_200MV']
    elif choice == 4:
        V_range = ps2000.PS2000_VOLTAGE_RANGE['PS2000_1V']
    elif choice == 5:
        V_range = ps2000.PS2000_VOLTAGE_RANGE['PS2000_500MV']
    else:
        V_range = ps2000.PS2000_VOLTAGE_RANGE['PS2000_5V']
        
    nr_of_loops = 1
    
    for i in range(nr_of_loops):
        times, S, V = measure(100, v_range=V_range, name=f"2.5uA_gain_{i}")
        times, S, SNR = FIR_filter(times, S, name=f"2.5uA_gain_{i}", raw_V=V)
        
        dI, distances = calc_radius(S=S, resis=1000, rotation=1, times=times)
        
        # Assuming dI and distances are numpy arrays of the same length
        dI = np.array(dI)  # Replace with your intensity data
        distances = np.array(distances)
        
        # Below are the steps to calculate FWHM:
        # Step 1: Find the maximum intensity and its corresponding index
        max_intensity = np.max(dI)
        half_max = max_intensity / 2
        
        # Step 2 & 3: Find the points where the intensity crosses half the maximum
        # The approach here is to find the first point to the left of the
        # peak and the first point to the right of the peak
        
        # Note: This simple slicing assumes the peak is well-defined and centered.
        left_index = np.where(dI >= half_max)[0][0]
        right_index = np.where(dI >= half_max)[0][-1]
        # Step 4: Calculate the FWHM
        FWHM = distances[right_index] - distances[left_index]
        
            
        print("FWHM:", FWHM)
        
        rad_fig = plt.figure()
        plt.plot(distances, dI)
        plt.ylabel(r"$\dfrac{dI}{dt}$")
        plt.xlabel(r"$x$")
        plt.title("Distribution of current density vs radius")
        plt.show()
        rad_fig.savefig(f"Derivative waveforms 2.5uA gain\\Derivative_of_2.5uA_gain{i}.png")
        signal_process(times, S, V, SNR, FWHM)
        
        plt.close(rad_fig)
        
        