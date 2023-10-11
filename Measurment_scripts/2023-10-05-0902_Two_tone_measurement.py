import os
import sys
import time
import shutil

import numpy as np
import matplotlib.pyplot as plt
import pyvisa as visa


#sys.path.append('C:\\Users\\Valve Control\\anaconda3\\envs\\202lab\\Lib\\site-packages')
sys.path.append('C:\\Users\\Lab\\anaconda3\\Lib\\site-packages')
from presto.utils import format_sec, get_sourcecode, sin2
from presto import pulsed
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from utils_presto import setup_IF_template, save_file

date_string = time.strftime('%Y-%m-%d-%H%M')
base_filename = os.path.basename(__file__)
file_name = "Data\\" + date_string + '_' + base_filename
sourcecode = get_sourcecode(__file__)
shutil.copy(base_filename, file_name)

save_to_file = 1
use_fitting = 0

# setting the general settings
num_averages = 1000
nr = 101
wait_decay = 200e-6  # delay between repetitions

# setting the LO
readout_LO =  7.7e9 + 43e6#7.743510000e9
qubit_LO = 5.5e9  #5.537790000e9
cavity_LO = 3.674278939e9

# qubit drive: control
control_freq = np.linspace(30e6,180e6,101)  # Hz, IF frequency
control_cutoff = 4
control_length = 40e-6
control_amp = 0.1#0.02

# cavity drive: readout
readout_freq = 100e6
readout_shape = "square"
readout_amp = 0.16
readout_amp_kick = 0.0
readout_length = 2000e-9  # s
readout_length_kick = 50e-9  # s
readout_delay = 10e-9

# digital mixer calibration
control_phaseI = 0.
control_phaseQ = control_phaseI + 0.5 * np.pi  # low sideband

readout_phaseI = + np.pi / 2
readout_phaseQ = readout_phaseI - 0.5 * np.pi  # high sideband


## Initialization
readout_port = 1
control_port = 2

nr = len(control_freq)


### setting the sampling window
sample_port = 1
sample_length = 2300e-9
readout_sample_delay = 10e-9

raw_data = []
raw_matches = []

with pulsed.Pulsed(ext_ref_clk=False, dry_run=False, force_config=True, address="192.168.88.72",adc_mode=AdcMode.Mixed,dac_mode=[DacMode.Mixed02,DacMode.Mixed02,DacMode.Mixed02,DacMode.Mixed02], adc_fsample=AdcFSample.G2,dac_fsample=[DacFSample.G10,DacFSample.G8,DacFSample.G8,DacFSample.G8]) as q:
    # *** Set parameters ***
    sampling_freq = q.get_fs('dac')
    sampling_freq_adc = q.get_fs('adc')
    print("sampling freq = ", sampling_freq)
    q.hardware.configure_mixer(freq=readout_LO, in_ports=sample_port, out_ports=readout_port)
    q.hardware.configure_mixer(freq=qubit_LO, out_ports=control_port,sync=True)
   
    #setup readout
    _template_readout_I, _template_readout_Q = setup_IF_template(q, readout_length, 0, readout_shape,0, 0)
    readout_pulse = q.setup_template(readout_port, 0, _template_readout_I, envelope=True)

    q.setup_scale_lut(readout_port, group=0, scales=[readout_amp])
    q.setup_freq_lut(readout_port, group=0, frequencies=readout_freq, phases=readout_phaseI,phases_q=readout_phaseI+readout_phaseQ)

    # setup control
    control_pulse = q.setup_long_drive(control_port, group=0, duration=control_length,amplitude=1 + 1j, rise_time=0,
                                              fall_time=0)

    q.setup_scale_lut(control_port, group=0, scales=control_amp)
    q.setup_freq_lut(control_port, group=0, frequencies=control_freq, phases=np.ones(nr)*control_phaseI, phases_q=np.ones(nr)*control_phaseQ)
    
    # Set sampling
    q.set_store_ports([sample_port])
    q.set_store_duration(sample_length)
    T0 = 2e-6  # s, start from some time
    T = T0
    n=1
# *** Program pulse sequence ***
    q.reset_phase(T, [control_port])
    q.output_pulse(T, [control_pulse])
    T += control_length
    q.output_pulse(T, [readout_pulse])
    q.next_frequency(T+readout_length, [control_port], group=0)
    # Sample
    q.store(T)
    T += readout_length
    T += wait_decay
    
    expected_runtime = (T - T0) * nr * num_averages  # s
    print("Expected total runtime: {:s}".format(format_sec(expected_runtime)))
    q.run(T, nr, num_averages, print_time=True)
    t_array, _result = q.get_store_data()
    raw_data.append(_result)

#sgs4a4.close()

phase = 0
p2_phase_shift = 0.5*np.pi
match_length = readout_length
start = int(round(268e-9 * sampling_freq_adc))
end = start + int(round(match_length * sampling_freq_adc))
t_template_match = np.linspace(0, match_length, int(round(match_length * sampling_freq_adc)), False)
demod_I_p = []
demod_Q_p = []
demod_I_m = []
demod_Q_m = []
for i in range(nr):
    carrier_m1_p1 = np.cos(2 * np.pi * readout_freq * t_template_match + np.pi * phase)
    carrier_m1_p2 = np.sin(2 * np.pi * readout_freq * t_template_match + np.pi * (phase + p2_phase_shift))
    carrier_m2_p1 = -np.sin(2 * np.pi * readout_freq * t_template_match + np.pi * phase)
    carrier_m2_p2 = np.cos(2 * np.pi * readout_freq * t_template_match + np.pi * (phase + p2_phase_shift))
    demod_I_p.append(np.sum(carrier_m1_p1 * _result[i, 0, start:end].real + carrier_m1_p2 * _result[i, 0, start:end].imag))
    demod_Q_p.append(np.sum(carrier_m2_p1 * _result[i, 0, start:end].real + carrier_m2_p2 * _result[i, 0, start:end].imag))

demod_I_p = np.array(demod_I_p)
demod_Q_p = np.array(demod_Q_p)


x_axis = control_freq
### SAVE FILE
if save_to_file:
    _dict = dict()
    for key in dir():
        if isinstance(globals()[key], (np.ndarray, float, int, str)):
            _dict[key] = globals()[key]

    np.savez(file_name[:-3], **_dict)

plt.figure()
plt.plot(x_axis, (demod_I_p **2 + demod_Q_p**2)**(1/2) * match_length, '.-')
#plt.figure()
#plt.plot(x_axis, np.arctan2(demod_Q_p,demod_I_p), '.-')
#plt.figure()
#plt.plot(np.abs(_result[50,0,:]))

