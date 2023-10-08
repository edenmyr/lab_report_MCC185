import os
import sys
import time
import pyvisa as visa
import shutil
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('C:\\Users\\Lab\\anaconda3\\Lib\\site-packages\\')
from presto.utils import format_sec, get_sourcecode, rotate_opt
from presto import pulsed as pls
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from utils_presto import setup_IF_template, save_file


date_string = time.strftime('%Y-%m-%d-%H%M')
base_filename = os.path.basename(__file__)
file_name = "Data\\" + date_string + base_filename
shutil.copy(base_filename, file_name)

# Option
use_template_matching = 0
use_fitting = 0
save_to_file = 1

## Time domain setup
num_averages = 500
wait_decay = 2000e-6  # delay between repetitions

# qubit drive: control
control_freq = 100e6  # Hz, IF frequency
control_shape = "sin2"
control_cutoff = 4
control_length = 2000e-9  # s, pi pulse
control_amp = 0.0082

# resonator drive: readout
readout_freq = 100e6  # Hz
readout_shape = "square"
readout_amp = 0.1
readout_amp_kick = 0.0
readout_length = 2000e-9  # s
readout_length_kick = 50e-9  # s
readout_delay = 0e-9

# cavity drive: displacement
disp_length = 20000e-9
disp_amp = np.array([0.01])
disp_shape = "sin2"
disp_frequency = np.linspace(50e6,150e6,81)




# sampling
sample_length = 2000e-9
sampling_start = 0e-9
matching_start = 268e-9

# template matching / demodulation
match_length = 1.6e-6  # s
phase = 0
p2_phase_shift = 0

# LO settings
readout_LO = 7.7e9 + 43e6 #7.743510000e9
qubit_LO = 5.5e9 + 38e6 - 423e3  #5.537576000e9
cavity_LO =  3.656706e9 #3.656978939e9

# digital mixer calibration
control_phaseI = 0.
control_phaseQ = control_phaseI + 0.5 * np.pi  # low sideband

readout_phaseI = + np.pi / 2
readout_phaseQ = readout_phaseI - 0.5 * np.pi  # high sideband

disp_phaseI = 0
disp_phaseQ = disp_phaseI + 0.5 * np.pi # low sideband

## Initialization
readout_port = 1
control_port = 2
disp_port = 5
sample_port = 1

#rm = visa.ResourceManager()
#sgs4a4 = rm.open_resource('TCPIP::sgs100a-4a4.qdp.chalmers.se')
#sgs4a4.write(':SOUR:FREQ:CW '+str(twpa_LO))
#sgs4a4.write(':SOUR:POW:POW '+str(twpa_POW))
#sgs4a4.write(':OUTPut ON')

nr = len(disp_frequency)
nr1 = len(disp_amp)

raw_data = []
raw_matches = []

with pls.Pulsed(ext_ref_clk=False, dry_run=False, force_config=True, address="192.168.88.72", adc_mode=AdcMode.Mixed,
                dac_mode=[DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
                adc_fsample=AdcFSample.G2, dac_fsample=[DacFSample.G10,DacFSample.G8,DacFSample.G8,DacFSample.G8]) as q:    # *** Set parameters ***
    sampling_freq_dac = q.get_fs('dac')
    sampling_freq_adc = q.get_fs('adc')
    q.hardware.configure_mixer(freq=readout_LO, in_ports=sample_port, out_ports=readout_port)
    q.hardware.configure_mixer(freq=qubit_LO, out_ports=control_port)
    q.hardware.configure_mixer(freq=cavity_LO, out_ports=disp_port, sync=True)

    # setup disp
    #_template_disp_I, _template_disp_Q = setup_IF_template(q, disp_length, 0, disp_shape, 0, 0)
    disp_pulse = q.setup_long_drive(disp_port, 0, duration=disp_length, rise_time=50e-9, fall_time=50e-9)

    q.setup_scale_lut(disp_port, group=0, scales=disp_amp)
    q.setup_freq_lut(disp_port, 0, frequencies=disp_frequency, phases=np.ones(nr)*disp_phaseI, phases_q=np.ones(nr)*disp_phaseQ)


    # setup control
    _template_control_I, _template_control_Q = setup_IF_template(q, control_length, 0, control_shape, 0, 0,
                                                                 cutoff=control_cutoff)
    control_pulse = q.setup_template(control_port, 0, _template_control_I, envelope=True)

    q.setup_scale_lut(control_port, group=0, scales=control_amp)
    q.setup_freq_lut(control_port, group=0, frequencies=control_freq, phases=control_phaseI, phases_q=control_phaseQ)

    # setup readout
    _template_readout_kick_I, _template_readout_kick_Q = setup_IF_template(q, readout_length_kick, readout_freq,
                                                                           'square',
                                                                           readout_phaseI, readout_phaseQ,
                                                                           readout_amp_kick)
    readout_kick_pulse = q.setup_template(readout_port, 0, _template_readout_kick_I, envelope=False)

    _template_readout_I, _template_readout_Q = setup_IF_template(q, readout_length, 0, readout_shape, 0, 0)
    readout_pulse = q.setup_template(readout_port, 0, _template_readout_I, envelope=True)

    q.setup_scale_lut(readout_port, group=0, scales=[readout_amp])
    q.setup_freq_lut(readout_port, group=0, frequencies=readout_freq, phases=readout_phaseI,
                     phases_q=readout_phaseQ)

    # Set sampling
    q.set_store_ports([sample_port])
    q.set_store_duration(sample_length)
    T0 = 0e-6  # s, start from some time
    T = T0

    # set template matching
    t_template_match = np.linspace(0, match_length, int(round(match_length * sampling_freq_adc)), False)
    carrier_m1_p1 = np.cos(2 * np.pi * readout_freq * t_template_match + np.pi * phase)
    carrier_m1_p2 = np.sin(2 * np.pi * readout_freq * t_template_match + np.pi * (phase + p2_phase_shift))
    carrier_m2_p1 = -np.sin(2 * np.pi * readout_freq * t_template_match + np.pi * phase)
    carrier_m2_p2 = np.cos(2 * np.pi * readout_freq * t_template_match + np.pi * (phase + p2_phase_shift))

    if use_template_matching:
        match_events_I = q.setup_template_matching_pair(input_port=sample_portI, template1=carrier_m1_p1,
                                                    template2=carrier_m1_p2, compare_next_port=True)
        match_events_Q = q.setup_template_matching_pair(input_port=sample_portI, template1=carrier_m2_p1,
                                                    template2=carrier_m2_p2, compare_next_port=True)

    # *** Program pulse sequence ***
    for i in range(nr1):
        q.reset_phase(T, [disp_port])

        q.select_scale(T, i, [disp_port], 0)
        q.output_pulse(T, [disp_pulse])
        T += disp_length

        q.reset_phase(T, [control_port])
        q.output_pulse(T, [control_pulse])
        T += control_length
        T += readout_delay

        q.output_pulse(T, [readout_pulse])
        # Sample
        q.store(T+sampling_start)
        if use_template_matching:
            q.match(T + sampling_start + matching_start, [match_events_I, match_events_Q])
        T += readout_length
        q.next_frequency(T + 2e-9, [disp_port], group=0)
        T += wait_decay

    expected_runtime = (T - T0) * nr * num_averages  # s
    print("Expected total runtime: {:s}".format(format_sec(expected_runtime)))
    q.run(T, nr, num_averages, print_time=True)
    t_array, result = q.get_store_data()
    raw_data.append(result)

    if use_template_matching:
        i_match_results = q.get_template_matching_data(match_events_I)
        q_match_results = q.get_template_matching_data(match_events_Q)
        raw_matches.append([i_match_results, q_match_results])

#sgs4a4.close()

if use_template_matching:
    i_data = i_match_results[0] + i_match_results[1]
    q_data = q_match_results[0] + q_match_results[1]

    avg_match_I_p = []
    avg_match_Q_p = []

    for ii in range(nr):
        avg_match_I_p.append(np.average(i_data[ii::nr]))
        avg_match_Q_p.append(np.average(q_data[ii::nr]))

    avg_match_I_p = np.array(avg_match_I_p) * match_length
    avg_match_Q_p = np.array(avg_match_Q_p) * match_length


start = int(round(matching_start * sampling_freq_adc))
end = start + len(carrier_m1_p1)
demod_I_p = []
demod_Q_p = []
demod_I_m = []
demod_Q_m = []
for i in range(nr):
    demod_I_p.append(np.sum(carrier_m1_p1 * result[i, 0, start:end].real + carrier_m1_p2 * result[i, 0, start:end].imag))
    demod_Q_p.append(np.sum(carrier_m2_p1 * result[i, 0, start:end].real + carrier_m2_p2 * result[i, 0, start:end].imag))

demod_I_p = np.array(demod_I_p)
demod_Q_p = np.array(demod_Q_p)

demod_I_p = demod_I_p.reshape(nr1,nr,order='C')
demod_Q_p = demod_Q_p.reshape(nr1,nr,order='C')




if save_to_file:
    _dict = dict()
    for key in dir():
        if isinstance(globals()[key], (np.ndarray, float, int, str)):
            _dict[key] = globals()[key]

    np.savez(file_name[:-3], **_dict)



## plotting
x_axis = disp_frequency

plt.figure()
#plt.plot(x_axis, demod_I_p * match_length, '.')
#plt.plot(control_amp, demod_Q_p * match_length, '.')
data = rotate_opt(demod_I_p+1j*demod_Q_p)
plt.plot(x_axis, np.real(data.T))

#cdata = np.real(data)

#plt.figure()
#plt.pcolor(cdata,cmap='RdBu_r',shading='auto')




#plt.plot(control_amp, avg_match_I_p, '.')
#plt.plot(control_amp, avg_match_Q_p, '.')

plt.show()

# close connections
#sgs4a1.close()
#sgs4a2.close()

