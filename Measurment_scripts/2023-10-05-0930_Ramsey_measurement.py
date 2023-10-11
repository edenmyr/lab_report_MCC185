import os
import sys
import time
import shutil
import pyvisa as visa
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('C:\\Users\\Lab\\anaconda3\\Lib\\site-packages\\')
from presto.utils import format_sec, get_sourcecode, rotate_opt
from presto import pulsed as pls
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from utils_presto import setup_IF_template, save_file

date_string = time.strftime('%Y-%m-%d-%H%M')
base_filename = os.path.basename(__file__)
file_name = "Data\\Coherence\\" + date_string + '_' + base_filename
sourcecode = get_sourcecode(__file__)
shutil.copy(base_filename, file_name)

# Option
use_template_matching = 0
use_fitting = 1
save_to_labber = 0
save_to_file = 1

## Time domain setup
num_averages = 1000
wait_decay = 300e-6  # delay between repetitions
dt = np.linspace(0, 40e-6, 101)
nr1 = 1  # repeats

# qubit drive: control
control_freq = 100e6  # Hz, IF frequency
control_shape = "sin2"
control_cutoff = 2
control_length = 100e-9
control_amp = 0.1743 / 2

# readout drive
readout_freq = 100e6  # Hz
readout_shape = "square"
readout_amp = 0.1
readout_amp_kick = 0.07
readout_length = 1200e-9  # s
readout_length_kick = 50e-9  # s
readout_delay = 0e-9

# sampling
sample_length = 900e-9
sampling_start = 0e-9
matching_start = 268e-9

# template matching / demodulation
match_length = 0.6e-6  # s
phase = 0
p2_phase_shift = 0

# LO settings
readout_LO =  7.7e9 + 43e6#7.743510000e9
qubit_LO = 5.5e9 + 38e6 #5.537790000e9
cavity_LO = 3.674278939e9

# digital mixer calibration
control_phaseI = 0.
control_phaseQ = control_phaseI + 0.5 * np.pi  # low sideband

readout_phaseI = + np.pi / 2
readout_phaseQ = readout_phaseI - 0.5 * np.pi  # high sideband

## Initialization
readout_port = 1
control_port = 2
sample_port = 1

nr = len(dt)

raw_data = []
raw_matches = []

for ii in range(nr1):
	with pls.Pulsed(ext_ref_clk=False, dry_run=False, force_config=True, address="192.168.88.72",
					adc_mode=AdcMode.Mixed,
					dac_mode=[DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
					adc_fsample=AdcFSample.G2, dac_fsample=[DacFSample.G10, DacFSample.G8, DacFSample.G8, DacFSample.G8]) as q:
		# *** Set parameters ***
		sampling_freq_dac = q.get_fs('dac')
		sampling_freq_adc = q.get_fs('adc')
		q.hardware.configure_mixer(freq=readout_LO, in_ports=sample_port, out_ports=readout_port)
		q.hardware.configure_mixer(freq=qubit_LO, out_ports=control_port, sync=True)
		q.hardware.set_dac_current(control_port, 32_500)

		# setup control
		_template_control_I, _template_control_Q = setup_IF_template(q, control_length, 0, control_shape, 0, 0,
																	 cutoff=control_cutoff)
		control_pulse = q.setup_template(control_port, 0, _template_control_I, _template_control_I, envelope=True)

		q.setup_scale_lut(control_port, group=0, scales=control_amp)
		q.setup_freq_lut(control_port, group=0, frequencies=control_freq, phases=control_phaseI,
						 phases_q=control_phaseQ)

		# setup readout
		_template_readout_kick_I, _template_readout_kick_Q = setup_IF_template(q, readout_length_kick, readout_freq,
																			   'square',
																			   readout_phaseI, readout_phaseQ,
																			   readout_amp_kick)
		readout_kick_pulse = q.setup_template(readout_port, 0, _template_readout_kick_I, _template_readout_kick_I,
											  envelope=False)

		_template_readout_I, _template_readout_Q = setup_IF_template(q, readout_length, 0, readout_shape, 0, 0)
		readout_pulse = q.setup_template(readout_port, 0, _template_readout_I, _template_readout_I, envelope=True)

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

		carrier_i = np.cos(2 * np.pi * readout_freq * t_template_match + np.pi * phase)
		carrier_q = np.sin(2 * np.pi * readout_freq * t_template_match + np.pi * (phase + p2_phase_shift))

		if use_template_matching:
			match_events_I = q.setup_template_matching_pair(input_port=sample_portI, template1=carrier_m1_p1,
															template2=carrier_m1_p2, compare_next_port=True)
			match_events_Q = q.setup_template_matching_pair(input_port=sample_portI, template1=carrier_m2_p1,
															template2=carrier_m2_p2, compare_next_port=True)

		# *** Program pulse sequence ***
		for i in range(nr):
			q.reset_phase(T, [control_port], group=0)
			q.output_pulse(T, [control_pulse])
			T += control_length
			T += dt[i]
			q.output_pulse(T, [control_pulse])
			T += control_length
			T += readout_delay
			q.reset_phase(T, [readout_port])
			q.output_pulse(T, [readout_pulse])
			# Sample
			q.store(T + sampling_start)
			if use_template_matching:
				q.match(T + sampling_start + matching_start, [match_events_I, match_events_Q])
			T += readout_length
			T += wait_decay

		expected_runtime = (T - T0) * nr * num_averages  # s
		print("Expected total runtime: {:s}".format(format_sec(expected_runtime)))
		q.run(T, 1, num_averages, print_time=True)
		t_array, _result = q.get_store_data()
		raw_data.append(_result)

	if use_template_matching:
		i_match_results = q.get_template_matching_data(match_events_I)
		q_match_results = q.get_template_matching_data(match_events_Q)
		raw_matches.append([i_match_results, q_match_results])


start = int(round(matching_start * sampling_freq_adc))
end = start + len(carrier_m1_p1)
demod_I_p = []
demod_Q_p = []
demod_I_m = []
demod_Q_m = []
for i in range(nr):
	demod_I_p.append(
		np.sum(carrier_m1_p1 * _result[i, 0, start:end].real + carrier_m1_p2 * _result[i, 0, start:end].imag))
	demod_Q_p.append(
		np.sum(carrier_m2_p1 * _result[i, 0, start:end].real + carrier_m2_p2 * _result[i, 0, start:end].imag))

# for i in range(nr):
#    demod_I_p.append(np.sum(carrier_i * _result[i, 0, start:end] + carrier_q * _result[i, 1, start:end]))
#    demod_Q_p.append(np.sum(-carrier_q * _result[i, 0, start:end] + carrier_i * _result[i, 1, start:end]))

demod_I_p = np.array(demod_I_p)
demod_Q_p = np.array(demod_Q_p)

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

if save_to_file:
	_dict = dict()
	for key in dir():
		if isinstance(globals()[key], (np.ndarray, float, int, str)):
			_dict[key] = globals()[key]

	np.savez(file_name[:-3], **_dict, raw_data=raw_data)


## plotting
xdata = dt

plt.figure()
# plt.plot(control_amp, demod_I_p * match_length, '.')
# plt.plot(control_amp, demod_Q_p * match_length, '.')
data, angle = rotate_opt(demod_I_p + 1j * demod_Q_p, True)
plt.plot(xdata, np.real(data))

if use_template_matching:
	plt.figure()
	data_matched = rotate_opt(avg_match_I_p + 1j * avg_match_Q_p)
	plt.plot(xdata, np.real(data_matched))

if use_fitting:
	import scipy.optimize as opt


	def func(t, A, f, T2, phi, B):
		return A * np.exp(-t / T2) * np.cos(2 * np.pi * t * f + phi) + B


	ydata = np.real(data)
	t = dt * 1e6
	A = max(ydata) - min(ydata)
	B = np.mean(ydata)
	ff = np.fft.fftfreq(len(t), (t[1] - t[0]))  # assume uniform spacing
	Fyy = abs(np.fft.fft(ydata))
	f = abs(ff[np.argmax(Fyy[1:]) + 1])
	phi = 0
	T2 = 20
	# p0 = [np.max(ydata)-np.min(ydata), (t[np.argmax(ydata)]-t[np.argmin(ydata)]), 20, 0, np.mean(ydata)]
	popt, pcov = opt.curve_fit(func, t, ydata, p0=[A, f, T2, phi, B])
	plt.plot(xdata, func(t, *popt), label=r'Freq=%.0f ' % (popt[1] * 1e6) + '\n' + r'T2=%.4f ' % (popt[2]))
	plt.legend()

# plt.plot(control_amp, avg_match_I_p, '.')
# plt.plot(control_amp, avg_match_Q_p, '.')

plt.show()



