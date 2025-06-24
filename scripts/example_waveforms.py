import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from usnm2p.wfutils import *

''' Examples on how to use the waveform utilities module to construct complex waveforms '''

# Example usage of get_pulse_envelope function to get 
# single pulse envelopes with different tapewring and padding 
# (wihout explicit time resolution)
n = 100  # number of points in pulse window
envs = {'rect': get_pulse_envelope(n)}
for x in [.2, 1.]:
    envs[f'{x * 100:.0f} % tapered'] = get_pulse_envelope(n, xramp=x)
envs['50% tapered + 50% post-padding'] = get_pulse_envelope(n, xramp=0.5, xpostpad=0.5)
envs['50% tapered + 50% pre&post-padding'] = get_pulse_envelope(n, xramp=0.5, xprepad=0.5, xpostpad=0.5)
fig, ax = plt.subplots(figsize=(8, 3))
ax.set_title('envelope waveform')
ax.set_xlabel('samples')
ax.set_ylabel('relative amplitude')
for label, env in envs.items():
    ax.plot(env, label=label)
ax.legend()
sns.despine(ax=ax)

# Example usage of get_pulse_train_envelope function to get
# pulse train envelopes with different tapering and padding
# (without explicit time resolution)
npulses = 10  # number of pulses in the train
envs = {'rect': get_pulse_envelope(n, nreps=npulses)}
for x in [.2, 1.]:
    envs[f'{x * 100:.0f} % tapered'] = get_pulse_envelope(n, xramp=x, nreps=npulses)
envs['50% tapered + 50% post-padding'] = get_pulse_envelope(n, xramp=0.5, xpostpad=0.5, nreps=npulses)
envs['50% tapered + 50% pre&post-padding'] = get_pulse_envelope(n, xramp=0.5, xprepad=0.5, xpostpad=0.5, nreps=npulses)
fig, ax = plt.subplots(figsize=(8, 3))
ax.set_title('pulse train envelope waveform')
ax.set_xlabel('samples')
ax.set_ylabel('relative amplitude')
for label, env in envs.items():
    ax.plot(env, label=label)
ax.legend()
sns.despine(ax=ax)

# Example usage of get_waveform function to get US waveform 
# (and envelope) directly from waveform parameters, with
# different tapering and padding (with explicit time resolution)
Fdrive = 10e3 # carrier frequency (Hz)
A = 0.5  # waveform amplitude
dur = 0.2  # pulse duration (s)
DC = 70. # duty cycle (%)
PRF = 100. # pulse repetition frequency (Hz)
fig, ax = plt.subplots(figsize=(8, 3))
ax.set_title('pulse train envelope waveform')
ax.set_xlabel('time (ms)')
ax.set_ylabel('amplitude')
colors = sns.color_palette('tab10', n_colors=2)
for tramp, c in zip(np.array([0, 2]) * 1e-3, colors):  # ramp time (s)
    t, y, yenv = get_waveform(Fdrive, A, dur, PRF=PRF, DC=DC, tramp=tramp)
    ax.plot(t * 1e3, y, label=f'tramp = {tramp * 1e3:.2f} ms', c=c, alpha=0.5)
    ax.plot(t * 1e3, yenv, c=c)
    ax.plot(t * 1e3, -yenv, c=c)
ax.legend()
sns.despine(ax=ax)

plt.show()
