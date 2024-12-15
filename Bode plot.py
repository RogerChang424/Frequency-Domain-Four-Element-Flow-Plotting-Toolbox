# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:51:35 2024

@author: Roger Chang
 - What you are you do not see, what you see is your shadow.
 
# 2024,12,15 (1.0.1)
             update_1: phase function correction, update with TransferFunc module
             update_2: non-minimum phase system and invalid PM identification,
                       determining real PM's from gain cross phases within (-360, 0) degs
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import TransferFunc as TsF
import datetime

# current time for plot image name
current_time = datetime.datetime.now()
current_time = current_time.strftime("%Y_%m%d_%H%M%S")

# plot save directory
dir_name = "./Bode_plot/"


"""
reading settings from csv
"""

# import the parameters in col 1
settings = np.loadtxt("./settings/Bode plot.csv", delimiter=",", 
                      dtype=int, 
                      skiprows = 0, usecols = 1)

digits      = settings[0]
ang_freq    = True
samp_range  = settings[2]
samp_range  = np.float_power(10, samp_range)
nsamps      = settings[3]



GN_path = "./Transfer Function Data/G(s)_nominator.csv"
GD_path = "./Transfer Function Data/G(s)_denominator.csv"
HN_path = "./Transfer Function Data/H(s)_nominator.csv"
HD_path = "./Transfer Function Data/H(s)_denominator.csv"

tf = TsF.TF(GN_path, GD_path, HN_path, HD_path)

"""
plot figure initialization
"""
fig = plt.figure(figsize=(12, 12), dpi=300)


"""
sampling
"""

A, w, gain, phase = tf.s_posjw(samp_range, nsamps, ang_freq)

# since the arrays are reversed, flip them back
A     = np.flip(A)
w     = np.flip(w)
gain  = np.flip(gain)
phase = np.flip(phase)


fig = plt.figure()


if(ang_freq):
    unit = " rad/s"
else:
    unit = " Hz"




"""
gain margin
"""
GM, wp = tf.gain_margin(dB=True, rads=ang_freq)

TsF.splittingline(60)
print("Gain Margin:              " + str(TsF.setdigit(4).format(GM)) + " dB")
print("Phase crossover freq:     " + str(TsF.setdigit(4).format(wp)) + unit)
TsF.splittingline(60)

# phase crossover on neg real axis: finite PM
if(np.isfinite(GM)):
    PCgain = -GM
    PCph   = -180
# phase crossover at origin: 
# do not calculate GMpoint, or div/0 err will be raised
else:
    PCgain = -np.inf
    PCph   = np.nan



"""
phase margin
"""

PM, wg = tf.phase_margin(rad=False, rads=ang_freq)


print("Phase Margin(if min. ph): " + str(TsF.setdigit(4).format(PM)) + " degrees")
print("Gain crossover freq:      " + str(TsF.setdigit(4).format(wg)) + unit)
TsF.splittingline(60)

GCgain = 0 
GCph   = PM - 180


"""
plotting
"""

fig = plt.figure(figsize=(12, 9), dpi=300)
if(ang_freq):
    xlabel = "ω [rad/s]"
else:
    xlabel = "f [Hz]"
"""
plotting symbols size
"""
PCP_size = 7e1
GCP_size = 2e2
arrw_PC  = 4e-1
arrw_GC  = 4e-3
hl_PC    = 1e1
hl_GC    = 1e1
fs       = 15
lw       = 2.5
"""
gain curve
"""
plt.subplot(211)

# gain = 0 line
zero_gain_line = np.zeros_like(w)
plt.plot(w, zero_gain_line, 'k:')

plt.plot(w, gain, c='g')


annotext_PC = "   GM: " + str(round(GM, digits)) + " dB"
anno_PC = plt.annotate(annotext_PC, (wp, PCgain), fontsize=fs)
if(ang_freq):
    annotext_wp = "   ωp: " + str(round(wp, digits)) + " rad/s"
else:
    annotext_wp = "   fp: " + str(round(wp, digits)) + " Hz"
anno_wp = plt.annotate(annotext_wp, (wp, 0), fontsize=fs)

line_PC = plt.plot([wp, wp], [0, PCgain], c='r', linewidth=lw)



PCG    = plt.scatter(wp, PCgain, c='c', marker='s', s=PCP_size)
GCG    = plt.scatter(wg, GCgain, c='m', marker='*', s=GCP_size)

plt.xscale('log')
plt.xlabel(xlabel)
plt.ylabel("gain [dB]")
plt.xlim([w[0], w[-1]])

plt.title("Bode Plot", fontsize = 15)
plt.grid()



"""
phase curve
"""
plt.subplot(212)

# phase = -180 line
mpi_line = np.ones_like(w) * (-180)
plt.plot(w, mpi_line, 'k:')

plt.plot(w, phase, c='b')

annotext_GC = "   PM: " + str(round(PM, digits)) + " deg"
anno_GC = plt.annotate(annotext_GC, (wg, GCph), fontsize=fs)

if(ang_freq):
    annotext_wg = "   ωg: " + str(round(wg, digits)) + " rad/s"
else:
    annotext_wg = "   fg: " + str(round(wg, digits)) + " Hz"
                                  
anno_wg = plt.annotate(annotext_wg, (wg, -180), fontsize=fs)

line_GC = plt.plot([wg, wg], [-180, GCph], c='y', linewidth=lw)

PCP  = plt.scatter(wp, PCph, c='c', marker='s', s=PCP_size)
GCP  = plt.scatter(wg, GCph, c='m', marker='*', s=GCP_size)

plt.xscale('log')
plt.xlabel(xlabel)
plt.ylabel("phase [deg]")
plt.xlim([w[0], w[-1]])
plt.grid()

if(not os.path.isdir(dir_name)):
    os.mkdir(dir_name)
plt.savefig(dir_name + str(current_time) + ".jpg")
plt.show()

