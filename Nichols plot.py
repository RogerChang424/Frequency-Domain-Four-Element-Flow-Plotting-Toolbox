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
dir_name = "./Nichols_plot/"


"""
reading settings from csv
"""
# import the parameters in col 1
settings = np.loadtxt("./settings/Nichols plot.csv", delimiter=",", 
                      dtype=int, 
                      skiprows = 0, usecols = 1)

# use float_power to expand value range
# prevent overflow cases
digits      = settings[0]
ang_freq    = settings[1]
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
fig = plt.figure(figsize=(12, 9), dpi=300)


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

GC     = tf.substitute(1j * wg)
GCgain = 0 
GCph   = PM - 180


"""
plotting
"""

fig = plt.figure(figsize=(12, 9), dpi=300)

"""
plotting symbols size
"""
PCP_size = 2e2
GCP_size = 8e2
arrw_PC  = 3e0
arrw_GC  = 4e0
hl_PC    = 1e1
hl_GC    = 1e1
fs       = 15

"""
auto-fit plot boundaries
"""
xMax = np.max(phase)
xMin = np.min(phase)
if(xMax < 0):
    xMax = 0
if(xMin > -180):
    xMin = -180

yMax = np.max(gain)
yMin = np.min(gain)

if(yMin < -200):
    yMin = -200


"""
gain-phase curve
"""

if(ang_freq):
    fsymb = "ω"
else:
    fsymb = "f"

# curve
plt.plot(phase, gain,  c='g')

# gain crossover
PCP  = plt.scatter(PCph, PCgain, c='c', marker='s', s=PCP_size)

annotext_PC = "   GM: " + str(round(GM, digits)) + " dB, " + fsymb + "p = " + str(round(wp, digits)) + unit
anno_PC = plt.annotate(annotext_PC, (-180, PCgain), fontsize=fs)

arr_PC = plt.arrow(-180, PCgain, 0, GM, width=arrw_PC, 
                   length_includes_head=True, fc = 'r', head_length=hl_PC)

# gain crossover
GCP  = plt.scatter(GCph, GCgain, c='m', marker='*', s=GCP_size)
annotext_GC = "   PM: " + str(round(PM, digits)) + " deg, " + fsymb + "g = " + str(round(wg, digits)) + unit
anno_GC = plt.annotate(annotext_GC, (GCph, 0), fontsize=fs)

arr_GC = plt.arrow(-180, 0, PM, 0, width=arrw_GC, 
                   length_includes_head=True, fc = 'y', head_length=hl_GC)

# gain = 0 line
phase_arr = np.linspace(xMin, xMax, w.shape[0])
zero_gain = np.zeros_like(w)
plt.plot(phase_arr, zero_gain, 'k:')

# phase = -180 line
mpi_phase = np.ones_like(w) * (-180)
gain_arr = np.linspace(yMin, yMax, w.shape[0])
plt.plot(mpi_phase, gain_arr, 'k:')


plt.xlabel("phase [deg]")
plt.ylabel("gain   [dB]")


plt.xlim([xMin, xMax])
plt.ylim([yMin, yMax])

plt.title("Nichols plot", fontsize=28)
plt.grid()



if(not os.path.isdir(dir_name)):
    os.mkdir(dir_name)
plt.savefig(dir_name + str(current_time) + ".jpg")
plt.show()

