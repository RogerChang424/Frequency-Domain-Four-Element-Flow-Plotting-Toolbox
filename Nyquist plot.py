# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:51:35 2024

@author: Roger Chang
 - What you are you do not see, what you see is your shadow.
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
dir_name = "./Nyquist_plot/"


"""
reading settings from csv
"""

# import the parameters in col 1
settings = np.loadtxt("./settings/Nyquist plot.csv", delimiter=",", 
                      dtype=int, 
                      skiprows = 0, usecols = 1)

# use float_power to expand value range
# prevent overflow cases
digits      = settings[0]
ang_freq    = settings[1]
samp_radius = settings[2]
samp_radius = np.float_power(10, samp_radius)
nsamps      = settings[3]
fit_uc      = settings[4]



GN_path = "./Transfer Function Data/G(s)_nominator.csv"
GD_path = "./Transfer Function Data/G(s)_denominator.csv"
HN_path = "./Transfer Function Data/H(s)_nominator.csv"
HD_path = "./Transfer Function Data/H(s)_denominator.csv"

tf = TsF.TF(GN_path, GD_path, HN_path, HD_path)


"""
plot figure initialization
"""
fig = plt.figure(figsize=(12, 9), dpi=300)
ax  = fig.add_subplot(1, 1, 1)

"""
set x and y axes
"""
if(fit_uc):
    title = 'Nyquist Plot - fitting unit circle'
else:
    title = 'Nyquist Plot - fitting phase crossover point'
ax.set_title(title, fontsize=20)
# Move left y-axis and bottom x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

"""
nyquist plot
"""

nsamps = int(nsamps)
pjx, pjy = TsF.sepReIm(tf.s_posjw(samp_radius, nsamps//4, ang_freq)[0])
njx, njy = TsF.sepReIm(tf.s_negjw(samp_radius, nsamps//4, ang_freq)[0])
ncx, ncy = TsF.sepReIm(tf.s_negcir(samp_radius, nsamps//4)[0])
pcx, pcy = TsF.sepReIm(tf.s_poscir(samp_radius, nsamps//4)[0])
cirx, ciry = TsF.sepReIm(TsF.unit_cir())



PJs = plt.plot(pjx, pjy, c='g')[0]
NJs = plt.plot(njx, njy, c='b')[0]
NCs = plt.plot(ncx, ncy, c='r')[0]
PCs = plt.plot(pcx, pcy, c='y')[0]


"""
plotting symbols size
"""
PCP_size = 7e1
GCP_size = 2e2
arrw   = 4e-2

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
    if(not ang_freq):
        PC = tf.substitute(2* np.pi * 1j * wp)
    else:
        PC = tf.substitute(1j * wp) 

# phase crossover at origin: 
# do not calculate GMpoint, or div/0 err will be raised
else:
    PC = 0
PC_r = np.real(PC)
PC_i = np.imag(PC)
PCP = plt.scatter(PC_r, PC_i, c='c', marker='s', s=PCP_size)



"""
phase margin
"""

PM, wg = tf.phase_margin(rad=False, rads=ang_freq)

print("Phase Margin(if min. ph): " + str(TsF.setdigit(4).format(PM)) + " degrees")
print("Gain crossover freq:      " + str(TsF.setdigit(4).format(wg)) + unit)
TsF.splittingline(60)


if(not ang_freq):
    GC = tf.substitute(2* np.pi * 1j * wg)
else:
    GC = tf.substitute(1j * wg) 

GC_r = np.real(GC)
GC_i = np.imag(GC)



if(fit_uc or PC_r >= 0):
    xlim = 2
    ylim = 0.73 * xlim

else:
    xlim = -1.5 * PC_r 
    ylim = 0.73 * xlim

# if the plot includes the unit circle fully
# plot GCP with an arrow from orig to GCP
if(ylim > 1):
    # plot unit circle
    plt.plot(cirx, ciry, 'k:')
    # mark GCP
    GCP  = plt.scatter(GC_r, GC_i, c='m', marker='*', s=GCP_size)
    # arrow from 0+0j to gain crossover point
    arr_GC = plt.arrow(0, 0, GC_r, GC_i, width=arrw, 
                       length_includes_head=True, fc = 'y')
    
    # arrow from -1+0j to phase crossover point
    dirPC_r  = PC_r - (-1)
    dirPC_i  = PC_i - 0
    arr_PC = plt.arrow(-1, 0, dirPC_r, dirPC_i, width=arrw, 
                       length_includes_head=True, fc = 'r')
    
    plt.legend((PJs, NJs, NCs, PCs,  PCP, GCP, arr_PC, arr_GC),  
               ("s on positve  jω", 
                "s on negative jω",
                "s in 4th quarter",
                "s in 1st quarter", 
                "phase crossover",
                "gain crossover", 
                "gain margin", 
                "phase margin"), 
               fontsize="9")

# if the plot doesn't include the unit circle fully
# skip both GCP and the arrow to it
else:
    plt.legend(( PJs, NJs, NCs, PCs, PCP),  
               ("s on positve  jω", 
                "s on negative jω",
                "s in 4th quarter",
                "s in 1st quarter",
                "phase crossover"), 
               fontsize="9")

plt.xlim(-xlim, xlim)
plt.ylim(-ylim, ylim)

if(not os.path.isdir(dir_name)):
    os.mkdir(dir_name)
plt.savefig(dir_name + str(current_time) + ".jpg")
plt.show()
