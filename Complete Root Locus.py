# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:16:27 2024
author: Roger Chang 

 - What you are you do not see, what you see is your shadow.
"""

import os

import TransferFunc as TsF
import numpy as np
import matplotlib.pyplot as plt
import datetime

# current time for plot image name
current_time = datetime.datetime.now()
current_time = current_time.strftime("%Y_%m%d_%H%M%S")

# plot save directory
dir_name = "./Complete_Root_Locus/"

"""
reading settings from csv
"""

# import the parameters in col 1
settings = np.loadtxt("./settings/Complete Root Locus.csv", delimiter=",", 
                      dtype=np.float32, 
                      skiprows = 0, usecols = 1)
# kernel displaying precision
prec      = settings[0].astype(int) 
# xlim (ylim is fixed to 0.65 * xlim)
# set to -1 for automatically fit all special points
# with minimum xlim = 5
xlim      = settings[1].astype(int)

# sampling power range: from 1e(lLim) to 1e(hLim)
lLim      = settings[2]
hLim      = settings[3]
nsamps    = settings[4].astype(int)

"""
system transfer function
"""
# forward:  G(s) = k * GN(s)/GD(s)
# feedback: H(s) = HN(s)/HD(s)
# set with factors
# eg. 4 => [4],  s+1 => [1, 1], s^2+2s+4 => [1, 2, 4]

# load GN(s), GD(s), HN(s) and HD(s) from independent csv files
GN_path = "./Transfer Function Data/G(s)_nominator.csv"
GD_path = "./Transfer Function Data/G(s)_denominator.csv"
HN_path = "./Transfer Function Data/H(s)_nominator.csv"
HD_path = "./Transfer Function Data/H(s)_denominator.csv"

tf = TsF.TF(GN_path, GD_path, HN_path, HD_path)


"""
set sampling values
"""
# closed loop characteristic equation
# forward_gain 
# set to 10^(linspace)
k = np.linspace(lLim, hLim, nsamps)

"""
plot figure initialization
"""
fig = plt.figure(figsize=(12, 9), dpi=300)
ax  = fig.add_subplot(1, 1, 1)

"""
set x and y axes
"""
ax.set_title('Complete Root Locus')
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
locus with scattering roots 
"""
dot_size = 5

RLPs = TsF.sepReIm(tf.RL_poles(k, prec))
CRPs = TsF.sepReIm(tf.CR_poles(k, prec))

RL = ax.scatter(RLPs[0], RLPs[1], s = dot_size, c='c')
CR = ax.scatter(CRPs[0], CRPs[1], s = dot_size, c='y')

"""
mark special points and lines
"""
# symbols size
pole_size  = 1e2
zero_size  = 4e2
inter_size = 3e1
bwpcs_size = 8e1



# asymptotes intersection
angles       = tf.findAngles(prec)
intersection = tf.findInters(prec)

int_real, int_imag = TsF.sepReIm(intersection)
INTER = plt.scatter(int_real, int_imag, 
                    c = 'b', marker = 's', s=inter_size)
    
# breakaway point candidates
candidate_breakaway = tf.cand_bwps(prec)
cbwp_real, cbwp_imag = TsF.sepReIm(candidate_breakaway)
CBWP = plt.scatter(cbwp_real, cbwp_imag, 
                   c = 'b', marker = '*', s=bwpcs_size)

# marginally stable poles
msp_imag   = tf.solveJWpoles(prec)[0]
# real part: zero 
# create an array of same length containing zeros
msp_real = np.zeros_like(msp_imag)
MSP      = plt.scatter(msp_real, msp_imag, 
                       c = 'g', marker = 'P', s=bwpcs_size)

# plot open loop poles and zeros last
# prevent from being covered by other markers or loci

# open loop poles with scatter
OLPs_real, OLPs_imag = TsF.sepReIm(tf.OLPoles(prec))
OLP  = ax.scatter(OLPs_real, OLPs_imag, marker='x', c='r', s = pole_size)
# open loop zeros with scatter
OLZs_real, OLZs_imag = TsF.sepReIm(tf.OLZeros(prec))
OLZ = ax.scatter(OLZs_real, OLZs_imag, marker='.', facecolors='none', edgecolors='r', s = zero_size)


# set xlim and ylim
lims = TsF.RLsetXYLims(xlim,      1.25, 
                        OLPs_real, OLPs_imag, 
                        OLZs_real, OLZs_imag, 
                        int_real,  int_imag, 
                        cbwp_real, cbwp_imag,
                        msp_real,  msp_imag)
plt.xlim([-lims[0], lims[0]])
plt.ylim([-lims[1], lims[1]])

plt.legend((RL,    CR, 
            OLP,   OLZ, 
            INTER, CBWP,
            MSP), 
           ("k>0 (Root Locus)",   "k<0 (Comp. Root Locus)", 
            "Open Loop poles",    "Open Loop zeros", 
            "Asym. intersection", "Cand. breakaway pts.",
            "Poles on jω axis"), 
           fontsize="8")



if(not os.path.isdir(dir_name)):
    os.mkdir(dir_name)
plt.savefig(dir_name + str(current_time) + ".jpg")
plt.show()
