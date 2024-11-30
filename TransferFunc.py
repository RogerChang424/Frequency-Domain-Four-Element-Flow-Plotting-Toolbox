# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:19:58 2024
author: Roger Chang 

 - What you are you do not see, what you see is your shadow.
"""

from tqdm import tqdm
import numpy as np


line_len = 60

"""
basic functions
"""

# print splitting line
def splittingline(length):
    for i in range (length-1):
        print('-', end = '')
    print('-')
    return 0

# set display precision
def setdigit(x):
    return "{:10." + str(x) + "f}"

# load factors in a csv file, export in np array
def loadFactors(path):
    F_raw = np.loadtxt(path, delimiter=",",
                        dtype=str, skiprows = 1)
    # replace all empty elems with 0
    F_raw[F_raw == ''] = 0
    # convert to float
    F = F_raw.astype(np.float32)
    return F

# multiply the factors in an array, return product polymial in 1D array 
def factor2product(factors):
    # 1-D cases (only one factor)
    if(len(factors.shape) == 1):
        return factors
    else:
        product = np.array([1])
        for factor in range (factors.shape[0]):
            product = np.polymul(product, factors[factor, :])
        return product

# load nominator and denominator in a transfer function with factors in csv files
def factors2TF(GN_factors, GD_factors, HN_factors, HD_factors):
    # forward gain: G(s) = GN(s)/GD(s)
    GN = factor2product(GN_factors)
    GD = factor2product(GD_factors)
    # feedback gain: H(s) = HN(s)/HD(s)
    HN = factor2product(HN_factors)
    HD = factor2product(HD_factors)
    # loop gain L = G * H
    LN = np.polymul(GN, HN)
    LD = np.polymul(GD, HD)
    return LN, LD

# frequency response: substitute s with jw
# return: real and imag part of freq resp function (func of w, discard j)
def freqResp(poly):
    poly_len = poly.shape[0]
    real_filter = np.zeros(poly_len)
    imag_filter = np.zeros(poly_len)
    for i in range (poly_len):
        deg = (poly_len-1) - i
        phase = deg % 4
        if(phase == 0):
            real_filter[i] =  1 
        elif(phase == 1):
            imag_filter[i] =  1
        elif(phase == 2):
            real_filter[i] = -1
        elif(phase == 3):
            imag_filter[i] = -1
        else:
            print("Unknown phase found.")
    real = poly * real_filter
    imag = poly * imag_filter

    return real, imag

# separate the real and imag parts of values in orig array into two arrays
# for plotting uses
# eg. input [1+1j, 1-1j], return [1, 1], [1, -1]
def sepReIm(val):
    val = val.astype(complex)
    re = np.real(val)
    im = np.imag(val)
    return re, im

# generate a unit circle point array on s plain
def unit_cir():
    rad = np.linspace(0, 2*np.pi, 10000)
    points = np.exp(1j * rad)
    return points

"""
class: TF (transfer func)
"""
class TF:
    # initialize with coefficient csv paths
    def __init__(self, GN_path, GD_path, HN_path, HD_path):
        GN_factors = loadFactors(GN_path)
        GD_factors = loadFactors(GD_path)
        HN_factors = loadFactors(HN_path)
        HD_factors = loadFactors(HD_path)
        self.LN, self.LD = factors2TF(GN_factors, GD_factors, HN_factors, HD_factors)
    
    def substitute(self, s):
        nom  = np.polyval(self.LN, s)
        dnom = np.polyval(self.LD, s)
        
        val  = nom / dnom
        return val
    
    def freq_gain(self, w):
        return np.absolute(self.substitute(1j * w))
    
    def freq_phase(self, w):
        # np.angel range: [-pi, pi]
        phase = np.angle(self.substitute(1j * w))
        # convert to [0, 2pi]
        if(phase < 0):
            phase = phase + 2 * np.pi
        return phase
    
    def OLPoles(self, precision):
        OLPs = np.roots(self.LD).astype(complex)

        if(OLPs.shape[0] == 0):
            print("Open loop poles: None")
        else:
            print("Open loop poles:")
            for pole in range (OLPs.shape[0]):
                print("  OLP. " + str(pole + 1) + ":  s = " + str(np.round(OLPs[pole], precision)))
        splittingline(line_len)
        return OLPs
        
    def OLZeros(self, precision):
        OLZs = np.roots(self.LN).astype(complex)

        if(OLZs.shape[0] == 0):
            print("Open loop zeros: None")
        else:
            print("Open loop zeros:")
            for zero in range (OLZs.shape[0]):
                print("  OLZ. " + str(zero + 1) + ":  s = " + str(np.round(OLZs[zero], precision)))
        splittingline(line_len)
        return OLZs

    """
    asymptotes
    """
    # intersections
    def findInters(self, precision):
        OLZs = np.roots(self.LN)
        OLPs = np.roots(self.LD)
        m = OLZs.shape[0]
        n = OLPs.shape[0]
        asym_nums = n - m
        if(asym_nums > 0):
            sumPs = np.sum(OLPs)
            sumZs = np.sum(OLZs)
    
            intersect = (sumPs - sumZs)/asym_nums
            if(intersect >= 0):
                blk = " "
            else:
                blk = ""
            print("Intersection of asymptotes:")
            print("  Inters:  s = " + blk + str(np.round(intersect, precision)))
            splittingline(line_len)
            return np.array([intersect])
        else:
            print("Intersection of asymptotes:     None")
            splittingline(line_len)
            return np.array([])
    # angles
    def findAngles(self, precision):
        OLZs = np.roots(self.LN)
        OLPs = np.roots(self.LD)
        n    = OLPs.shape[0]
        m    = OLZs.shape[0]
        asym_nums = n - m
        if(asym_nums > 0):
            k = np.arange(asym_nums)
            RL_angs_unit_pi = (2*k + 1)/asym_nums
            CR_angs_unit_pi =  2*k     /asym_nums
            print("Angles of asymptotes (k>0):")
            for i in range (asym_nums):
                print("  θr_" + str(i+1) + " = " 
                      + str(np.round(RL_angs_unit_pi[i], precision))
                      + " π")
            print()
            print("Angles of asymptotes (k<0):")
            for j in range (asym_nums):
                print("  θc_" + str(j+1) + " = " 
                      + str(np.round(CR_angs_unit_pi[j], precision))
                      + " π")
            splittingline(line_len)
            return RL_angs_unit_pi * np.pi, CR_angs_unit_pi * np.pi
        else: 
            print("Angles of asymptotes (k>0): None")
            print("Angles of asymptotes (k<0): None")
            splittingline(line_len)
            return np.array([]), np.array([])
    
    """
    candidates of breakaway points
    """
    def cand_bwps(self, precision):
        # bwps occurrs at 1 + k * LN/LD = 0 has 2 or more identical roots
        # d/ds(1 + k * LN/LD) must equal to 0 when s = identical root value
        # d/ds(1 + k * LN/LD) = d/ds(LN/LD) = (LN' * LD - LD' * LN)/LD**2
        # LD ** 2 doesn't affect whether d/ds(LN/LD) = 0
        # solve (LN' * LD - LD' * LN) = 0
        dLN = np.polyder(self.LN, 1)
        dLD = np.polyder(self.LD, 1)
        eqLHS = np.polysub(np.polymul(dLN, self.LD), 
                           np.polymul(dLD, self.LN)) 
        cands = np.roots(eqLHS)
        if(cands.shape[0] > 0):
            print("Candidates of breakaway points:")
            for cand in range (cands.shape[0]):
                if(cands[cand] >= 0):
                    blk = " "
                else:
                    blk = ""
                print("  cand. " + str(cand + 1) + ": s = " + blk + str(np.round(cands[cand], precision)))
            splittingline(line_len)
            return cands
        else:
            print("Candidates of breakaway points: None")
            splittingline(line_len)
            return np.array([])
    
    """
    root locus
    """
    # forward_gain choosing range:
    # center: the gain that the highest degree term in k * LN
    #         has equal coefficeint in LD
    
    # define as k0 = LD[0]/LN[0]
    # sampling range: k0 * 10^k, k within [lLim, hLim]
    # with given sample amount
    
    # Root Locus - sampling roots in K>0
    def RL_poles(self, k, precision):
        k0 = self.LD[0]/self.LN[0]
        RL_CLPs  = np.zeros(1)
        print("Sampling root locus (k>0)...")
        for samp in tqdm(range (k.shape[0])):
            forward_gain = np.power(10, k[samp]) * k0
            GCL_CE = np.polyadd(forward_gain * self.LN, self.LD)
            samp_poles = np.roots(GCL_CE)
            # appending poles
            if(samp == 0):
                RL_CLPs = samp_poles
            else:    
                RL_CLPs = np.concatenate([RL_CLPs, samp_poles],
                                         axis=None)
        splittingline(line_len)
        return RL_CLPs
        
    # Complementary Root Locus - sampling roots in K>0
    def CR_poles(self, k, precision):
        k0 = self.LD[0]/self.LN[0]
        CR_CLPs = np.zeros(1)
        print("Sampling complementary root locus (k<0)...")
        for samp in tqdm(range (k.shape[0])):
            forward_gain = -1 * np.power(10, k[samp]) * k0
            GCL_CE = np.polyadd(forward_gain * self.LN, self.LD)
            samp_poles = np.roots(GCL_CE)
            # appending poles
            if(samp == 0):
                CR_CLPs = samp_poles
            else:    
                CR_CLPs = np.concatenate([CR_CLPs, samp_poles], axis=None)
        splittingline(line_len)
        return CR_CLPs

    """
    root locus - marginally stable points
    """
    def solveJWpoles(self, precision):
        """
        poly_real(w) = Re(poly(s=jw))
        poly_imag(w) = Im(poly(s=jw))
                
        eq1: P_real + k * Z_real = 0
        eq2: P_imag + k * Z_imag = 0
        
        <=> k = -P_real/Z_real = - P_imag/Z_imag
        <=> P_real / Z_real = P_imag / Z_imag
        <=> P_real * Z_imag = P_imag * Z_real
        """
        Z_real, Z_imag = freqResp(self.LN)
        P_real, P_imag = freqResp(self.LD)
        lhs = np.polymul(P_real, Z_imag)
        rhs = np.polymul(P_imag, Z_real)
        eq  = np.polysub(lhs, rhs)
        
        # only keep real w's
        marg_w = np.roots(eq)
        real_w = np.isreal(marg_w)
        marg_w = marg_w[real_w]
        marg_w = np.real(marg_w)
        # case: no real w's
        if(marg_w == []):
            print("Poles on jω axis: None")
            return []
        else:            
            num_marg = int(marg_w.shape[0])
            print("Poles on jω axis: " + str(num_marg) + " found.")
            marg_k = -1 * (np.polyval(P_real, marg_w) 
                           / np.polyval(Z_real, marg_w))

            for marg in range (num_marg):
                w = str(np.round(marg_w[marg], precision))
                k = str(np.round(marg_k[marg], precision))
                # fill a blank for non-neg j's to align
                if(float(np.real(w)) >= 0):
                    blk = " "
                else:
                    blk = ""
                print("  pole " + str(marg + 1) + ":  s = " + blk + w + "j" + ", k = " + k)
        splittingline(line_len)
        return marg_w, marg_k

    """
    nyquist plot sampling
    """
    def s_posjw(self, samp_radius, nsamps, rads):
        # sampling on -jw axis, in exponential
        # sampling on +jw axis
        # counter clockwise (+j * inf to j0)
        power_range = np.log10(samp_radius)
        power = np.linspace(-power_range, power_range, num = nsamps)
        w     = np.float_power(10, power)
        if(not rads):
            w *= 2 * np.pi
        samps = 1j * w

        A = np.polyval(self.LN, samps)/np.polyval(self.LD, samps)
        if(not rads):
            return A, w/(2*np.pi)
        else:
            return A, w
    
    def s_negjw(self, samp_radius, nsamps, rads):
        # sampling on -jw axis, in exponential
        # counter clockwise (j0 * inf to -j * inf)
        power_range = np.log10(samp_radius)
        power = np.linspace(-power_range, power_range, num = nsamps)
        w     = -np.float_power(10, power)
        if(not rads):
            w *= 2 * np.pi
        samps = 1j * w



        A = np.polyval(self.LN, samps)/np.polyval(self.LD, samps)
        if(not rads):
            return A, w/(2*np.pi)
        else:
            return A, w
    
    def s_negcir(self, samp_radius, nsamps):
        # sampling on the large half-circle in 4th quarter
        # sampling angle from -0.5pi to 0
        # counter clockwise
        ang_cir = 1j * np.linspace(-0.5 * np.pi, 0, num=nsamps)
        samps = np.exp(ang_cir) * samp_radius
        A = np.polyval(self.LN, samps)/np.polyval(self.LD, samps)
        return A, samps
    
    def s_poscir(self, samp_radius, nsamps):
        # sampling on the large half-circle in 4th quarter
        # sampling angle from 0 to +0.5 * pi
        # counter clockwise
        ang_cir = 1j * np.linspace(0, 0.5 * np.pi, num=nsamps)
        samps = np.exp(ang_cir) * samp_radius
        A = np.polyval(self.LN, samps)/np.polyval(self.LD, samps)
        return A, samps
    
    """
    Gain marg. and Phase marg.
    
    A = N(jw)/D(jw)
    let N(jw) = Nr(w) + jNi(w)
        D(jw) = Dr(w) + jDi(w)
    
    Then, A = (Nr + Ni)/(Dr + Di)
            = ((NrDr + NiDi) + j(NiDr-NrDi))/(Dr**2 + Di**2)
    let (Dr**2 + Di**2) = D
    Thus, A = (NrDr + NiDi)/D + j(NiDr-NrDi)/D
    
    Since Dr**2 + Di**2 >= 0
    Arg(A) = arctan((NiDr - NrDi)/(NrDr + NiDi))
    |A|    = ((NrDr + NiDi)/D)**2 + ((NiDr-NrDi)/D)**2
           = ((NrDr + NiDi)**2 + (NiDr-NrDi)**2)/D**2
    
    gain marg. |A| @ Arg(A) = -pi
        NiDr - NrDi = 0
        NrDr + NiDi < 0
    
    phase marg. Arg(A) - pi @ |A| = 1
        ((NrDr + NiDi)**2 + (NiDr-NrDi)**2)/D**2 = 1
        ((NrDr + NiDi)**2 + (NiDr-NrDi)**2) = (Dr**2 + Di**2) ** 2
        (NrDr + NiDi)**2 + (NiDr-NrDi)**2 - Dr**2 - Di**2 = 0
    """
    def NrDrPlusNiDi(self):
        Nr, Ni = freqResp(self.LN)
        Dr, Di = freqResp(self.LD)
        
        nrdr = np.polymul(Nr, Dr)
        nidi = np.polymul(Ni, Di)
        result = np.polyadd(nrdr, nidi)
        return result
    
    def NiDrMinusNrDi(self):
        Nr, Ni = freqResp(self.LN)
        Dr, Di = freqResp(self.LD)
        
        nidr = np.polymul(Ni, Dr)
        nrdi = np.polymul(Nr, Di)
        result = np.polysub(nidr, nrdi)
        return result
    
    def gain_margin(self, dB, rads):
        # solve |A(jwp)| @ Arg(A(jwp)) = -pi
        # NiDr - NrDi = 0
        # NrDr + NiDi < 0
        cand_wps = np.roots(self.NiDrMinusNrDi())

        # only keep real positive wp's
        real_wps = np.isreal(cand_wps)
        cand_wps = np.real(cand_wps[real_wps])
        pos_wps  = (cand_wps > 0)
        cand_wps = cand_wps[pos_wps]
        # only keep wp's such that NrDr + NiDi < 0 
        # (non-pos real part)
        npos_wps = (np.polyval(self.NrDrPlusNiDi(), cand_wps) < 0)
        wps = cand_wps[npos_wps]
        # remove the repeated values
        wps = np.unique(wps)
        
        if(wps.shape[0] == 0):
            #print("Gain Margin: infinity!")
            return np.array([np.inf, np.NaN])
        
        elif(wps.shape[0] == 1):
            wp    = wps[0]

            Dnom  = np.polyval(self.LD, 1j * wp)
            

            if(Dnom == 0):
                #print("Gain Margin: infinity!")
                if(not rads):
                    wp /= (2 * np.pi)
                return np.array([np.inf, wp])
            
            else:
                GM    = 1/self.freq_gain(wp)
                GM_dB = 20 * np.log10(GM)
                if(not rads):
                    wp /= (2 * np.pi)
                if(dB):
                    return np.array([GM_dB, wp])
                else:
                    return np.array([GM,    wp])
        else:
            print("more than one wp's were found!")
            print(wps)
            wp    = wps[0]
            GM    = self.freq_gain(wp)
            GM_dB = 20 * np.log10(GM)
            if(not rads):
                wp /= (2 * np.pi)
            if(dB):
                return np.array([GM_dB, wp])
            else:
                return np.array([GM,    wp])

    def phase_margin(self, rad, rads):
        # solve Arg(A(jwp)) @ |A(jwp)| = 0 
        # ((NrDr + NiDi)**2 + (NiDr-NrDi)**2) = (Dr**2 + Di**2)
        NrDrPNiDi  = self.NrDrPlusNiDi()
        NiDrMNrDi  = self.NiDrMinusNrDi()
        Nr, Ni     = freqResp(self.LN)
        Dr, Di     = freqResp(self.LD)
        NrDrPNiDi2 = np.polymul(NrDrPNiDi, NrDrPNiDi)
        NiDrMNrDi2 = np.polymul(NiDrMNrDi, NiDrMNrDi)

        Dr2        = np.polymul(Dr, Dr)
        Di2        = np.polymul(Di, Di)
        Dr2PDi2    = np.polyadd(Dr2, Di2)
        LHS   = np.polyadd(NrDrPNiDi2, NiDrMNrDi2)
        RHS   = np.polymul(Dr2PDi2, Dr2PDi2)
        poly  = np.polysub(LHS, RHS)
        cand_wgs = np.roots(poly)
        
        # only keep real positive wg's
        real_wgs = np.isreal(cand_wgs)
        cand_wgs = np.real(cand_wgs[real_wgs])
        pos_wgs  = (cand_wgs > 0)
        wgs      = cand_wgs[pos_wgs]
        
        # remove the repeated values
        wgs = np.unique(wgs)
        if(wgs.shape[0] == 0):
            print("Phase Margin: invalid!")
            return np.array([])
        
        elif(wgs.shape[0] == 1):
            wg     = wgs[0]
            Nom    = np.polyval(self.LN, 1j * wg)
            Dnom   = np.polyval(self.LD, 1j * wg)
            Phase  = np.angle(Nom/Dnom)
            # np.angel range: [-pi, pi]
            # convert to [0, 2pi]
            if(Phase < 0):
                Phase = Phase + 2 * np.pi
            PM = Phase - np.pi
            PM_deg = PM/np.pi * 180
            if(not rads):
                wg /= (2 * np.pi)
            if(rad):
                return np.array([PM, wg])
            else:
                return np.array([PM_deg, wg])
        else:
            print("more than one wg's were found!")
            print(wgs)
            wg     = wgs[0]
            Nom    = np.polyval(self.LN, 1j * wg)
            Dnom   = np.polyval(self.LD, 1j * wg)
            Phase  = np.angle(Nom/Dnom)
            # np.angel range: [-pi, pi]
            # convert to [0, 2pi]
            if(Phase < 0):
                Phase = Phase + 2 * np.pi
            PM = Phase - np.pi
            PM_deg = PM/np.pi * 180
            if(not rads):
                wg /= (2 * np.pi)
            if(rad):
                return np.array([PM, wg])
            else:
                return np.array([PM_deg, wg])

"""
root locus plotting boundary - xlim and ylim
"""

# xlim and ylim: maximum and minimum of open loop poles and zeros
# fixed aspect ratio: h:w = 3:4
# if xlim is set to -1, adjust automatically, minimum range is [-5, 5] for real axis
# er: expansion ratio
def RLsetXYLims(xlim, er, 
              OLPs_real, OLPs_imag, 
              OLZs_real, OLZs_imag, 
              int_real,  int_imag, 
              cbwp_real, cbwp_imag, 
              msp_real,  msp_imag):  
    # aspect_ratio = h/w
    aspect_ratio = 0.73
    if(xlim == -1):
        points_real = np.concatenate([OLPs_real, OLZs_real, int_real, cbwp_real, msp_real])
        points_imag = np.concatenate([OLPs_imag, OLZs_imag, int_imag, cbwp_imag, msp_imag])
        real_max = np.max(np.absolute(points_real))
        imag_max = np.max(np.absolute(points_imag))

        if(imag_max > aspect_ratio * real_max):
            if(imag_max < aspect_ratio * 5 * er):
                imag_max = aspect_ratio * 5 * er
            imag_max = imag_max * er
            real_max = imag_max / aspect_ratio
        else:
            if(real_max < 5 * er):
                real_max = 5
            real_max = real_max * er
            imag_max = real_max * aspect_ratio
        return real_max, imag_max
    else:
        return xlim, xlim * aspect_ratio