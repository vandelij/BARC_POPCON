import popcon
import scipy.integrate
import scipy.optimize
import numpy as np
import pint
ureg = pint.get_application_registry()
from matplotlib import pyplot as plt
import matplotlib.patches
import time
import os
import imageio

T_at_0 = 1
n_at_0 = 1e20 # m^(-3)

desired_power = 2150 * ureg.MW

# Exponential decay time constant for DT ratio evolution
tau_DT = {2:10000, 3:10000}


T_change = 16 #keV
final_T = 18

plot_dWdt_contour = True

inputs = {}
kr_frac = 1e-4

final_Kr_frac = 6e-4


inputs['reaction'] = 'DT'
inputs['f_He'] = 0.02
inputs['impurities'] = {'Kr': [36, kr_frac]}
inputs['DT_ratio'] = 1.0
# inputs['impurities'] = None

inputs['major_radius'] = 4.8 * ureg.meter
inputs['inverse_aspect_ratio'] = 1.35/4.8
inputs['areal_elongation'] = 1.8
inputs['plasma_current'] = 16.0 * ureg.MA
inputs['magnetic_field_on_axis'] = 8.66 *ureg.tesla

inputs['confinement'] = {}
# Scaling options are 'ITER98y2', 'ITER97', or 'ITER89'
inputs['confinement']['scaling'] = 'ITER89'
inputs['confinement']['H'] = 1.15
inputs['confinement']['lower_bound'] = 1e-5
inputs['confinement']['upper_bound'] = 5000
inputs['A'] = 2.5

# Method for calculating P_SOL
inputs['P_SOL_method'] = 'partial'

inputs['num_r_points'] = 20

# Electron Temperature Inputs
inputs['T_min'] = 1 * ureg.keV
inputs['T_max'] = 30 * ureg.keV
inputs['num_T_points'] = 35
inputs['profile_alpha'] = {}
inputs['profile_alpha']['T'] = 1.75
inputs['T_edge'] = 0.1 * ureg.keV

# Electron Density Inputs
inputs['n_min'] = 0.1e20 * ureg.meter**(-3)
inputs['n_max'] = 6e20 * ureg.meter**(-3)
inputs['num_n_points'] = 30
inputs['profile_alpha']['n'] = 1.5
inputs['n_edge_factor'] = 0.4

inputs['P_aux_0'] = 20 * ureg.MW
starting_p_aux_0 = inputs['P_aux_0']



inputs['minor_radius'] = inputs['major_radius'] * inputs['inverse_aspect_ratio']

inputs['reduce_P_aux'] = False

# def get_Kr_frac(T0, n0):
#     # f_Kr = 5e-5
#     # f_Kr = 5e-5 * T0**(0.5)
#     f_Kr = 2e-4 * (T0)**(-1) * (n0/1e20 - 1.0)**(2.0)
#     return f_Kr

def get_P_aux(t, P_aux_0, reduce=False):
    if reduce:
        P_aux = P_aux_0 - 0.1*ureg.MW * t
        # P_aux = P_aux_0
        if P_aux < 0:
            P_aux = 0 * ureg.MW
    else:
        P_aux = P_aux_0
    return P_aux


def get_f_DT(t, final_f_DT=1.0, time_constant=5, starting_f_DT=1.0):
    # if t<25:
    #     f_DT = 1.0 + 0.0402 * t
    # else:
    #     f_DT = 1.0 + 0.0402 * 25
    f_DT = (starting_f_DT - final_f_DT) * np.exp(-t/time_constant) + final_f_DT

    return f_DT



def get_p_ohmic(n0, T0, inputs):
    T0 *= ureg.keV
    n0 *= ureg.m**(-3)
    rs = np.linspace(0, inputs['minor_radius'], inputs['num_r_points'])
    Ts = popcon.get_parabolic_profile(T0, rs, inputs['minor_radius'],
                                      inputs['T_edge'],
                                      alpha=inputs['profile_alpha']['T'])
    ns = popcon.get_parabolic_profile(n0, rs, inputs['minor_radius'],
                                      inputs['n_edge_factor']*n0,
                                      alpha=inputs['profile_alpha']['n'])
    p_ohmic = popcon.get_p_ohmic_classical(Ts, ns, rs, inputs['plasma_current'],
                                           inputs['major_radius'], inputs['minor_radius'],
                                           inputs['areal_elongation'], reaction=inputs['reaction'],
                                           impurities=inputs['impurities'], f_He=inputs['f_He'])
    return p_ohmic


def get_p_fusion(n0, T0, inputs):

    T0 *= ureg.keV
    n0 *= ureg.m**(-3)
    rs = np.linspace(0, inputs['minor_radius'], inputs['num_r_points'])
    Ts = popcon.get_parabolic_profile(T0, rs, inputs['minor_radius'],
                                      inputs['T_edge'],
                                      alpha=inputs['profile_alpha']['T'])
    ns = popcon.get_parabolic_profile(n0, rs, inputs['minor_radius'],
                                      inputs['n_edge_factor']*n0,
                                      alpha=inputs['profile_alpha']['n'])
    if isinstance(inputs['impurities'], dict):
        p_radiation, z_dilutes = popcon.get_p_line_rad(ns, Ts, inputs['impurities'],
                                                    rs, inputs['major_radius'], inputs['areal_elongation'],
                                                    reaction=inputs['reaction'], f_DT=inputs['DT_ratio'])
        # print(z_dilutes)
    else:
        z_dilutes = None

    p_fusion = popcon.get_p_fusion(ns, Ts, rs, inputs['areal_elongation'],
                                   inputs['major_radius'], reaction=inputs['reaction'],
                                   impurities=inputs['impurities'],
                                    f_DT=inputs['DT_ratio'],
                                    f_He=inputs['f_He'],
                                    z_dilutes=z_dilutes)
    return p_fusion.to(ureg.MW)


def get_p_fusion_root(f_DT, n0, T0, inputs, desired_power):
    # inputs['impurities']['Kr'][1] = get_Kr_frac(T0, n0)
    T0 *= ureg.keV
    n0 *= ureg.m**(-3)
    inputs['DT_ratio'] = f_DT
    rs = np.linspace(0, inputs['minor_radius'], inputs['num_r_points'])
    Ts = popcon.get_parabolic_profile(T0, rs, inputs['minor_radius'],
                                      inputs['T_edge'],
                                      alpha=inputs['profile_alpha']['T'])
    ns = popcon.get_parabolic_profile(n0, rs, inputs['minor_radius'],
                                      inputs['n_edge_factor']*n0,
                                      alpha=inputs['profile_alpha']['n'])
    
    if isinstance(inputs['impurities'], dict):
        p_radiation, z_dilutes = popcon.get_p_line_rad(ns, Ts, inputs['impurities'],
                                                    rs, inputs['major_radius'], inputs['areal_elongation'],
                                                    reaction=inputs['reaction'], f_DT=inputs['DT_ratio'])
        # print(z_dilutes)
    else:
        z_dilutes = None
    p_fusion = popcon.get_p_fusion(ns, Ts, rs, inputs['areal_elongation'],
                                   inputs['major_radius'], reaction=inputs['reaction'],
                                   impurities=inputs['impurities'],
                                    f_DT=inputs['DT_ratio'],
                                    f_He=inputs['f_He'],
                                    z_dilutes=z_dilutes)
    root = p_fusion.to(ureg.MW).magnitude - desired_power.to(ureg.MW).magnitude
    return root


def get_p_LH(n0, inputs):
    rs = np.linspace(0, inputs['minor_radius'], inputs['num_r_points'])
    ns = popcon.get_parabolic_profile(n0*ureg.m**(-3), rs, inputs['minor_radius'],
                                      inputs['n_edge_factor']*n0*ureg.m**(-3),
                                      alpha=inputs['profile_alpha']['n'])
    n_line_ave = np.mean(ns)
    # p_LH  = popcon.get_p_LH_transition(n_line_ave, inputs['magnetic_field_on_axis'],
    #                                    inputs['major_radius'], inputs['minor_radius'])
    p_LH = popcon.get_p_LH_transition_3(n_line_ave, inputs['magnetic_field_on_axis'],
                                        inputs['major_radius'], inputs['minor_radius'], 
                                        inputs['areal_elongation'],
                                        reaction=inputs['reaction'], impurities=inputs['impurities'])
    return p_LH.to(ureg.MW)


def get_powers(n0, t, T0, inputs, p_fusion=None):
    T0 *= ureg.keV
    n0 *= ureg.m**(-3)
    rs = np.linspace(0, inputs['minor_radius'], inputs['num_r_points'])
    Ts = popcon.get_parabolic_profile(T0, rs, inputs['minor_radius'],
                                      inputs['T_edge'],
                                      alpha=inputs['profile_alpha']['T'])
    T_ave = popcon.get_volume_average(rs, Ts, inputs['major_radius'], 
                                  inputs['minor_radius'], inputs['areal_elongation']).to(ureg.keV)
    T_line_ave = np.mean(Ts)
    n_ave = popcon.get_n_ave(n0, inputs['n_edge_factor'], inputs['profile_alpha']['n'])

    ns = popcon.get_parabolic_profile(n0, rs, inputs['minor_radius'],
                                      inputs['n_edge_factor']*n0,
                                      alpha=inputs['profile_alpha']['n'])
    n_line_ave = np.mean(ns)

    # print('n0={}'.format(n0))
    # print('n_ave = {}'.format(n_ave))
    # print('n_line_ave = {}\n'.format(n_line_ave))
    # print('T0={}'.format(T0))
    # print('T_ave = {}'.format(T_ave))
    # print('T_line_ave = {}'.format(T_line_ave))
    V_plasma = 2 * np.pi**2 * inputs['major_radius'] * inputs['minor_radius']**2 * inputs['areal_elongation']
    
    if isinstance(inputs['impurities'], dict):
        p_radiation, z_dilutes = popcon.get_p_line_rad(ns, Ts, inputs['impurities'],
                                                    rs, inputs['major_radius'], inputs['areal_elongation'],
                                                    reaction=inputs['reaction'], f_DT=inputs['DT_ratio'])
        # print(z_dilutes)
    else:
        print('Hello World')
        p_radiation = popcon.get_p_bremmstrahlung(ns, Ts, rs, inputs['areal_elongation'],
                                                        inputs['major_radius'], reaction=inputs['reaction'],
                                                        impurities=inputs['impurities'])
        z_dilutes = None
    if not p_fusion:
        p_fusion = popcon.get_p_fusion(ns, Ts, rs, inputs['areal_elongation'],
                                    inputs['major_radius'], reaction=inputs['reaction'],
                                    impurities=inputs['impurities'],
                                        f_DT=inputs['DT_ratio'],
                                        f_He=inputs['f_He'],
                                        z_dilutes=z_dilutes)

    p_alpha = 0.2013 * p_fusion

    
    p_aux = get_P_aux(t, inputs['P_aux_0'], reduce=inputs['reduce_P_aux'])

    # p_ohmic = popcon.get_p_ohmic_neoclassical(inputs['plasma_current'], T_line_ave, inputs['inverse_aspect_ratio'],
    #                                           inputs['major_radius'], inputs['areal_elongation'])
    p_ohmic = popcon.get_p_ohmic_classical(Ts, ns, rs, inputs['plasma_current'],
                                        inputs['major_radius'], inputs['minor_radius'],
                                        inputs['areal_elongation'], reaction=inputs['reaction'],
                                        impurities=inputs['impurities'], f_He=inputs['f_He'])
    
    # p_ohmic = 0 * ureg.MW
    
    p_heating = p_alpha + p_aux + p_ohmic
    # print('n0={}, T0={}'.format(n0, T0))

    # print('p_ohmic = {}'.format(p_ohmic.to(ureg.MW)))

    # print('p_heating = {}'.format(p_heating))

    tau_E = popcon.get_energy_confinement_time(method=inputs['confinement']['scaling'],
                                            p_external=p_heating,
                                            plasma_current=inputs['plasma_current'],
                                            major_radius=inputs['major_radius'],
                                            minor_radius=inputs['minor_radius'],
                                            kappa=inputs['areal_elongation'],
                                            density=n_line_ave,
                                            magnetic_field_on_axis=inputs['magnetic_field_on_axis'],
                                            H=inputs['confinement']['H'],
                                            A=2.5)
    # p_loss = 3 * 1.11 * n_ave * T_ave * V_plasma / tau_E
    p_loss = popcon.get_p_total_loss(ns, Ts, rs, inputs['major_radius'],
                                    inputs['areal_elongation'], 
                                    energy_confinement_time=tau_E, 
                                    reaction=inputs['reaction'],
                                    impurities=inputs['impurities'],
                                    f_He=inputs['f_He'])
    # print('p_loss = {}'.format(p_loss.to(ureg.MW)))

    ignition_fraction = (p_alpha / p_loss).magnitude

    p_SOL = (p_loss - p_radiation).to(ureg.MW)

    return p_heating, p_loss, p_SOL, p_radiation, ignition_fraction, tau_E, n_line_ave


def get_f_LH_root(kr_frac, n0, t, T0, inputs, goal_f_LH=0.25):
    inputs['impurities']['Kr'][1] = kr_frac
    p_heating, p_loss, p_SOL, p_radiation, _, _, n_line_ave = get_powers(n0, t, T0, inputs)

    p_LH = popcon.get_p_LH_transition_3(n_line_ave, inputs['magnetic_field_on_axis'],
                                        inputs['major_radius'], inputs['minor_radius'],
                                        inputs['areal_elongation'])
    f_LH = p_SOL.to(ureg.MW).magnitude / p_LH.to(ureg.MW).magnitude
    # print('f_LH = {}'.format(f_LH))
    root = f_LH - goal_f_LH
    return root


def get_dWdt(n0, t, T0, inputs, fix_f_Kr=False, goal_f_LH=0.25):

    if not fix_f_Kr:
        inputs['impurities']['Kr'][1] = 0.0
        first_root = get_f_LH_root(0, n0, t, T0, inputs, goal_f_LH=goal_f_LH)
        if first_root>0:
            kr_frac = scipy.optimize.toms748(get_f_LH_root, 
                                            0,
                                            1/30,
                                            args=(n0, t, T0, inputs, goal_f_LH),
                                            xtol=1e-2)
            inputs['impurities']['Kr'][1] = kr_frac
            # print('kr_frac = {}'.format(kr_frac))
    else:
        # print('f_Kr is fixed: {}'.format(inputs['impurities']['Kr'][1]))
        hello = 1
    
    p_heating, p_loss, p_SOL, p_radiation, ignition_fraction, tau_E, n_line_ave = get_powers(n0, t, T0, inputs)




    # if p_SOL < 0:
    #     print('n0={}, T0={}'.format(n0, T0))

    #     print('p_loss = {}'.format(p_loss))
    #     print('p_radiation = {}'.format(p_radiation.to(ureg.MW)))

    dWdt = p_heating - p_loss

    dWdt = dWdt.to(ureg.MW)

    return dWdt, ignition_fraction, tau_E.to(ureg.s), p_SOL.to(ureg.MW), p_radiation.to(ureg.MW), p_loss.to(ureg.MW)


def get_dWdt_root(n0, t, T0, inputs, p_fusion=None, fix_f_Kr=False, goal_f_LH=0.25):

    if not fix_f_Kr:
        inputs['impurities']['Kr'][1] = 0.0
        first_root = get_f_LH_root(0, n0, t, T0, inputs, goal_f_LH=goal_f_LH)
        if first_root>0:
            kr_frac = scipy.optimize.toms748(get_f_LH_root, 
                                            0,
                                            1,
                                            args=(n0, t, T0, inputs, goal_f_LH),
                                            xtol=1e-2)
            inputs['impurities']['Kr'][1] = kr_frac
            # print('kr_frac = {}'.format(kr_frac))
    p_heating, p_loss, p_SOL, p_radiation, ignition_fraction, tau_E, n_line_ave = get_powers(n0, t, T0, inputs, p_fusion=p_fusion)


    dWdt = p_heating - p_loss

    dWdt = dWdt.to(ureg.MW)
    return dWdt.magnitude


if plot_dWdt_contour:
    ns2 = np.linspace(1e19, 1e21, 55)
    Ts2 = np.linspace(1, 30, 50)
    dWdts2 = np.zeros((len(Ts2), len(ns2)))
    previous_p_aux_0 = inputs['P_aux_0']
    inputs['P_aux_0'] = 20 * ureg.MW
    # inputs['P_aux_0'] = 20 * ureg.MW
    # temp = asymptote_T
    # temp = 7
    for j, n_i in enumerate(ns2):
        for i,T_i in enumerate(Ts2):
            dWdt2_ij, *_ = get_dWdt(n_i, 0, T_i, inputs, fix_f_Kr=False)
            dWdts2[i,j] = dWdt2_ij.magnitude
            # dndts_i = get_dndt(0, n_i, inputs, T_i)
            # dndts2[i,j] = dndts_i

    fig3, ax3 = plt.subplots(1,1, figsize=[10,8])
    xmesh, ymesh = np.meshgrid(Ts2, ns2)
    print(np.max(np.abs(dWdts2)))
    cntr = ax3.contourf(xmesh, ymesh, dWdts2.transpose(), 
                        cmap='seismic', 
                        vmin=-np.nanmax(np.abs(dWdts2)),
                        vmax=np.nanmax(np.abs(dWdts2)))
    ax3.set_xlabel('T0 [keV]')
    ax3.set_ylabel('n0 [$m^{-3}$]')
    cbar = plt.colorbar(cntr, label='dW/dt [MW]')
    ax3.set_title('P_aux={} MW, Ip={:.1f} MA, alpha_T={:.2f}, alpha_n={:.2f}'.format(inputs['P_aux_0'].magnitude,
                                                                inputs['plasma_current'].magnitude,
                                                                inputs['profile_alpha']['T'],
                                                                inputs['profile_alpha']['n']
                                                                ))

    fig3.savefig('dWdt_cntr_Paux={}MW_Ip={:.1f}MA_alpha_T={:.2f}_alpha_n={:.2f}.png'.format(
                                                                inputs['P_aux_0'].magnitude,
                                                                inputs['plasma_current'].magnitude,
                                                                inputs['profile_alpha']['T'],
                                                                inputs['profile_alpha']['n']
                                                                ))

plt.show()



    



    




