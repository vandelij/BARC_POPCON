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

desired_power = 2175 * ureg.MW

# Exponential decay time constant for DT ratio evolution
tau_DT = {2:10000, 3:7}


T_change = 16 #keV
final_T = 17.9

plot_dWdt_contour = False

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
inputs['plasma_current'] = 16.5 * ureg.MA
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

inputs['n0_slope'] = 1e18 * ureg.m**(-3)
inputs['n0_start'] = 1e20 


inputs['minor_radius'] = inputs['major_radius'] * inputs['inverse_aspect_ratio']

inputs['reduce_P_aux'] = False

# def get_Kr_frac(T0, n0):
#     # f_Kr = 5e-5
#     # f_Kr = 5e-5 * T0**(0.5)
#     f_Kr = 2e-4 * (T0)**(-1) * (n0/1e20 - 1.0)**(2.0)
#     return f_Kr

def get_P_aux(t, P_aux_0, reduce=False):
    if reduce:
        first_slope = (P_aux_0 - 3/4 * P_aux_0) / 22
        second_slope = 3/4 * P_aux_0 / 15
        if t < 22:
            P_aux = P_aux_0 - first_slope * t
        elif t>=22 and t<40:
            P_aux = P_aux_0 - first_slope * 22
        else:
            P_aux = P_aux_0 - first_slope * 22 - second_slope * (t - 40)
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


def get_plasma_energy(n0, T0, inputs):
    V_plasma = 2 * np.pi**2 * inputs['major_radius'] * inputs['minor_radius']**2 * inputs['areal_elongation']
    rs = np.linspace(0, inputs['minor_radius'], inputs['num_r_points'])
    ns = popcon.get_parabolic_profile(n0*ureg.m**(-3), rs, inputs['minor_radius'],
                                      inputs['n_edge_factor']*n0*ureg.m**(-3),
                                      alpha=inputs['profile_alpha']['n'])
    Ts = popcon.get_parabolic_profile(T0*ureg.keV, rs, inputs['minor_radius'],
                                      inputs['T_edge'],
                                      alpha=inputs['profile_alpha']['T'])
    W = popcon.get_vol_integral(popcon.return_func, (3 * ns * Ts), [0], rs,
                                inputs['major_radius'], inputs['areal_elongation'])
    return W.to(ureg.MJ)


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
    if T0 < 0:
        print('t=', t)
        print('T0=', T0)
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


def get_dndt(t, n0, inputs, T0, dTdt=0*ureg.keV/ureg.s, change_DT=False, final_f_DT=1.0):
    if change_DT:
        inputs['DT_ratio'] = get_f_DT(t, final_f_DT=final_f_DT, 
                                      time_constant=inputs['tau_DT'],
                                      starting_f_DT=1.0)
    else:
        inputs['DT_ratio'] = 1.0
    # print(inputs['DT_ratio'])
    dWdt, *_ = get_dWdt(n0, t, T0, inputs)
    n_ave = popcon.get_n_ave(n0*ureg.m**(-3), inputs['n_edge_factor'], inputs['profile_alpha']['n'])
    T_ave = popcon.get_T_ave(T0*ureg.keV, inputs['T_edge'], inputs['profile_alpha']['T'])

    V_plasma = 2 * np.pi**2 * inputs['major_radius'] * inputs['minor_radius']**2 * inputs['areal_elongation']

    pressure_factor = popcon.get_total_pressure_factor(reaction=inputs['reaction'], impurities=inputs['impurities'],
                                                       f_He=inputs['f_He'])

    dwdt = dWdt / V_plasma
    dndt = (dwdt - 1.5 * pressure_factor * n_ave * dTdt) / (1.5 * pressure_factor * T_ave)
    dndt = dndt.to(ureg.m**(-3)/ureg.s)
    return dndt.magnitude


def get_dTdt(t, T0, inputs, n0, dndt=0*ureg.m**(-3)/ureg.s, change_DT=False, final_f_DT=1.0, fix_f_Kr=False, t_start=0):
    if change_DT:
        inputs['DT_ratio'] = get_f_DT(t + t_start, final_f_DT=final_f_DT,
                                      time_constant=inputs['tau_DT'],
                                      starting_f_DT=inputs['starting_f_DT'])
    else:
        inputs['DT_ratio'] = 1.0
    dWdt, *_ = get_dWdt(n0, t, T0, inputs, fix_f_Kr=fix_f_Kr)
    n_ave = popcon.get_n_ave(n0*ureg.m**(-3), inputs['n_edge_factor'], inputs['profile_alpha']['n'])
    T_ave = popcon.get_T_ave(T0*ureg.keV, inputs['T_edge'], inputs['profile_alpha']['T'])

    V_plasma = 2 * np.pi**2 * inputs['major_radius'] * inputs['minor_radius']**2 * inputs['areal_elongation']

    pressure_factor = popcon.get_total_pressure_factor(reaction=inputs['reaction'], impurities=inputs['impurities'],
                                                       f_He=inputs['f_He'])

    dwdt = dWdt / V_plasma
    dTdt = (dwdt - 1.5 * pressure_factor * T_ave * dndt) / (1.5 * pressure_factor * n_ave)
    dTdt = dTdt.to(ureg.keV / ureg.s)
    return dTdt.magnitude


# Leg 1: Temperature ramp up
inputs['reduce_P_aux'] = False

sol_T = scipy.integrate.solve_ivp(get_dTdt, 
                                [0, 100], 
                                np.array([T_at_0]),
                                args=(inputs,n_at_0),
                                method='RK45',
                                t_eval=np.linspace(0, 100, 200))

Ts = sol_T.y.squeeze() * ureg.keV
print(sol_T)

# print(sol.y)
fig_T, ax_T= plt.subplots(nrows=2, ncols=2, figsize=[10, 6])
ax_T[1,0].plot(sol_T.t, Ts)
ax_T[1,0].set_xlabel('Time [s]')
ax_T[1,0].set_ylabel('T0 [keV]')
ax_T[1,0].grid()
ax_T[1,0].set_title('Temperature')

dTdts = np.zeros(Ts.shape)
dWdts_T = np.zeros(Ts.shape)
ignition_fracs_T = np.zeros(Ts.shape)
p_ohmics_T = np.zeros(Ts.shape)
p_fusions_T = np.zeros(Ts.shape)
p_SOLs_T = np.zeros(Ts.shape)
p_LHs_T = np.zeros(Ts.shape)
p_rads_T = np.zeros(Ts.shape)
p_losses_T = np.zeros(Ts.shape)
f_Krs_T = np.zeros(Ts.shape)

for i,t_i in enumerate(sol_T.t):
    dTdts[i] = get_dTdt(t_i, Ts[i].magnitude, inputs, n_at_0)
    dWdt_T, ign_frac_T, _, p_SOL_T, p_rad_T, p_loss_T = get_dWdt(n_at_0, t_i, Ts[i].magnitude, inputs)
    dWdts_T[i] = dWdt_T.magnitude
    ignition_fracs_T[i] = ign_frac_T
    f_Krs_T[i] = inputs['impurities']['Kr'][1]
    p_ohmics_T[i] = get_p_ohmic(n_at_0, Ts[i].magnitude, inputs).to(ureg.MW).magnitude
    p_fusions_T[i] = get_p_fusion(n_at_0, Ts[i].magnitude, inputs).to(ureg.MW).magnitude
    p_SOLs_T[i] = p_SOL_T.magnitude
    p_LHs_T[i] = get_p_LH(n_at_0, inputs).magnitude
    p_rads_T[i] = p_rad_T.magnitude
    p_losses_T[i] = p_loss_T.magnitude

ax_T[0,0].plot(sol_T.t, dWdts_T, label='$dW/dt$')
ax_T[0,0].plot(sol_T.t, p_ohmics_T, label='$P_{ohmic}$')
ax_T[0,0].set_xlabel('Time [s]')
ax_T[0,0].set_ylabel('dW/dt [MW]')
ax_T[0,0].grid()
ax_T[0,0].legend()
ax_T[0,0].set_title('$dW/dt$')

ax_T[0,1].plot(sol_T.t, dTdts)
ax_T[0,1].set_xlabel('Time [s]')
ax_T[0,1].set_ylabel('dT/dt [keV/s]')
ax_T[0,1].grid()
ax_T[0,1].set_title('$dT_0/dt$')

ax_T[1,1].plot(sol_T.t, ignition_fracs_T)
ax_T[1,1].set_xlabel('Time [s]')
ax_T[1,1].set_ylabel('Ignition Fraction')
ax_T[1,1].grid()
ax_T[1,1].set_title('$P_{fusion} / P_{loss}$')

fig_T.suptitle('Leg 1: n0={:.3e} $1/m^3$, P_aux={} MW'.format(n_at_0, starting_p_aux_0.magnitude))
fig_T.tight_layout()
fig_T.savefig('leg_1_DT_Ip={}MA_a_n={}.png'.format(inputs['plasma_current'].magnitude, inputs['profile_alpha']['n']))

############ Leg 2: Density ramp up #######################
T_change_ind = np.argmin(np.abs(Ts - T_change*ureg.keV))

previous_p_aux_0 = inputs['P_aux_0']

fix_f_Kr = False

inputs['impurities']['Kr'][1] = final_Kr_frac

inputs['reduce_P_aux'] = False
inputs['P_aux_0'] = 0 * ureg.MW

# Find density to achieve desired power where dW/dt = 0, where fusion power is just the input desired power at the T_change temperature
final_density = scipy.optimize.toms748(get_dWdt_root, 1e20, 1e21, args=([0], final_T, inputs, desired_power, fix_f_Kr))
print('final n0 = {:.3e} m^(-3)'.format(final_density))

# Now use that density to find the f_DT fraction needed to achieve the desired power at the T_change temperature
final_f_DT = scipy.optimize.toms748(get_p_fusion_root, 1.0, 5.0, args=(final_density, final_T, inputs, desired_power))
print('final f_DT = {:.3f}'.format(final_f_DT))

inputs['reduce_P_aux'] = True
inputs['P_aux_0'] = previous_p_aux_0

change_DT_n = True
inputs['tau_DT'] = tau_DT[2]
fix_f_Kr = False
sol_n = scipy.integrate.solve_ivp(get_dndt, 
                                [0, 70], 
                                np.array([n_at_0]),
                                args=(inputs, T_change, 0 * ureg.keV/ureg.s, change_DT_n, final_f_DT),
                                method='RK45',
                                t_eval=np.linspace(0, 70, 70 * 4))

ns = sol_n.y.squeeze() * ureg.m**(-3)
print(sol_n)

fig_n, ax_n= plt.subplots(nrows=2, ncols=2, figsize=[10, 6])
ax_n[1,0].plot(sol_n.t, ns)
ax_n[1,0].set_xlabel('Time [s]')
ax_n[1,0].set_ylabel('n0 [$m^{-3}$]')
ax_n[1,0].set_ylim(0, 1e21)
ax_n[1,0].grid()
ax_n[1,0].set_title('Density')

dndts = np.zeros(ns.shape)
dWdts_n = np.zeros(ns.shape)
pumping_rates = np.zeros(ns.shape)
ignition_fracs_n = np.zeros(ns.shape)
p_ohmics_n = np.zeros(ns.shape)
p_fusions_n = np.zeros(ns.shape)
p_SOLs_n =- np.zeros(ns.shape)
p_LHs_n = np.zeros(ns.shape)
p_rads_n = np.zeros(ns.shape)
p_losses_n = np.zeros(ns.shape)
f_Krs_n = np.zeros(ns.shape)
f_DTs_n = np.zeros(ns.shape)
p_auxs_n = np.zeros(ns.shape)


for i,t_i in enumerate(sol_n.t):
    dndts[i] = get_dndt(t_i, ns[i].magnitude, inputs, T_change, change_DT=change_DT_n, final_f_DT=final_f_DT)
    dWdt_n, ign_frac_n, tau_E, p_SOL_n, p_rad_n, p_loss_n = get_dWdt(ns[i].magnitude, t_i, T_change, inputs)
    dWdts_n[i] = dWdt_n.magnitude
    ignition_fracs_n[i] = ign_frac_n
    pumping_rates[i] = (ns[i] / tau_E).to(ureg.m**(-3)/ureg.s).magnitude
    f_Krs_n[i] = inputs['impurities']['Kr'][1]
    p_ohmics_n[i] = get_p_ohmic(ns[i].magnitude, T_change, inputs).to(ureg.MW).magnitude
    p_fusions_n[i] = get_p_fusion(ns[i].magnitude, T_change, inputs).to(ureg.MW).magnitude
    p_SOLs_n[i] = p_SOL_n.magnitude
    p_LHs_n[i] = get_p_LH(ns[i].magnitude, inputs).magnitude
    p_rads_n[i] = p_rad_n.magnitude
    p_losses_n[i] = p_loss_n.magnitude
    f_DTs_n[i] = inputs['DT_ratio']
    p_auxs_n[i] = get_P_aux(t_i, starting_p_aux_0, inputs['reduce_P_aux']).magnitude

ax_n[0,0].plot(sol_n.t, dWdts_n, label='$dW/dt$')
ax_n[0,0].plot(sol_n.t, p_ohmics_n, label='$P_{ohmic}$')
ax_n[0,0].plot(sol_n.t, p_auxs_n, label='$P_{aux}$')
ax_n[0,0].set_xlabel('Time [s]')
ax_n[0,0].set_ylabel('dW/dt [MW]')
ax_n[0,0].set_ylim(-20, 100)
ax_n[0,0].grid()
ax_n[0,0].legend()
ax_n[0,0].set_title('$dW/dt$')

ax_n[0,1].plot(sol_n.t, dndts, label='$dn_0/dt$')
ax_n[0,1].plot(sol_n.t, pumping_rates, label='$n_0/tau_E$')
ax_n[0,1].set_xlabel('Time [s]')
ax_n[0,1].set_ylabel('$dn_0/dt$ $[m^{-3} s^{-1}]$')
ax_n[0,1].set_ylim(-1e20, 1e21)
ax_n[0,1].legend()
ax_n[0,1].grid()
ax_n[0,1].set_title('$dn_0/dt$')

ax_n[1,1].plot(sol_n.t, ignition_fracs_n)
ax_n[1,1].set_xlabel('Time [s]')
ax_n[1,1].set_ylabel('Ignition Fraction')
ax_n[1,1].grid()
ax_n[1,1].set_title('$P_{fusion} / P_{loss}$')

fig_n.suptitle('Leg 2: T0={} keV'.format(T_change))
fig_n.tight_layout()
fig_n.savefig('leg_2_DT_Ip={}MA_a_n={}.png'.format(inputs['plasma_current'].magnitude, inputs['profile_alpha']['n']))

print('.........End of Leg 2............')

#######   Leg 3: Controlling DT ratio and seeing temperature effect ########
inputs['reduce_P_aux'] = False
inputs['P_aux_0'] = 0 * ureg.MW
change_DT = True
fix_f_Kr = False


# test_ns = np.linspace(1e20, 1e21)
# test_dWdts = np.zeros(test_ns.shape)
# for i,n in enumerate(test_ns):
#     test_dWdts[i] = get_dWdt_root(n, [0], T_change, inputs, p_fusion=desired_power, fix_f_Kr=fix_f_Kr)
# fig, ax = plt.subplots()
# ax.plot(test_ns, test_dWdts)
# plt.show()


# Use the density at which ignition is approximately achieved
# n_2_ind = np.argmin(np.abs(ignition_fracs_n - 1.04))

# print(n_2_ind)
# n_2 = ns[n_2_ind].magnitude
n_2_ind = np.argmin(np.abs(ns.magnitude - final_density))
print(n_2_ind)
n_2 = final_density

inputs['tau_DT'] = tau_DT[3]
print('tau_DT = {}'.format(inputs['tau_DT']))
inputs['starting_f_DT'] = f_DTs_n[n_2_ind]

fix_f_Kr = False
inputs['impurities']['Kr'][1] = final_Kr_frac
# inputs['impurities']['Kr'][1] = f_Krs_n[n_2_ind]

print('final f_Kr = {}'.format(inputs['impurities']['Kr'][1]))

rs = np.linspace(0, inputs['minor_radius'], 1000)
n_2s = popcon.get_parabolic_profile(n_2 * ureg.m**(-3), rs, inputs['minor_radius'], 0.4*n_2*ureg.m**(-3), alpha=inputs['profile_alpha']['n'])

print(np.mean(n_2s))
greenwald_limit = popcon.get_greenwald_density(inputs['plasma_current'], inputs['minor_radius'])
print('greenwald limit: {}'.format(greenwald_limit))



sol_T3 = scipy.integrate.solve_ivp(get_dTdt, 
                                [0, 85], 
                                np.array([T_change]),
                                args=(inputs,n_2, 0*ureg.m**(-3)/ureg.s, change_DT, final_f_DT, fix_f_Kr),
                                method='RK45',
                                t_eval=np.linspace(0, 85, 170))

Ts3 = sol_T3.y.squeeze() * ureg.keV
print(sol_T3)
print('-------------------------------------------')

print('Final Temperature: ', Ts3[-1])
# print(sol.y)
fig_T3, ax_T3= plt.subplots(nrows=3, ncols=2, figsize=[10, 6])
ax_T3[1,0].plot(sol_T3.t, Ts3)
ax_T3[1,0].set_xlabel('Time [s]')
ax_T3[1,0].set_ylabel('T0 [keV]')
ax_T3[1,0].grid()
ax_T3[1,0].set_title('Temperature')
# plt.show()

dTdts3 = np.zeros(Ts3.shape)
dWdts_T3 = np.zeros(Ts3.shape)
ignition_fracs_T3 = np.zeros(Ts3.shape)
p_ohmics_T3 = np.zeros(Ts3.shape)
p_fusions_T3 = np.zeros(Ts3.shape)
p_SOLs_T3 = np.zeros(Ts3.shape)
p_LHs_T3 = np.zeros(Ts3.shape)
p_rads_T3 = np.zeros(Ts3.shape)
p_losses_T3 = np.zeros(Ts3.shape)
f_Krs_T3 = np.zeros(Ts3.shape)
f_DTs_T3 = np.zeros(Ts3.shape)

# time.sleep(0.5)
# print(inputs['impurities'])
for i,t_i in enumerate(sol_T3.t):
    # print(inputs['impurities'])
    dTdts3[i] = get_dTdt(t_i, Ts3[i].magnitude, inputs, n_2, change_DT=True, final_f_DT=final_f_DT, fix_f_Kr=fix_f_Kr)
    # print(inputs['impurities'])
    dWdt_T, ign_frac_T, _, p_SOL_T, p_rad_T, p_loss_T = get_dWdt(n_2, t_i, Ts3[i].magnitude, inputs, fix_f_Kr=fix_f_Kr)
    # print(inputs['impurities'])
    # time.sleep(0.5)
    dWdts_T3[i] = dWdt_T.magnitude
    ignition_fracs_T3[i] = ign_frac_T
    f_Krs_T3[i] = inputs['impurities']['Kr'][1]
    # print(inputs['impurities'])
    p_ohmics_T3[i] = get_p_ohmic(n_2, Ts3[i].magnitude, inputs).to(ureg.MW).magnitude
    # print(inputs['impurities'])
    p_fusions_T3[i] = get_p_fusion(n_2, Ts3[i].magnitude, inputs).to(ureg.MW).magnitude
    # print(inputs['impurities'])
    p_SOLs_T3[i] = p_SOL_T.magnitude
    p_LHs_T3[i] = get_p_LH(n_2, inputs).magnitude
    # print(inputs['impurities'])
    p_rads_T3[i] = p_rad_T.magnitude
    p_losses_T3[i] = p_loss_T.magnitude
    f_DTs_T3[i] = inputs['DT_ratio']
    # print('\n')


ax_T3[0,0].plot(sol_T3.t, dWdts_T3, label='$dW/dt$')
ax_T3[0,0].plot(sol_T3.t, p_ohmics_T3, label='$P_{ohmic}$')
ax_T3[0,0].set_xlabel('Time [s]')
ax_T3[0,0].set_ylabel('dW/dt [MW]')
ax_T3[0,0].grid()
ax_T3[0,0].legend()
ax_T3[0,0].set_title('$dW/dt$')

ax_T3[0,1].plot(sol_T3.t, dTdts3)
ax_T3[0,1].set_xlabel('Time [s]')
ax_T3[0,1].set_ylabel('dT/dt [keV/s]')
ax_T3[0,1].grid()
ax_T3[0,1].set_title('$dT_0/dt$')

ax_T3[1,1].plot(sol_T3.t, ignition_fracs_T3)
ax_T3[1,1].set_xlabel('Time [s]')
ax_T3[1,1].set_ylabel('Ignition Fraction')
ax_T3[1,1].grid()
ax_T3[1,1].set_title('$P_{fusion} / P_{loss}$')

ax_T3[2,0].plot(sol_T3.t, f_DTs_T3)
ax_T3[2,0].set_xlabel('Time [s]')
ax_T3[2,0].set_ylabel('$n_D/n_T$')
ax_T3[2,0].grid()
ax_T3[2,0].set_title('$f_{DT}$')


fig_T3.suptitle('Leg 3, n0={:.3e} $1/m^3$, Final P_fusion={:.0f} MW'.format(n_2, p_fusions_T3[-1]))
fig_T3.tight_layout()
fig_T3.savefig('leg_3_DT_Ip={}MA_a_n={}.png'.format(inputs['plasma_current'].magnitude, inputs['profile_alpha']['n']))



################ Plot All Legs Together ##########################

line_color = 'blue'
lw = 1.5

def add_title(ax, title, colors=['black'], y_frac=[0.6, 0.4, 0.1]):
    if not isinstance(title, list):
        title = [title]
    for i,title_i in enumerate(title):
        bot, top = ax.get_ylim()
        left, right = ax.get_xlim()
        right = 150
        ax.text((right - left)*0.95 + left,
                (top - bot)*y_frac[i] + bot,
                title_i,
                ha='right',
                va='center',
                fontweight=600,
                fontsize=12,
                color=colors[i])
    return

t1_end = sol_T.t[T_change_ind]
t2_end = sol_n.t[n_2_ind] + t1_end
ts_all = np.concatenate((sol_T.t[:T_change_ind], 
                         sol_n.t[:n_2_ind]+t1_end, 
                         sol_T3.t + t2_end))
Ts_all = np.concatenate((Ts[:T_change_ind], 
                         T_change*ureg.keV*np.ones(ns[:n_2_ind].shape), 
                         Ts3))
ns_all = np.concatenate((n_at_0*ureg.m**(-3)*np.ones(Ts[:T_change_ind].shape), 
                         ns[:n_2_ind], 
                         n_2*ureg.m**(-3)*np.ones(Ts3.shape)))

p_auxs_leg_2 = np.zeros(sol_n.t[:n_2_ind].shape)
for i, t_i in enumerate(sol_n.t[:n_2_ind]):
    p_auxs_leg_2[i] = get_P_aux(t_i, starting_p_aux_0, reduce=True).magnitude

p_auxs_all = np.concatenate((starting_p_aux_0.magnitude * np.ones(Ts[:T_change_ind].shape),
                             p_auxs_leg_2,
                             np.zeros(Ts3.shape)))
ignition_fracs_all = np.concatenate((ignition_fracs_T[:T_change_ind], 
                                     ignition_fracs_n[:n_2_ind],
                                     ignition_fracs_T3))
p_fusions_all = np.concatenate((p_fusions_T[:T_change_ind],
                                p_fusions_n[:n_2_ind],
                                p_fusions_T3))
p_SOLs_all = np.concatenate((p_SOLs_T[:T_change_ind],
                             p_SOLs_n[:n_2_ind],
                             p_SOLs_T3))
p_LHs_all = np.concatenate((p_LHs_T[:T_change_ind],
                             p_LHs_n[:n_2_ind],
                             p_LHs_T3))
f_DTs_all = np.concatenate((np.ones(Ts[:T_change_ind].shape),
                             f_DTs_n[:n_2_ind],
                             f_DTs_T3))
p_rads_all = np.concatenate((p_rads_T[:T_change_ind],
                             p_rads_n[:n_2_ind],
                             p_rads_T3))
p_losses_all = np.concatenate((p_losses_T[:T_change_ind],
                             p_losses_n[:n_2_ind],
                             p_losses_T3))
f_Krs_all = np.concatenate((f_Krs_T[:T_change_ind],
                            f_Krs_n[:n_2_ind],
                            f_Krs_T3))

q_star_eff = popcon.get_q_star_eff(inputs['major_radius'], inputs['minor_radius'],
                                   inputs['areal_elongation'], inputs['plasma_current'],
                                   inputs['magnetic_field_on_axis'])

n_GRs_all = popcon.get_giacomin_density_limit(inputs['inverse_aspect_ratio'],
                                              inputs['minor_radius'],
                                              inputs['major_radius'],
                                              p_SOLs_all,
                                              q_star_eff,
                                              inputs['areal_elongation'],
                                              inputs['magnetic_field_on_axis'])

n_edges_all = ns_all * inputs['n_edge_factor']

peak_greenwald = popcon.get_peak_greenwald(inputs['plasma_current'], inputs['minor_radius'], inputs['n_edge_factor'], inputs['profile_alpha']['n'])


time_inds = [T_change_ind, n_2_ind + T_change_ind, len(ts_all)-1]
leg_names = ['leg_1', 'leg_2', 'leg_3']

for j,ind in enumerate(time_inds):
    fig_all, ax_all = plt.subplots(nrows=4, ncols=2, figsize=[14, 6])

    ax_all[0,0].plot(ts_all[:ind], Ts_all[:ind], color=line_color, linewidth=lw)
    ax_all[0,0].set_ylabel('[keV]')
    ax_all[0,0].set_ylim(0, Ts_all.max().magnitude*1.05)
    add_title(ax_all[0,0], 'T_0', colors=[line_color])

    ax_all[1,0].plot(ts_all[:ind], ns_all[:ind]/(1e20), color=line_color, linewidth=lw)
    ax_all[1,0].plot(ts_all[:ind], peak_greenwald/(1e20) * np.ones(ns_all[:ind].shape), color='red', linewidth=lw)
    ax_all[1,0].set_ylabel('[$10^{20} m^{-3}$]')
    ax_all[1,0].set_ylim(0, np.max([ns_all.max().magnitude, peak_greenwald.magnitude]) / (1e20) *1.05)
    add_title(ax_all[1,0], ['n_e0', 'Greenwald Limit'], colors=[line_color, 'red'], y_frac=[0.7, 0.4])

    ax_all[2,0].plot(ts_all[:ind], f_Krs_all[:ind], color=line_color, linewidth=lw)
    ax_all[2,0].set_ylabel('$n_{Kr}/n_e$')
    ax_all[2,0].set_ylim(0, f_Krs_all.max()*1.05)
    add_title(ax_all[2,0], 'Krypton Fraction', colors=[line_color])

    ax_all[3,0].plot(ts_all[:ind], f_DTs_all[:ind], color=line_color, linewidth=lw)
    ax_all[3,0].set_ylabel('$n_D/n_T$')
    ax_all[3,0].set_ylim(0.95, f_DTs_all.max()*1.05)
    add_title(ax_all[3,0], 'D:T Ratio', colors=[line_color])

    ax_all[0,1].plot(ts_all[:ind], p_fusions_all[:ind], color=line_color, linewidth=lw)
    ax_all[0,1].set_ylabel('[MW]')
    ax_all[0,1].set_ylim(0, p_fusions_all.max()*1.05)
    add_title(ax_all[0,1],'Fusion Power', colors=[line_color])

    ax_all[1,1].plot(ts_all[:ind], p_auxs_all[:ind], color=line_color, linewidth=lw)
    ax_all[1,1].set_ylabel('[MW]')
    ax_all[1,1].set_ylim(0, p_auxs_all.max()*1.05)
    add_title(ax_all[1,1],'RF Power', colors=[line_color])

    ax_all[2,1].plot(ts_all[:ind], p_SOLs_all[:ind], color=line_color, linewidth=lw)
    ax_all[2,1].plot(ts_all[:ind], p_LHs_all[:ind], color='red', linewidth=lw)
    # ax_all[1,1].plot(ts_all, p_rads_all, color='green', linewidth=lw)
    # ax_all[1,1].plot(ts_all, p_losses_all, color='purple', linewidth=lw)
    ax_all[2,1].set_ylabel('[MW]')
    ax_all[2,1].set_ylim(0, np.max([p_LHs_all.max(), p_SOLs_all.max()])*1.05)
    add_title(ax_all[2,1],['P_SOL', 'LH Threshold'], colors=[line_color, 'red'],
            y_frac=[0.4, 0.7])

    ax_all[3,1].plot(ts_all[:ind], n_edges_all[:ind]/(1e20), color=line_color, linewidth=lw)
    ax_all[3,1].plot(ts_all[:ind], n_GRs_all[:ind]/(1e20), color='red', linewidth=lw)
    ax_all[3,1].set_ylabel('[$10^{20} m^{-3}$]')
    ax_all[3,1].set_ylim(0, np.max([n_edges_all.max().magnitude, n_GRs_all.max().magnitude]) / (1e20) * 1.05)
    add_title(ax_all[3,1], ['Separatrix Density', 'Giacomin-Ricci Limit'], colors=[line_color, 'red'],
              y_frac=[0.4, 0.7])

    for i,a in enumerate(ax_all.flatten()):
        a.grid()
        bottom, top = a.get_ylim()
        a.plot([t1_end]*2, [bottom, top], '--k', linewidth=1.5)
        a.plot([t2_end]*2, [bottom, top], '--k', linewidth=1.5)
        a.set_ylim(bottom, top)
        a.set_xlim(ts_all[0] - 2, ts_all[-1])
        if i>=6:
            a.set_xlabel('Time [s]')
        else:
            a.get_xaxis().set_ticklabels([])


    # fig_all.suptitle('$f_{He}= $' + '{:.2e}, '.format(inputs['f_He']) \
    #                 + 'Final $f_{Kr}= $' + '{:.2e}'.format(f_Krs_all[-1]))

    fig_all.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    fig_all.savefig('all_params_leg_{}_DT_Ip={}MA_a_n={}.png'.format(j, inputs['plasma_current'].magnitude, inputs['profile_alpha']['n']))

first_wall_area = (2 * np.pi * inputs['major_radius']) * (2 * np.pi * (inputs['minor_radius'] + 0.01 * ureg.m)) * np.sqrt((1 + inputs['areal_elongation']**2) / 2)


## Get average values:
total_time = ts_all[-1] - ts_all[0]
ave_P_aux = scipy.integrate.trapezoid(p_auxs_all, ts_all) / total_time
ave_P_fus = scipy.integrate.trapezoid(p_fusions_all, ts_all) / total_time
print('\n')
print('Total Time ......................................... {:.1f}'.format(total_time))
print('Final Temperature: ................................. {:.1f}'.format(Ts3[-1]))
print('Final Peak Density: ................................ {:.3e}'.format(ns_all[-1]))
print('Final Separatrix Density: .......................... {:.3e}'.format(ns_all[-1] * inputs['n_edge_factor']))
print('Final f_Kr ......................................... {:.3e}'.format(f_Krs_all[-1]))
print('Final f_DT ......................................... {:.3e}'.format(f_DTs_all[-1]))
print('Final Fusion Power: ................................ {:.1f}'.format( p_fusions_T3[-1]))
print('Final P_SOL ........................................ {:.1f}'.format(p_SOLs_all[-1]))
print('Average Auxillary Power ............................ {:.3f}'.format(ave_P_aux))
print('Average Fusion Power ............................... {:.3f}'.format(ave_P_fus))
print('Ave Auxillary Power over 1000 s .................... {:.3f}'.format((ave_P_aux * total_time + 0.0 * (1000 - total_time))/1000))
print('Ave Fusion Power over 1000 s ....................... {:.3f}'.format((ave_P_fus * total_time + p_fusions_all[-1] * (1000 - total_time))/1000))
print('q* ................................................. {:.3f}'.format(q_star_eff))
print('\n')

print('P_neutron: {} MW'.format(p_fusions_all[-1]*4/5))
print('P_neutron / S: {} MW/m^2'.format(p_fusions_all[-1]*4/5 / first_wall_area))
print('P_rad / S: {} MW/m^2'.format(p_rads_all[-1] / first_wall_area))
print('P_SOL: {} MW'.format(p_SOLs_all[-1]))
print('n_sep: {} m^-3'.format(ns_all[-1]*inputs['n_edge_factor']))
print('P_SOL B_T / R: {}'.format(p_SOLs_all[-1] * ureg.MW * inputs['magnetic_field_on_axis'] / inputs['major_radius']))
print('P_SOL B_T / R / n^2: {}'.format((p_SOLs_all[-1] * ureg.MW * inputs['magnetic_field_on_axis'] / inputs['major_radius'] / (ns_all[-1] * inputs['n_edge_factor'])**2).to_reduced_units()))

fig_p, ax_p = plt.subplots(1,1, figsize=[3,4], dpi=150)
ns_start = popcon.get_parabolic_profile(ns_all[0], rs, inputs['minor_radius'], ns_all[0]*inputs['n_edge_factor'], alpha=inputs['profile_alpha']['n'])
ns_end = popcon.get_parabolic_profile(ns_all[-1], rs, inputs['minor_radius'], ns_all[-1]*inputs['n_edge_factor'], alpha=inputs['profile_alpha']['n'])
ax_p.plot(rs/inputs['minor_radius'], ns_start, color='blue', label='$t=0$')
ax_p.plot(rs/inputs['minor_radius'], ns_end, color='red', label='Flat-top')
ax_p.set_xlabel('$r/a$', fontsize=14)
ax_p.set_ylabel('$n_e$ [m${}^{-3}$]', fontsize=14)
ax_p.legend()
fig_p.tight_layout()
fig_p.savefig('density_profile.png')

fig_p2, ax_p2 = plt.subplots(1,1, figsize=[3,4], dpi=150)
Ts_start = popcon.get_parabolic_profile(Ts_all[0], rs, inputs['minor_radius'], inputs['T_edge'], alpha=inputs['profile_alpha']['T'])
Ts_end = popcon.get_parabolic_profile(Ts_all[-1], rs, inputs['minor_radius'], inputs['T_edge'], alpha=inputs['profile_alpha']['T'])
ax_p2.plot(rs/inputs['minor_radius'], Ts_start, color='blue', label='$t=0$')
ax_p2.plot(rs/inputs['minor_radius'], Ts_end, color='red', label='Flat-top')
ax_p2.set_xlabel('$r/a$', fontsize=14)
ax_p2.set_ylabel('$T$ [keV]', fontsize=14)
ax_p2.legend()
fig_p2.tight_layout()
fig_p2.savefig('temperature_profile.png')


########## Plot Final Popcon #################
######## Plot Inputs ##########
plot_inputs = {}
plot_inputs['contours'] = {}
plot_inputs['contours']['P_fusion'] = {'levels': [500, 1000, 2000, 4000], 'colors':'black'}
# plot_inputs['contours']['P_ohmic'] = {'levels': [1, 10, 100, 1000, 1e4]}
plot_inputs['contours']['P_auxillary'] = {'levels': [0, 10, 20, 40, 60, 80, 100], 'colors':'red'}
# plot_inputs['contours']['P_SOL'] = {'levels': [10, 100, 1000, 1e4]}
# plot_inputs['contours']['P_radiation'] = {'levels': [10, 100, 1000, 1e4]}
# plot_inputs['contours']['energy_confinement_time'] = {'levels':[0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 10.0]}
# plot_inputs['contours']['Q'] = {'levels': [1, 10, 20, 40, 1000], 'colors':'tab:purple'}
# plot_inputs['contours']['peak_greenwald_fraction'] = {'levels': [1.0], 'colors':'green'}
# plot_inputs['contours']['sepOS_density_fraction'] = {'levels': [0.1, 1.0, 5.0, 10.0, 20.0], 'colors':'blue'}
# plot_inputs['contours']['bernert_density_fraction'] = {'levels': [0.1, 1.0, 5.0, 10.0, 20.0], 'colors':'grey'}
# plot_inputs['contours']['P_LH_fraction'] = {'levels': [1.0], 'colors':'gold'}

plot_inputs['plot_ignition'] = True

time_inds = [T_change_ind, n_2_ind + T_change_ind, np.argmin(np.abs(120 - ts_all))]
for j,time_ind in enumerate(time_inds):
    if j==0:
        step_inds = np.arange(0, time_ind, 1)
    else:
        step_inds = np.arange(time_inds[j-1], time_ind, 2)
    # Make sure last time_step is always included in gif
    if step_inds[-1] < time_ind:
        step_inds = np.append(step_inds, time_ind)
    directory = 'Leg_{}'.format(j+1)
    if not os.path.isdir(directory):
        os.mkdir(directory)

    images = []
    for k,step_ind in enumerate(step_inds):
        plot_inputs['filename'] = os.path.join(directory, 'iris_t={:.1f}s.png'.format(ts_all[step_ind]))

        if not os.path.isfile(plot_inputs['filename']):

            inputs['DT_ratio'] = f_DTs_all[step_ind]
            inputs['impurities'] = {'Kr':[36, f_Krs_all[step_ind]]}

            output = popcon.get_all_parameters(inputs)
            fig, ax = popcon.plot_popcon(output, plot_inputs)

            ax.plot(Ts_all[step_ind], ns_all[step_ind], '*g', ms=20)
            fig.savefig(plot_inputs['filename'])
            plt.close(fig)

        images.append(imageio.imread(plot_inputs['filename']))

    if j==0:
        imageio.mimsave(os.path.join(directory, 'iris_leg_{}.gif'.format(j+1)), images, fps=10)
    else:
        imageio.mimsave(os.path.join(directory, 'iris_leg_{}.gif'.format(j+1)), images, fps=15)





# output = popcon.get_all_parameters(inputs)
# fig, ax = popcon.plot_popcon(output, plot_inputs)

if plot_dWdt_contour:
    ns2 = np.linspace(0.0e20, 1e21, 55)
    Ts2 = np.linspace(0, 30, 50)
    dWdts2 = np.zeros((len(Ts2), len(ns2)))
    previous_p_aux_0 = inputs['P_aux_0']
    inputs['P_aux_0'] = 0 * ureg.MW
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



    



    




