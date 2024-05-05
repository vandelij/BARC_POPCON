import popcon
import scipy.integrate
import scipy.optimize
import numpy as np
import pint
ureg = pint.get_application_registry()
from matplotlib import pyplot as plt
import matplotlib.patches

T_at_0 = 1
n_at_0 = 1e20 # m^(-3)

desired_power = 2200 * ureg.MW

plot_dWdt_contour = False

inputs = {}
kr_frac = 1.8e-3

inputs['reaction'] = 'DT'
inputs['impurities'] = [[36, kr_frac],
                        [2, 0.02]]
# inputs['impurities'] = None

inputs['major_radius'] = 4.8 * ureg.meter
inputs['inverse_aspect_ratio'] = 1.35/4.5
inputs['areal_elongation'] = 1.8
inputs['plasma_current'] = 17.5 * ureg.MA
inputs['magnetic_field_on_axis'] = 10.5 *ureg.tesla

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
inputs['num_T_points'] = 20
inputs['profile_alpha'] = {}
inputs['profile_alpha']['T'] = 1.75
inputs['T_edge'] = 0.1 * ureg.keV

# Electron Density Inputs
inputs['n_min'] = 0.1e20 * ureg.meter**(-3)
inputs['n_max'] = 1.5*3.14e20 * ureg.meter**(-3)
inputs['num_n_points'] = 20
inputs['profile_alpha']['n'] = 1.1
inputs['n_edge_factor'] = 0.25

inputs['P_aux_0'] = 15 * ureg.MW
starting_p_aux_0 = inputs['P_aux_0']

inputs['n0_slope'] = 1e18 * ureg.m**(-3)
inputs['n0_start'] = 1e20 


inputs['minor_radius'] = inputs['major_radius'] * inputs['inverse_aspect_ratio']

inputs['reduce_P_aux'] = False



def get_P_aux(t, P_aux_0, reduce=False):
    if reduce:
        P_aux = P_aux_0 - 0.25*ureg.MW * t
        # P_aux = P_aux_0
        if P_aux < 0:
            P_aux = 0 * ureg.MW
    else:
        P_aux = P_aux_0
    return P_aux

def get_f_DT(t, final_f_DT=1.0):
    # if t<25:
    #     f_DT = 1.0 + 0.0402 * t
    # else:
    #     f_DT = 1.0 + 0.0402 * 25
    f_DT = (1.0 - final_f_DT) * np.exp(-t/5) + final_f_DT

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
                                           impurities=inputs['impurities'])
    return p_ohmic


def get_p_fusion(n0, T0, inputs, f_DT=1.0):
    T0 *= ureg.keV
    n0 *= ureg.m**(-3)
    rs = np.linspace(0, inputs['minor_radius'], inputs['num_r_points'])
    Ts = popcon.get_parabolic_profile(T0, rs, inputs['minor_radius'],
                                      inputs['T_edge'],
                                      alpha=inputs['profile_alpha']['T'])
    ns = popcon.get_parabolic_profile(n0, rs, inputs['minor_radius'],
                                      inputs['n_edge_factor']*n0,
                                      alpha=inputs['profile_alpha']['n'])
    p_fusion = popcon.get_p_fusion(ns, Ts, rs, inputs['areal_elongation'],
                                   inputs['major_radius'], reaction=inputs['reaction'],
                                   impurities=inputs['impurities'], f_DT=f_DT)
    return p_fusion.to(ureg.MW)


def get_p_fusion_root(f_DT, n0, T0, inputs, desired_power):
    T0 *= ureg.keV
    n0 *= ureg.m**(-3)
    rs = np.linspace(0, inputs['minor_radius'], inputs['num_r_points'])
    Ts = popcon.get_parabolic_profile(T0, rs, inputs['minor_radius'],
                                      inputs['T_edge'],
                                      alpha=inputs['profile_alpha']['T'])
    ns = popcon.get_parabolic_profile(n0, rs, inputs['minor_radius'],
                                      inputs['n_edge_factor']*n0,
                                      alpha=inputs['profile_alpha']['n'])
    p_fusion = popcon.get_p_fusion(ns, Ts, rs, inputs['areal_elongation'],
                                   inputs['major_radius'], reaction=inputs['reaction'],
                                   impurities=inputs['impurities'], f_DT=f_DT)
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


def get_dWdt(n0, t, T0, inputs, f_DT=1.0):
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
    
    p_fusion = popcon.get_p_fusion(ns, Ts, rs, inputs['areal_elongation'],
                                inputs['major_radius'], reaction=inputs['reaction'],
                                impurities=inputs['impurities'], f_DT=f_DT)

    p_alpha = 0.2013 * p_fusion
    
    p_brem = popcon.get_p_bremmstrahlung(ns, Ts, rs, inputs['areal_elongation'],
                                   inputs['major_radius'], reaction=inputs['reaction'],
                                   impurities=inputs['impurities'])
    
    p_aux = get_P_aux(t, inputs['P_aux_0'], reduce=inputs['reduce_P_aux'])

    # p_ohmic = popcon.get_p_ohmic_neoclassical(inputs['plasma_current'], T_line_ave, inputs['inverse_aspect_ratio'],
    #                                           inputs['major_radius'], inputs['areal_elongation'])
    p_ohmic = popcon.get_p_ohmic_classical(Ts, ns, rs, inputs['plasma_current'],
                                           inputs['major_radius'], inputs['minor_radius'],
                                           inputs['areal_elongation'], reaction=inputs['reaction'],
                                           impurities=inputs['impurities'])
    
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
                                     impurities=inputs['impurities'])
    # print('p_loss = {}'.format(p_loss.to(ureg.MW)))

    ignition_fraction = (p_alpha / p_loss).magnitude

    p_SOL = p_loss - p_brem

    dWdt = p_heating - p_loss

    dWdt = dWdt.to(ureg.MW)

    return dWdt, ignition_fraction, tau_E.to(ureg.s), p_SOL.to(ureg.MW)

def get_dWdt_root(n0, t, T0, inputs, f_DT=1.0, p_fusion=None):
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
    
    if not p_fusion:
        p_fusion = popcon.get_p_fusion(ns, Ts, rs, inputs['areal_elongation'],
                                    inputs['major_radius'], reaction=inputs['reaction'],
                                    impurities=inputs['impurities'], f_DT=f_DT)

    p_alpha = 0.2013 * p_fusion
    
    p_brem = popcon.get_p_bremmstrahlung(ns, Ts, rs, inputs['areal_elongation'],
                                   inputs['major_radius'], reaction=inputs['reaction'],
                                   impurities=inputs['impurities'])
    
    p_aux = get_P_aux(t, inputs['P_aux_0'], reduce=inputs['reduce_P_aux'])

    # p_ohmic = popcon.get_p_ohmic_neoclassical(inputs['plasma_current'], T_line_ave, inputs['inverse_aspect_ratio'],
    #                                           inputs['major_radius'], inputs['areal_elongation'])
    p_ohmic = popcon.get_p_ohmic_classical(Ts, ns, rs, inputs['plasma_current'],
                                           inputs['major_radius'], inputs['minor_radius'],
                                           inputs['areal_elongation'], reaction=inputs['reaction'],
                                           impurities=inputs['impurities'])
    
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
                                     impurities=inputs['impurities'])
    # print('p_loss = {}'.format(p_loss.to(ureg.MW)))

    ignition_fraction = (p_alpha / p_loss).magnitude

    p_SOL = p_loss - p_brem

    dWdt = p_heating - p_loss

    dWdt = dWdt.to(ureg.MW)

    return dWdt.magnitude


def get_dndt(t, n0, inputs, T0, dTdt=0*ureg.keV/ureg.s):
    dWdt, *_ = get_dWdt(n0, t, T0, inputs)
    n_ave = popcon.get_n_ave(n0*ureg.m**(-3), inputs['n_edge_factor'], inputs['profile_alpha']['n'])
    T_ave = popcon.get_T_ave(T0*ureg.keV, inputs['T_edge'], inputs['profile_alpha']['T'])

    V_plasma = 2 * np.pi**2 * inputs['major_radius'] * inputs['minor_radius']**2 * inputs['areal_elongation']

    pressure_factor = popcon.get_total_pressure_factor(reaction=inputs['reaction'], impurities=inputs['impurities'])

    dwdt = dWdt / V_plasma
    dndt = (dwdt - 1.5 * pressure_factor * n_ave * dTdt) / (1.5 * pressure_factor * T_ave)
    dndt = dndt.to(ureg.m**(-3)/ureg.s)
    return dndt.magnitude


def get_dTdt(t, T0, inputs, n0, dndt=0*ureg.m**(-3)/ureg.s, change_DT=False, final_f_DT=1.0):
    if change_DT:
        f_DT = get_f_DT(t, final_f_DT=final_f_DT)
    else:
        f_DT = 1.0
    dWdt, *_ = get_dWdt(n0, t, T0, inputs, f_DT=f_DT)
    n_ave = popcon.get_n_ave(n0*ureg.m**(-3), inputs['n_edge_factor'], inputs['profile_alpha']['n'])
    T_ave = popcon.get_T_ave(T0*ureg.keV, inputs['T_edge'], inputs['profile_alpha']['T'])

    V_plasma = 2 * np.pi**2 * inputs['major_radius'] * inputs['minor_radius']**2 * inputs['areal_elongation']

    pressure_factor = popcon.get_total_pressure_factor(reaction=inputs['reaction'], impurities=inputs['impurities'])

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
                                t_eval=np.linspace(0, 100, 100))

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

for i,t_i in enumerate(sol_T.t):
    dTdts[i] = get_dTdt(t_i, Ts[i].magnitude, inputs, n_at_0)
    dWdt_T, ign_frac_T, _, p_SOL_T = get_dWdt(n_at_0, t_i, Ts[i].magnitude, inputs)
    dWdts_T[i] = dWdt_T.magnitude
    ignition_fracs_T[i] = ign_frac_T
    p_ohmics_T[i] = get_p_ohmic(n_at_0, Ts[i].magnitude, inputs).to(ureg.MW).magnitude
    p_fusions_T[i] = get_p_fusion(n_at_0, Ts[i].magnitude, inputs).to(ureg.MW).magnitude
    p_SOLs_T[i] = p_SOL_T.magnitude
    p_LHs_T[i] = get_p_LH(n_at_0, inputs).magnitude

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
T_change = 17 #keV
T_change_ind = np.argmin(np.abs(Ts - T_change*ureg.keV))
inputs['reduce_P_aux'] = True

sol_n = scipy.integrate.solve_ivp(get_dndt, 
                                [0, 70], 
                                np.array([n_at_0]),
                                args=(inputs, T_change),
                                method='RK45',
                                t_eval=np.linspace(0, 70, 70 * 4))

ns = sol_n.y.squeeze() * ureg.m**(-3)
print(sol_n)

fig_n, ax_n= plt.subplots(nrows=2, ncols=2, figsize=[10, 6])
ax_n[1,0].plot(sol_n.t, ns)
ax_n[1,0].set_xlabel('Time [s]')
ax_n[1,0].set_ylabel('n0 [$m^{-3}$]')
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

for i,t_i in enumerate(sol_n.t):
    dndts[i] = get_dndt(t_i, ns[i].magnitude, inputs, T_change)
    dWdt_n, ign_frac_n, tau_E, p_SOL_n = get_dWdt(ns[i].magnitude, t_i, T_change, inputs)
    dWdts_n[i] = dWdt_n.magnitude
    ignition_fracs_n[i] = ign_frac_n
    pumping_rates[i] = (ns[i] / tau_E).to(ureg.m**(-3)/ureg.s).magnitude
    p_ohmics_n[i] = get_p_ohmic(ns[i].magnitude, T_change, inputs).to(ureg.MW).magnitude
    p_fusions_n[i] = get_p_fusion(ns[i].magnitude, T_change, inputs).to(ureg.MW).magnitude
    p_SOLs_n[i] = p_SOL_n.magnitude
    p_LHs_n[i] = get_p_LH(ns[i].magnitude, inputs).magnitude

ax_n[0,0].plot(sol_n.t, dWdts_n, label='$dW/dt$')
ax_n[0,0].plot(sol_n.t, p_ohmics_n, label='$P_{ohmic}$')
ax_n[0,0].set_xlabel('Time [s]')
ax_n[0,0].set_ylabel('dW/dt [MW]')
ax_n[0,0].grid()
ax_n[0,0].legend()
ax_n[0,0].set_title('$dW/dt$')

ax_n[0,1].plot(sol_n.t, dndts, label='$dn_0/dt$')
ax_n[0,1].plot(sol_n.t, pumping_rates, label='$n_0/tau_E$')
ax_n[0,1].set_xlabel('Time [s]')
ax_n[0,1].set_ylabel('$dn_0/dt$ $[m^{-3} s^{-1}]$')
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


#######   Leg 3: Controlling DT ratio and seeing temperature effect ########
inputs['reduce_P_aux'] = False
inputs['P_aux_0'] = 0 * ureg.MW
change_DT = True

# Find density to achieve desired power where dW/dt = 0, where fusion power is just the input desired power at the T_change temperature
final_density = scipy.optimize.toms748(get_dWdt_root, 1e20, 1e21, args=([0], T_change, inputs, 1.0, desired_power))
print('final n0 = {:.3e} m^(-3)'.format(final_density))

# Now use that density to find the f_DT fraction needed to achieve the desired power at the T_change temperature
final_f_DT = scipy.optimize.toms748(get_p_fusion_root, 1.0, 3.0, args=(final_density, T_change, inputs, desired_power))
print('final f_DT = {:.3f}'.format(final_f_DT))

# Use the density at which ignition is approximately achieved
# n_2_ind = np.argmin(np.abs(ignition_fracs_n - 1.04))

# print(n_2_ind)
# n_2 = ns[n_2_ind].magnitude
n_2_ind = np.argmin(np.abs(ns.magnitude - final_density))
print(n_2_ind)
n_2 = final_density

sol_T3 = scipy.integrate.solve_ivp(get_dTdt, 
                                [0, 500], 
                                np.array([T_change]),
                                args=(inputs,n_2, 0*ureg.m**(-3)/ureg.s, change_DT, final_f_DT),
                                method='RK45',
                                t_eval=np.linspace(0, 500, 1000))

Ts3 = sol_T3.y.squeeze() * ureg.keV
print(sol_T3)

# print(sol.y)
fig_T3, ax_T3= plt.subplots(nrows=3, ncols=2, figsize=[10, 6])
ax_T3[1,0].plot(sol_T3.t, Ts3)
ax_T3[1,0].set_xlabel('Time [s]')
ax_T3[1,0].set_ylabel('T0 [keV]')
ax_T3[1,0].grid()
ax_T3[1,0].set_title('Temperature')

dTdts3 = np.zeros(Ts3.shape)
dWdts_T3 = np.zeros(Ts3.shape)
ignition_fracs_T3 = np.zeros(Ts3.shape)
p_ohmics_T3 = np.zeros(Ts3.shape)
p_fusions_T3 = np.zeros(Ts3.shape)
p_SOLs_T3 = np.zeros(Ts3.shape)
p_LHs_T3 = np.zeros(Ts3.shape)


for i,t_i in enumerate(sol_T3.t):
    dTdts3[i] = get_dTdt(t_i, Ts3[i].magnitude, inputs, n_2, change_DT=True, final_f_DT=final_f_DT)
    dWdt_T, ign_frac_T, _, p_SOL_T= get_dWdt(n_2, t_i, Ts3[i].magnitude, inputs, f_DT=get_f_DT(t_i, final_f_DT))
    dWdts_T3[i] = dWdt_T.magnitude
    ignition_fracs_T3[i] = ign_frac_T
    p_ohmics_T3[i] = get_p_ohmic(n_2, Ts3[i].magnitude, inputs).to(ureg.MW).magnitude
    p_fusions_T3[i] = get_p_fusion(n_2, Ts3[i].magnitude, inputs, f_DT=get_f_DT(t_i, final_f_DT)).to(ureg.MW).magnitude
    p_SOLs_T3[i] = p_SOL_T.magnitude
    p_LHs_T3[i] = get_p_LH(n_2, inputs).magnitude


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

ax_T3[2,0].plot(sol_T3.t, get_f_DT(sol_T3.t, final_f_DT=final_f_DT))
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
        ax.text((right - left)*0.95 + left,
                (top - bot)*y_frac[i] + bot,
                title_i,
                ha='right',
                va='center',
                fontweight=600,
                fontsize=12,
                color=colors[i])
    return


fig_all, ax_all = plt.subplots(nrows=4, ncols=2, figsize=[10, 6])
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
                             np.ones(ns[:n_2_ind].shape),
                             get_f_DT(sol_T3.t, final_f_DT=final_f_DT)))

ax_all[0,0].plot(ts_all, Ts_all, color=line_color, linewidth=lw)
ax_all[0,0].set_ylabel('[keV]')
add_title(ax_all[0,0], 'T0', colors=[line_color])

ax_all[1,0].plot(ts_all, ns_all/(1e20), color=line_color, linewidth=lw)
ax_all[1,0].set_ylabel('[$10^{20} m^{-3}$]')
add_title(ax_all[1,0], 'n0', colors=[line_color])

ax_all[2,0].plot(ts_all, p_auxs_all, color=line_color, linewidth=lw)
ax_all[2,0].set_ylabel('[MW]')
add_title(ax_all[2,0],'RF Power', colors=[line_color])

ax_all[3,0].plot(ts_all, f_DTs_all, color=line_color, linewidth=lw)
ax_all[3,0].set_ylabel('$n_D/n_T$')
add_title(ax_all[3,0], 'D:T Ratio', colors=[line_color])

ax_all[0,1].plot(ts_all, p_fusions_all, color=line_color, linewidth=lw)
ax_all[0,1].set_ylabel('[MW]')
add_title(ax_all[0,1],'Fusion Power', colors=[line_color])

ax_all[1,1].plot(ts_all, p_SOLs_all, color=line_color, linewidth=lw)
ax_all[1,1].plot(ts_all, p_LHs_all, color='red', linewidth=lw)
ax_all[1,1].set_ylabel('[MW]')
add_title(ax_all[1,1],['P_SOL', 'P_LH'], colors=[line_color, 'red'],
          y_frac=[0.4, 0.7])

ax_all[2,1].plot(ts_all, ignition_fracs_all, color=line_color, linewidth=lw)
ax_all[2,1].set_ylabel('$P_{fusion} / P_{loss}$')
add_title(ax_all[2,1], 'Ignition Fraction', colors=[line_color])

for i,a in enumerate(ax_all.flatten()):
    if i==len(ax_all.flatten())-1:
        a.axis('off')
    else:
        a.grid()
        bottom, top = a.get_ylim()
        a.plot([t1_end]*2, [bottom, top], '--k', linewidth=1.0)
        a.plot([t2_end]*2, [bottom, top], '--k', linewidth=1.0)
        a.set_ylim(bottom, top)
        if i>=5:
            a.set_xlabel('Time [s]')
        else:
            a.get_xaxis().set_ticklabels([])


fig_all.tight_layout()
plt.subplots_adjust(hspace=0.1)
fig_all.savefig('all_legs_DT_Ip={}MA_a_n={}.png'.format(inputs['plasma_current'].magnitude, inputs['profile_alpha']['n']))




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
            dWdt2_ij, *_ = get_dWdt(n_i, 0, T_i, inputs)
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



    



    




