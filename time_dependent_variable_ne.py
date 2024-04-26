import popcon
import scipy.integrate
import numpy as np
import pint
ureg = pint.get_application_registry()
from matplotlib import pyplot as plt
import matplotlib.patches

T_at_0 = 1 * ureg.keV
n0 = 5e20 * ureg.m**(-3)

inputs = {}
kr_frac = 1.1e-3
inputs['reaction'] = 'DT'
inputs['impurities'] =  [[36, kr_frac]]
# inputs['impurities'] = None
inputs['fuel_ratio'] = 1.0
# inputs['impurities'] = None

inputs['major_radius'] = 4.5 * ureg.meter
inputs['inverse_aspect_ratio'] = 0.3
inputs['minor_radius'] = inputs['major_radius'] * inputs['inverse_aspect_ratio']
inputs['areal_elongation'] = 1.8
inputs['plasma_current'] = 15 * ureg.MA
inputs['magnetic_field_on_axis'] = 10.51 *ureg.tesla

inputs['confinement'] = {}
# Scaling options are 'ITER98y2', 'ITER97', or 'ITER89'
inputs['confinement']['scaling'] = 'ITER89'
inputs['confinement']['H'] = 1.3
inputs['confinement']['lower_bound'] = 1e-5
inputs['confinement']['upper_bound'] = 5000
inputs['A'] = 2.5

# Method for calculating P_SOL
inputs['P_SOL_method'] = 'total'

inputs['num_r_points'] = 50

# Electron Temperature Inputs
inputs['profile_alpha'] = {}
inputs['profile_alpha']['T'] = 1.5
inputs['T_edge'] = 0.1 * ureg.keV

# Electron Density Inputs
inputs['profile_alpha']['n'] = 1.1
inputs['n_edge_factor'] = 0.47

inputs['P_aux_0'] = 30 * ureg.MW

inputs['n0_slope'] = 1e18 * ureg.m**(-3)
inputs['n0_start'] = 1e20 * ureg.m**(-3)

inputs['density_ramp_temperature'] = 19.5 * ureg.keV

inputs['minor_radius'] = inputs['major_radius'] * inputs['inverse_aspect_ratio']

global start_ramp
start_ramp = False


def get_n0_variable(t, T0, t0, n0_start, n0_slope):
    # n0 = n0_start + n0_slope * ((t-t0)*ureg.second/tau_E)
    n0 = n0_start + n0_slope * ((t - t0) / 1)
    return n0

def get_dn0dt(t, T0, n0_slope, density_ramp_temperature):
    if T0<density_ramp_temperature:
        dn0dt = 0 * ureg.m**(-3) / ureg.s
    else:
        # dn0dt = 0 * ureg.m**(-3) / ureg.s
        dn0dt = n0_slope / tau_E
    return dn0dt

def P_aux(t, P_aux_0):
    return P_aux_0

def get_ignition_frac(rs, s_alpha, s_brem, s_cond, major_radius, areal_elongation):
    p_alpha = popcon.get_vol_integral(popcon.return_func, s_alpha, [0], rs, 
                            major_radius, areal_elongation)
    p_brem = popcon.get_vol_integral(popcon.return_func, s_brem, [0], rs, 
                            major_radius, areal_elongation)
    p_cond = popcon.get_vol_integral(popcon.return_func, s_cond, [0], rs, 
                            major_radius, areal_elongation)
    ignition_frac = p_alpha / (p_brem + p_cond)
    return ignition_frac


def get_dWdt_density(t, T0, inputs):
    global tau_E, start_ramp, t0
    print(T0)
    T0 *= ureg.keV
    print(start_ramp)
    if T0 < inputs['density_ramp_temperature'] and start_ramp==False:
        n0 = inputs['n0_start']
        dn0dt = 0 * ureg.m**(-3) / ureg.second
    elif start_ramp == False:
        start_ramp = True
        t0 = t
        n0 = get_n0_variable(t, T0, t0, inputs['n0_start'], inputs['n0_slope'])
        # dn0dt = inputs['n0_slope'] / tau_E
        dn0dt = inputs['n0_slope'] / (1 * ureg.second)
        print('t0 = {}'.format(t0))
    else:
        n0 = get_n0_variable(t, T0, t0, inputs['n0_start'], inputs['n0_slope'])
        # dn0dt = inputs['n0_slope'] / tau_E
        dn0dt = inputs['n0_slope'] / (1 * ureg.second)
        print('t0 = {}'.format(t0))
    print('n0 = {}'.format(n0))
    rs = np.linspace(0, inputs['minor_radius'])
    ns = popcon.get_parabolic_profile(n0, rs, inputs['minor_radius'],
                                      inputs['n_edge_factor']*n0,
                                      alpha=inputs['profile_alpha']['n'])
    Ts = popcon.get_parabolic_profile(T0, rs, inputs['minor_radius'],
                                      inputs['T_edge'],
                                      alpha=inputs['profile_alpha']['n'])
    
    V_plasma = 2 * np.pi**2 * inputs['major_radius'] * inputs['minor_radius']**2 * inputs['areal_elongation']
    n_ave = popcon.get_volume_average(rs, ns, inputs['major_radius'], inputs['minor_radius'], inputs['areal_elongation'])


    s_fusion = popcon.get_s_fusion(ns, Ts, impurities=inputs['impurities'],
                                   reaction=inputs['reaction'])
    s_brem = popcon.get_s_bremsstrahlung(ns, Ts, reaction=inputs['reaction'],
                                         impurities=inputs['impurities'])
    # print('p_fusion = {}'.format(p_fusion))
    # print('p_brem = {}'.format(p_brem))
    s_aux = P_aux(t, inputs['P_aux_0']) / V_plasma

    # Using maximum temperature rather than average for calculating P_ohmic
    js = popcon.get_current_density_profile(inputs['plasma_current'], rs, inputs['minor_radius'],
                                            inputs['inverse_aspect_ratio'], inputs['areal_elongation'])
    s_ohmic = popcon.get_s_ohmic_neoclassical(js, Ts)

    s_ohmic = np.zeros(rs.shape)


    s_heating = 0.2013 * s_fusion + s_aux + s_ohmic
    # print('p_heating = {}'.format(p_heating))

    p_heating = popcon.get_vol_integral(popcon.return_func, s_heating, [0], rs, 
                                        inputs['major_radius'], inputs['areal_elongation'])

    # Using maximum number density rather than average
    tau_E = popcon.get_energy_confinement_time(method=inputs['confinement']['scaling'],
                                               p_external=p_heating,
                                               plasma_current=inputs['plasma_current'],
                                               major_radius=inputs['major_radius'],
                                               minor_radius=inputs['minor_radius'],
                                               kappa=inputs['areal_elongation'],
                                               density=n_ave,
                                               magnetic_field_on_axis=inputs['magnetic_field_on_axis'],
                                               H=inputs['confinement']['H'],
                                               A=2.5)
    # tau_E = 1.2 * ureg.second
    
    s_cond = popcon.get_s_loss(ns, Ts, energy_confinement_time=tau_E, reaction=inputs['reaction'], impurities=inputs['impurities'])
    # print('p_cond = {}'.format(p_cond))
    ignition_frac = get_ignition_frac(rs, 0.2013 * s_fusion, s_brem, s_cond, 
                                      inputs['major_radius'], inputs['areal_elongation'])

    if inputs['P_SOL_method'] == 'total':
        dwdt = (s_heating - s_brem - s_cond)
    elif inputs['P_SOL_method'] == 'partial':
        dwdt = (s_heating - s_cond)
    dwdt = dwdt.to(ureg.watt / ureg.meter**3)
    return dwdt, ignition_frac, n0, dn0dt

def get_dWdt(t, T0, inputs):
    rs = np.linspace(0, inputs['minor_radius'])
    dwdt_density, ignition_frac, n0, _ = get_dWdt_density(t, T0, inputs)
    dWdt = popcon.get_vol_integral(popcon.return_func, dwdt_density, [0], rs, 
                                   inputs['major_radius'], inputs['areal_elongation'])
    return dWdt.to(ureg.MW).magnitude, ignition_frac.magnitude, n0.to(ureg.m**(-3)).magnitude


def get_dTdt(t, T0, inputs):
    rs = np.linspace(0, inputs['minor_radius'])
    dwdt, ignition_fraction, n0, dn0dt = get_dWdt_density(t, T0, inputs)

    ns = popcon.get_parabolic_profile(n0, rs, inputs['minor_radius'],
                                      inputs['n_edge_factor']*n0,
                                      alpha=inputs['profile_alpha']['n'])
    dndts = popcon.get_parabolic_profile(dn0dt, rs, inputs['minor_radius'],
                                      inputs['n_edge_factor']*dn0dt,
                                      alpha=inputs['profile_alpha']['n'])
    
    Ts = popcon.get_parabolic_profile(T0*ureg.keV, rs, inputs['minor_radius'],
                                      inputs['T_edge'],
                                      alpha=inputs['profile_alpha']['n'])
    
    V_plasma = 2 * np.pi**2 * inputs['major_radius'] * inputs['minor_radius']**2 * inputs['areal_elongation']


    dTdt = (dwdt/3 - dndts * Ts) / (ns)
    dTdt_ave = popcon.get_vol_integral(popcon.return_func, dTdt, [0], rs, 
                                   inputs['major_radius'], inputs['areal_elongation']) / V_plasma
    dTdt_ave = dTdt_ave.to(ureg.keV / ureg.second)
    print(dTdt_ave)
    return dTdt_ave.magnitude

sol = scipy.integrate.solve_ivp(get_dTdt, 
                                [0, 300], 
                                np.array([T_at_0.magnitude]),
                                args=(inputs,),
                                method='RK45',
                                t_eval=np.linspace(0, 300, 500))
V_plasma = 2 * np.pi**2 * inputs['major_radius'] * inputs['minor_radius']**2 * inputs['areal_elongation']
# Ts = (sol.y.squeeze() *ureg.MJ / ureg.meter**(3) / (3 * n0)).to(ureg.keV)
Ts = sol.y.squeeze() * ureg.keV
print(sol)

print(sol.y)
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[10, 6])
ax[1,0].plot(sol.t, Ts)
ax[1,0].set_xlabel('Time [s]')
ax[1,0].set_ylabel('T0 [keV]')
# ax[1,0].set_title(('$P_{aux}=$' + '{:.0f} MW,'.format(inputs['P_aux_0'].to(ureg.MW).magnitude) \
#               + ' $n_0=$' + '{:.1e} '.format(n0.to(ureg.m**(-3)).magnitude) + 'm${}^{-3}$'))

start_ramp = False
n0s = np.zeros(sol.t.shape)
dTdts1 = np.zeros(sol.t.shape)
dWdts1 = np.zeros(sol.t.shape)
ignition_fractions = np.zeros(sol.t.shape)
for i, t_i in enumerate(sol.t):
    dWdt_i, ignition_fraction, n0s[i] = get_dWdt(t_i, Ts[i].magnitude, inputs)
    dWdts1[i] = dWdt_i
    dTdts_i = get_dTdt(t_i, Ts[i].magnitude, inputs)
    dTdts1[i] = dTdts_i
    ignition_fractions[i] = ignition_fraction
ax[1,1].plot(sol.t, n0s)
ax[1,1].set_xlabel('Time [s]')
ax[1,1].set_ylabel('n0 [1/m^3]')

ax[0,0].plot(sol.t, dWdts1)
ax[0,0].set_xlabel('Time [s]')
ax[0,0].set_ylabel('dW/dt [MW]')
ax[0,0].grid()

ax[0,1].plot(sol.t, dTdts1)
ax[0,1].set_xlabel('Time [s]')
ax[0,1].set_ylabel('dT/dt [keV/s]')
ax[0,1].grid()

try:
    for a in ax.flatten():
        bottom, top = a.get_ylim()
        a.plot([t0]*2, [bottom, top], '--r', alpha=0.5)
except:
    print('No t0')


# Ts2 = np.linspace(0, 50)
# T_aves = np.zeros(Ts2.shape)

# dTdts2 = np.zeros(Ts2.shape)
# dWdts2 = np.zeros(Ts2.shape)
# ignition_fractions = np.zeros(Ts2.shape)
# tau_Es = np.zeros(Ts2.shape)
# start_ramp = False
# for i in range(len(dTdts2)):
#     dWdts2[i], ignition_fractions[i], _ = get_dWdt(0, Ts2[i], inputs)
#     print('Ignition Fraction: {}'.format(ignition_fractions[i]))
#     dTdt_i = get_dTdt(0, Ts2[i], inputs)
#     dTdts2[i] = dTdt_i
    # T_aves[i] = popcon.get_n_ave(Ts2[i]*ureg.keV, inputs['T_edge']/(Ts2[i]*ureg.keV), inputs['profile_alpha']['n']).magnitude
# print(Ts.shape)
# print(dWdts.shape)

ignition_mask = ignition_fractions >= 1.0
# # fig2, ax2 = plt.subplots(nrows=1, ncols=1)
# # ax2 = [ax2]
# ax[0,1].plot(Ts2, dTdts2)
# ax[0,1].set_ylabel('$dT/dt$ [keV/s]')
# ax[0,1].set_xlabel('T0 [keV]')
# ax[0,1].grid()
# if ignition_mask.any():
#     bottom, top = ax[0,1].get_ylim()
#     ax[0,1].add_patch(matplotlib.patches.Rectangle((Ts2[ignition_mask][0], bottom),
#                                                 Ts2[ignition_mask][-1] - Ts2[ignition_mask][0],
#                                                 top - bottom,
#                                                 color='red',
#                                                 alpha=0.3
#                                                 ))

# # # ax2[1].plot(T_aves, tau_Es)
# # # ax2[1].set_ylabel('tau_E [s]')
# # # ax2[1].set_yscale('log')
# # # for a in ax2:
# # #     a.set_xlabel('T0 [keV]')
# # #     a.grid()


# # # fig3, ax3 = plt.subplots(nrows=1, ncols=1)
# ax[0,0].plot(Ts2, dWdts2)
# ax[0,0].set_ylabel('$dW/dt$ [MJ/m^3]')
# ax[0,0].set_xlabel('T_ave [keV]')
# ax[0,0].grid()
# if ignition_mask.any():
#     bottom, top = ax[1,1].get_ylim()
#     ax[1,1].add_patch(matplotlib.patches.Rectangle((Ts2[ignition_mask][0], bottom),
#                                                 Ts2[ignition_mask][-1] - Ts2[ignition_mask][0],
#                                                 top - bottom,
#                                                 color='red',
#                                                 alpha=0.3
#                                                 ))

fig.suptitle('$P_{aux}=$' + '{:.0f} MW,'.format(inputs['P_aux_0'].to(ureg.MW).magnitude) \
              + ' n_0_slope=' + '{:.1e}'.format(inputs['n0_slope'].magnitude) + ' $m^{-3} s^{-1}$')

fig.tight_layout()

fig.savefig('P_aux={:.0f}MW_n0slope={:.1e}_Ip={:.1f}MA.png'.format(inputs['P_aux_0'].to(ureg.MW).magnitude,
                                                             inputs['n0_slope'].magnitude,
                                                             inputs['plasma_current'].to(ureg.MA).magnitude))





plt.show()