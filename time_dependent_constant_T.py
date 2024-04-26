import popcon
import scipy.integrate
import scipy.optimize
import numpy as np
import pint
ureg = pint.get_application_registry()
from matplotlib import pyplot as plt
import matplotlib.patches

T_at_0 = 1 * ureg.keV
n0 = 1e20 # m^(-3)

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
inputs['plasma_current'] = 16 * ureg.MA
inputs['magnetic_field_on_axis'] = 10.51 *ureg.tesla

inputs['confinement'] = {}
# Scaling options are 'ITER98y2', 'ITER97', or 'ITER89'
inputs['confinement']['scaling'] = 'ITER89'
inputs['confinement']['H'] = 1.3
inputs['confinement']['lower_bound'] = 1e-5
inputs['confinement']['upper_bound'] = 5000
inputs['A'] = 2.5

# Method for calculating P_SOL
inputs['P_SOL_method'] = 'partial'

inputs['num_r_points'] = 50

# Electron Temperature Inputs
inputs['T_min'] = 1 * ureg.keV
inputs['T_max'] = 30 * ureg.keV
inputs['num_T_points'] = 20
inputs['profile_alpha'] = {}
inputs['profile_alpha']['T'] = 1.75
inputs['T_edge'] = 0.1 * ureg.keV

# Electron Density Inputs
inputs['n_min'] = 0.01e20 * ureg.meter**(-3)
inputs['n_max'] = 5e20 * ureg.meter**(-3)
inputs['num_n_points'] = 20
inputs['profile_alpha']['n'] = 1.5
inputs['n_edge_factor'] = 0.3

inputs['P_aux_0'] = 20 * ureg.MW

inputs['n0_slope'] = 1e18 * ureg.m**(-3)
inputs['n0_start'] = 1e20 

inputs['density_ramp_temperature'] = 19.5 * ureg.keV

inputs['minor_radius'] = inputs['major_radius'] * inputs['inverse_aspect_ratio']

inputs['reduce_P_aux'] = False


def P_aux(t, P_aux_0):
    return P_aux_0

def get_P_aux(t, P_aux_0, reduce=False):
    if reduce:
        P_aux = P_aux_0 - 0.1*ureg.MW * t
        # P_aux = P_aux_0
        if P_aux < 0:
            P_aux = 0 * ureg.MW
    else:
        P_aux = P_aux_0
    return P_aux

def get_ignition_frac(rs, s_alpha, s_brem, s_loss, major_radius, areal_elongation, method='partial'):
    p_alpha = popcon.get_vol_integral(popcon.return_func, s_alpha, [0], rs, 
                            major_radius, areal_elongation)
    p_brem = popcon.get_vol_integral(popcon.return_func, s_brem, [0], rs, 
                            major_radius, areal_elongation)
    p_loss = popcon.get_vol_integral(popcon.return_func, s_loss, [0], rs, 
                            major_radius, areal_elongation)
    if method=='partial':
        ignition_frac = p_alpha / p_loss
    else:
        ignition_frac = p_alpha / (p_brem + p_loss)
    return ignition_frac


def get_dWdt_density(t, T0, inputs, n0):
    T0 *= ureg.keV
    n0 *= ureg.m**(-3)
    dn0dt = 0 * ureg.m**(-3) / ureg.second
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
    s_aux = get_P_aux(t, inputs['P_aux_0'], reduce=inputs['reduce_P_aux']) / V_plasma
    # print('S_aux = {}'.format(s_aux))

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
    
    s_loss = popcon.get_s_loss(ns, Ts, energy_confinement_time=tau_E, reaction=inputs['reaction'], impurities=inputs['impurities'])
    # print('p_cond = {}'.format(p_cond))
    ignition_frac = get_ignition_frac(rs, 0.2013 * s_fusion, s_brem, s_loss, 
                                      inputs['major_radius'], inputs['areal_elongation'])

    if inputs['P_SOL_method'] == 'total':
        dwdt = (s_heating - s_brem - s_loss)
    elif inputs['P_SOL_method'] == 'partial':
        dwdt = (s_heating - s_loss)
    dwdt = dwdt.to(ureg.watt / ureg.meter**3)
    return dwdt, ignition_frac, dn0dt

def get_dWdt(t, T0, inputs, n0):
    rs = np.linspace(0, inputs['minor_radius'])
    dwdt_density, ignition_frac, *_ = get_dWdt_density(t, T0, inputs, n0)
    dWdt = popcon.get_vol_integral(popcon.return_func, dwdt_density, [0], rs, 
                                   inputs['major_radius'], inputs['areal_elongation'])
    return dWdt.to(ureg.MW).magnitude, ignition_frac.magnitude


def get_dTdt(T0, t, inputs, n0):
    # print(T0)
    rs = np.linspace(0, inputs['minor_radius'])
    dwdt, ignition_fraction, dn0dt = get_dWdt_density(t, T0, inputs, n0)
    ns = popcon.get_parabolic_profile(n0*ureg.m**(-3), rs, inputs['minor_radius'],
                                      inputs['n_edge_factor']*n0*ureg.m**(-3),
                                      alpha=inputs['profile_alpha']['n'])
    dndts = popcon.get_parabolic_profile(dn0dt, rs, inputs['minor_radius'],
                                      inputs['n_edge_factor']*dn0dt,
                                      alpha=inputs['profile_alpha']['n'])
    
    Ts = popcon.get_parabolic_profile(T0*ureg.keV, rs, inputs['minor_radius'],
                                      inputs['T_edge'],
                                      alpha=inputs['profile_alpha']['n'])
    
    V_plasma = 2 * np.pi**2 * inputs['major_radius'] * inputs['minor_radius']**2 * inputs['areal_elongation']


    dTdt = (dwdt/3) / (ns)
    dTdt_ave = popcon.get_vol_integral(popcon.return_func, dTdt, [0], rs, 
                                   inputs['major_radius'], inputs['areal_elongation']) / V_plasma
    dTdt_ave = dTdt_ave.to(ureg.keV / ureg.second)
    # print(dTdt_ave)
    return dTdt_ave.magnitude


def get_dTdt2(t, T0, inputs, n0):
    # print(T0)
    rs = np.linspace(0, inputs['minor_radius'])
    dwdt, ignition_fraction, dn0dt = get_dWdt_density(t, T0, inputs, n0)
    ns = popcon.get_parabolic_profile(n0*ureg.m**(-3), rs, inputs['minor_radius'],
                                      inputs['n_edge_factor']*n0*ureg.m**(-3),
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
    # print(dTdt_ave)
    return dTdt_ave.magnitude

def get_dndt(t, n0, inputs, T0):
    # print(n0)
    rs = np.linspace(0, inputs['minor_radius'])
    dwdt, ignition_fraction, *_ = get_dWdt_density(t, T0, inputs, n0)

    ns = popcon.get_parabolic_profile(n0*ureg.m**(-3), rs, inputs['minor_radius'],
                                      inputs['n_edge_factor']*n0*ureg.m**(-3),
                                      alpha=inputs['profile_alpha']['n'])
    
    Ts = popcon.get_parabolic_profile(T0*ureg.keV, rs, inputs['minor_radius'],
                                      inputs['T_edge'],
                                      alpha=inputs['profile_alpha']['n'])
    V_plasma = 2 * np.pi**2 * inputs['major_radius'] * inputs['minor_radius']**2 * inputs['areal_elongation']

    # Assuming constant temperature in time
    dndt = dwdt / (3 * Ts)
    dndt_ave = popcon.get_vol_integral(popcon.return_func, dndt, [0], rs,
                                       inputs['major_radius'], inputs['areal_elongation']) / V_plasma
    dndt_ave = dndt_ave.to(ureg.m**(-3) / ureg.second)
    return dndt_ave.magnitude



## Find time and temperature at which dT/dt = 0

# Ts = np.linspace(1, 50)
# dTdts = np.zeros(Ts.shape)
# for i, T0 in enumerate(Ts):
#     dTdts[i] = get_dTdt(T0, 0, inputs, n0)
# d2Tdt2 = np.diff(dTdts)
# neg_slope_mask = d2Tdt2 < 0
# # Use the first temperature with a negative second derivative as the first derivative zero
# # will lead to a stable solution
# lower_bound = Ts[1:][neg_slope_mask][0]

# asymptote_T  = scipy.optimize.toms748(get_dTdt,
#                                 lower_bound,
#                                 50,
#                                 args=(0, inputs, n0))
# print('Asmptotic T0 = {} keV'.format(asymptote_T))

asymptote_T = 20

inputs['reduce_P_aux'] = True
## Calculate n0(t) while T=asymptote_T
sol_n = scipy.integrate.solve_ivp(get_dndt,
                                  [0, 600],
                                  np.array([n0]),
                                  args=(inputs, asymptote_T),
                                  method='RK45',
                                  t_eval=np.linspace(0, 600, 500))
print(sol_n)
ns = sol_n.y.squeeze() * ureg.m**(-3)

dndts = np.zeros(sol_n.t.shape)
dWdts = np.zeros(sol_n.t.shape)
ignition_fractions = np.zeros(sol_n.t.shape)

for i,t_i in enumerate(sol_n.t):
    dWdts[i], ignition_fractions[i] = get_dWdt(t_i, asymptote_T, inputs, ns[i].magnitude)
    dndts[i] = get_dndt(t_i, ns[i].magnitude, inputs, asymptote_T)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[10, 6])

ax[0,0].plot(sol_n.t, dWdts)
ax[0,0].set_xlabel('Time [s]')
ax[0,0].set_ylabel('dW/dt [MW]')
ax[0,0].grid()

ax[0,1].plot(sol_n.t, dndts)
ax[0,1].set_xlabel('Time [s]')
ax[0,1].set_ylabel('dn/dt [$m^{-3} s^{-1}$]')
ax[0,1].grid()

ax[1,0].plot(sol_n.t, ns)
ax[1,0].set_xlabel('Time [s]')
ax[1,0].set_ylabel('n0 [$m^{-3}$]')

ax[1,1].plot(sol_n.t, ignition_fractions)
ax[1,1].set_xlabel('Time [s]')
ax[1,1].set_ylabel('Ignition Fraction')




inputs['reduce_P_aux'] = False

sol = scipy.integrate.solve_ivp(get_dTdt2, 
                                [0, 100], 
                                np.array([T_at_0.magnitude]),
                                args=(inputs,n0),
                                method='RK45',
                                t_eval=np.linspace(0, 100, 200))
Ts = sol.y.squeeze() * ureg.keV
# print(sol)

# print(sol.y)
fig2, ax2= plt.subplots(nrows=2, ncols=2, figsize=[10, 6])
ax2[1,0].plot(sol.t, Ts)
ax2[1,0].set_xlabel('Time [s]')
ax2[1,0].set_ylabel('T0 [keV]')
# # ax[1,0].set_title(('$P_{aux}=$' + '{:.0f} MW,'.format(inputs['P_aux_0'].to(ureg.MW).magnitude) \
# #               + ' $n_0=$' + '{:.1e} '.format(n0.to(ureg.m**(-3)).magnitude) + 'm${}^{-3}$'))

# start_ramp = False
# n0s = np.zeros(sol.t.shape)
dTdts1 = np.zeros(sol.t.shape)
dWdts1 = np.zeros(sol.t.shape)
ignition_fractions2 = np.zeros(sol.t.shape)
for i, t_i in enumerate(sol.t):
    dWdt_i, ignition_fraction2 = get_dWdt(t_i, Ts[i].magnitude, inputs, n0)
    dWdts1[i] = dWdt_i
    dTdts_i = get_dTdt2(t_i, Ts[i].magnitude, inputs, n0)
    dTdts1[i] = dTdts_i
    ignition_fractions2[i] = ignition_fraction2
# ax[1,1].plot(sol.t, n0s)
# ax[1,1].set_xlabel('Time [s]')
# ax[1,1].set_ylabel('n0 [1/m^3]')

ax2[0,0].plot(sol.t, dWdts1)
ax2[0,0].set_xlabel('Time [s]')
ax2[0,0].set_ylabel('dW/dt [MW]')
ax2[0,0].grid()

ax2[0,1].plot(sol.t, dTdts1)
ax2[0,1].set_xlabel('Time [s]')
ax2[0,1].set_ylabel('dT/dt [keV/s]')
ax2[0,1].grid()


# ns2 = np.linspace(0.0e20, 2e21, 100)
# Ts2 = np.linspace(0, 40, 50)
# dndts2 = np.zeros((len(Ts2), len(ns2)))
# dWdts2 = np.zeros((len(Ts2), len(ns2)))
# # inputs['P_aux_0'] = 20 * ureg.MW
# # temp = asymptote_T
# # temp = 7
# for i, n_i in enumerate(ns2):
#     for j,T_i in enumerate(Ts2):
#         dWdt_i, ignition_fraction2 = get_dWdt([0], T_i, inputs, n_i)
#         dWdts2[j,i] = dWdt_i
#         # dndts_i = get_dndt(0, n_i, inputs, T_i)
#         # dndts2[i,j] = dndts_i

# fig3, ax3 = plt.subplots(1,1, figsize=[10,8])
# xmesh, ymesh = np.meshgrid(Ts2, ns2)
# print(np.max(np.abs(dWdts2)))
# cntr = ax3.contourf(xmesh, ymesh, dWdts2.transpose(), 
#                     cmap='seismic', 
#                     vmin=-np.nanmax(np.abs(dWdts2)),
#                     vmax=np.nanmax(np.abs(dWdts2)))
# ax3.set_xlabel('T0 [keV]')
# ax3.set_ylabel('n0 [$m^{-3}$]')
# cbar = plt.colorbar(cntr, label='dW/dt [MW]')
# ax3.set_title('P_aux={} MW, Ip={:.1f} MA, alpha_T={:.2f}, alpha_n={:.2f}'.format(inputs['P_aux_0'].magnitude,
#                                                               inputs['plasma_current'].magnitude,
#                                                               inputs['profile_alpha']['T'],
#                                                               inputs['profile_alpha']['n']
#                                                               ))

# fig3.savefig('dWdt_cntr_Paux={}MW_Ip={:.1f}MA_alpha_T={:.2f}_alpha_n={:.2f}.png'.format(
#                                                               inputs['P_aux_0'].magnitude,
#                                                               inputs['plasma_current'].magnitude,
#                                                               inputs['profile_alpha']['T'],
#                                                               inputs['profile_alpha']['n']
#                                                               ))

ns4 = np.linspace(0.0e20, 1e21, 50)
Pauxs4 = np.linspace(0, 20, 20)
dndts4 = np.zeros((len(Pauxs4), len(ns4)))
dWdts4 = np.zeros((len(Pauxs4), len(ns4)))
# inputs['P_aux_0'] = 20 * ureg.MW
# temp = asymptote_T
# temp = 7
for i, n_i in enumerate(ns4):
    for j,P_aux_i in enumerate(Pauxs4):
        inputs['P_aux_0'] = P_aux_i * ureg.MW
        dWdt_i, ignition_fraction2 = get_dWdt([0], asymptote_T, inputs, n_i)
        dWdts4[j,i] = dWdt_i
        # dndts_i = get_dndt(0, n_i, inputs, T_i)
        # dndts2[i,j] = dndts_i

fig4, ax4 = plt.subplots(1,1, figsize=[10,8])
xmesh4, ymesh4 = np.meshgrid(Pauxs4, ns4)
print(np.max(np.abs(dWdts4)))
cntr4 = ax4.contourf(xmesh4, ymesh4, dWdts4.transpose(), 
                    cmap='seismic', 
                    vmin=-np.nanmax(np.abs(dWdts4)),
                    vmax=np.nanmax(np.abs(dWdts4)))
ax4.set_xlabel('P_aux [MW]')
ax4.set_ylabel('n0 [$m^{-3}$]')
cbar = plt.colorbar(cntr4, label='dW/dt [MW]')
ax4.set_title('T0={} keV, Ip={:.1f} MA, alpha_T={:.2f}, alpha_n={:.2f}'.format(asymptote_T,
                                                              inputs['plasma_current'].magnitude,
                                                              inputs['profile_alpha']['T'],
                                                              inputs['profile_alpha']['n']
                                                              ))

fig4.savefig('dWdt_cntr_T0={:.0f}keV_Ip={:.1f}MA_alpha_T={:.2f}_alpha_n={:.2f}.png'.format(
                                                              asymptote_T,
                                                              inputs['plasma_current'].magnitude,
                                                              inputs['profile_alpha']['T'],
                                                              inputs['profile_alpha']['n']
                                                              ))


# fig3, ax3 = plt.subplots(nrows=1, ncols=2, figsize=[10, 6])
# ax3[0].plot(ns2, dWdts2)
# ax3[0].set_xlabel('n0')
# ax3[0].set_ylabel('dW/dt')

# ax3[1].plot(ns2, dndts2)
# ax3[1].set_xlabel('n0')
# ax3[1].set_ylabel('dn0/dt')





# fig.suptitle('$P_{aux}=$' + '{:.0f} MW,'.format(inputs['P_aux_0'].to(ureg.MW).magnitude) \
#               + ' n_0_slope=' + '{:.1e}'.format(inputs['n0_slope'].magnitude) + ' $m^{-3} s^{-1}$')

# fig.tight_layout()

# fig.savefig('P_aux={:.0f}MW_n0slope={:.1e}_Ip={:.1f}MA.png'.format(inputs['P_aux_0'].to(ureg.MW).magnitude,
#                                                              inputs['n0_slope'].magnitude,
#                                                              inputs['plasma_current'].to(ureg.MA).magnitude))

######## Plot Inputs ##########
plot_inputs = {}
plot_inputs['filename'] = 'popcon_ARCH.png'
plot_inputs['contours'] = {}
plot_inputs['contours']['P_fusion'] = {'levels': [50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000], 'colors':'black'}
# plot_inputs['contours']['P_ohmic'] = {'levels': [1, 10, 100, 1000, 1e4]}
plot_inputs['contours']['P_auxillary'] = {'levels': [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80], 'colors':'red'}
# plot_inputs['contours']['P_SOL'] = {'levels': [10, 100, 1000, 1e4]}
# plot_inputs['contours']['P_radiation'] = {'levels': [10, 100, 1000, 1e4]}
# plot_inputs['contours']['energy_confinement_time'] = {'levels':[0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 10.0]}
plot_inputs['contours']['Q'] = {'levels': [1, 10, 20, 40, 1000], 'colors':'tab:purple'}
plot_inputs['contours']['peak_greenwald_fraction'] = {'levels': [1.0], 'colors':'green'}
# plot_inputs['contours']['sepOS_density_fraction'] = {'levels': [0.1, 1.0, 5.0, 10.0, 20.0], 'colors':'blue'}
# plot_inputs['contours']['bernert_density_fraction'] = {'levels': [0.1, 1.0, 5.0, 10.0, 20.0], 'colors':'grey'}
plot_inputs['contours']['P_LH_fraction'] = {'levels': [1.0, 5.0, 10.0, 20.0], 'colors':'gold'}

plot_inputs['plot_ignition'] = True

output = popcon.get_all_parameters(inputs)

fig_p, ax_p = popcon.plot_popcon(output, plot_inputs)

# Plot temperature, density path on popcon
ax_p.plot(Ts, [n0]*len(Ts), '-', color='gray', linewidth=3)
ax_p.plot([asymptote_T]*len(ns), ns, '-', color='gray', linewidth=3)



plt.show()