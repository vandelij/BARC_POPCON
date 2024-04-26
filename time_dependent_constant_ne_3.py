import popcon
import scipy.integrate
import numpy as np
import pint
ureg = pint.get_application_registry()
from matplotlib import pyplot as plt

T_at_0 = 0.1 * ureg.keV
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
inputs['plasma_current'] = 18 * ureg.MA
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
inputs['profile_alpha']['T'] = 1.75
inputs['T_edge'] = 0.1 * ureg.keV

# Electron Density Inputs
inputs['profile_alpha']['n'] = 1.5
inputs['n_edge_factor'] = 0.1

inputs['P_aux_0'] = 50 * ureg.MW

inputs['minor_radius'] = inputs['major_radius'] * inputs['inverse_aspect_ratio']

def n0_func(t, T0, tau_E=1*ureg.second):
    if T0<15*ureg.keV:
        n0 = 5e20 * ureg.m**(-3)
    else:
        n0 = 5e20 * ureg.m**(-3)
        # n0 = 2e20*ureg.m**(-3) + 1e19*ureg.m**(-3) * (t*ureg.second/tau_E)
    return n0

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
    print(T0)
    T0 *= ureg.keV

    n0 = n0_func(t, T0)
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

    ignition_frac = get_ignition_frac(rs, 0.2013 * s_fusion, s_brem, s_cond, 
                                      inputs['major_radius'], inputs['areal_elongation'])
    # print('p_cond = {}'.format(p_cond))
    if inputs['P_SOL_method'] == 'total':
        dwdt = (s_heating - s_brem - s_cond)
    elif inputs['P_SOL_method'] == 'partial':
        dwdt = (s_heating - s_cond)
    dwdt = dwdt.to(ureg.watt / ureg.meter**3)
    return dwdt, ignition_frac

def get_dWdt(t, T0, inputs, n0):
    rs = np.linspace(0, inputs['minor_radius'])
    dwdt_density, ignition_frac = get_dWdt_density(t, T0, inputs)
    dWdt = popcon.get_vol_integral(popcon.return_func, dwdt_density, [0], rs, 
                                   inputs['major_radius'], inputs['areal_elongation'])
    return dWdt.to(ureg.MW).magnitude


def get_dTdt(t, T0, inputs, n0):
    rs = np.linspace(0, inputs['minor_radius'])
    n0 = n0_func(t, T0*ureg.keV)
    ns = popcon.get_parabolic_profile(n0, rs, inputs['minor_radius'],
                                      inputs['n_edge_factor']*n0,
                                      alpha=inputs['profile_alpha']['n'])
    
    V_plasma = 2 * np.pi**2 * inputs['major_radius'] * inputs['minor_radius']**2 * inputs['areal_elongation']

    dwdt, ignition_frac = get_dWdt_density(t, T0, inputs)

    dTdt = dwdt / (3 * ns)
    dTdt_ave = popcon.get_vol_integral(popcon.return_func, dTdt, [0], rs, 
                                   inputs['major_radius'], inputs['areal_elongation']) / V_plasma
    dTdt_ave = dTdt_ave.to(ureg.keV / ureg.second)
    print(dTdt_ave)
    print('Ignition Fraction = {}'.format(ignition_frac))
    return dTdt_ave.magnitude

sol = scipy.integrate.solve_ivp(get_dTdt, 
                                [0, 500], 
                                np.array([T_at_0.magnitude]),
                                args=(inputs, n0),
                                method='RK45')
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


Ts2 = np.linspace(0, 50)
T_aves = np.zeros(Ts2.shape)

dTdts = np.zeros(Ts2.shape)
dWdts2 = np.zeros(Ts2.shape)
tau_Es = np.zeros(Ts2.shape)

for i in range(len(dTdts)):
    dWdts2[i] = get_dWdt(0, Ts2[i], inputs, n0)
    dTdt_i = get_dTdt(0, Ts2[i], inputs, n0)
    dTdts[i] = dTdt_i
    # T_aves[i] = popcon.get_n_ave(Ts2[i]*ureg.keV, inputs['T_edge']/(Ts2[i]*ureg.keV), inputs['profile_alpha']['n']).magnitude
# print(Ts.shape)
# print(dWdts.shape)

# fig2, ax2 = plt.subplots(nrows=1, ncols=1)
# ax2 = [ax2]
ax[0,1].plot(Ts2, dTdts)
ax[0,1].set_ylabel('$dT/dt$ [keV/s]')
ax[0,1].set_xlabel('T0 [keV]')
ax[0,1].grid()

# ax2[1].plot(T_aves, tau_Es)
# ax2[1].set_ylabel('tau_E [s]')
# ax2[1].set_yscale('log')
# for a in ax2:
#     a.set_xlabel('T0 [keV]')
#     a.grid()


# fig3, ax3 = plt.subplots(nrows=1, ncols=1)
ax[0,0].plot(Ts2, dWdts2)
ax[0,0].set_ylabel('$dW/dt$ [MJ/m^3]')
ax[0,0].set_xlabel('T_ave [keV]')
ax[0,0].grid()

fig.suptitle('$P_{aux}=$' + '{:.0f} MW,'.format(inputs['P_aux_0'].to(ureg.MW).magnitude) \
              + ' $n_0=$' + '{:.1e} '.format(n0.to(ureg.m**(-3)).magnitude) + 'm${}^{-3}$')
fig.tight_layout()

fig.savefig('P_aux={:.0f}MW_n0={:.1e}_Ip={:.1f}MA.png'.format(inputs['P_aux_0'].to(ureg.MW).magnitude,
                                                             n0.magnitude,
                                                             inputs['plasma_current'].to(ureg.MA).magnitude))



# ave_factor = np.sqrt(5.34e4 / (1.6022e-19 * 1e20 * 1000 * 3))
# plasma_volume = 960 * ureg.meter**(3)
# print(ave_factor)
# Ts3 = np.linspace(0, 50) * ureg.keV
# dWdts3 = np.zeros(Ts3.shape)
# tau_Es3 = np.zeros(Ts3.shape)
# n_bar = 1.1e20 * ureg.m**(-3)
# for i in range(len(Ts3)):
#     s_fusion = ave_factor**5 * popcon.get_s_fusion(n_bar, Ts3[i])
#     s_brem = ave_factor**2.5 * popcon.get_s_bremsstrahlung(n_bar, Ts3[i])
#     s_heating = s_fusion * 0.2
#     tau_E = popcon.get_energy_confinement_time(method=inputs['confinement']['scaling'],
#                                                p_external=s_heating*plasma_volume,
#                                                plasma_current=inputs['plasma_current'],
#                                                major_radius=inputs['major_radius'],
#                                                minor_radius=inputs['minor_radius'],
#                                                kappa=inputs['areal_elongation'],
#                                                density=n_bar,
#                                                magnetic_field_on_axis=inputs['magnetic_field_on_axis'],
#                                                H=inputs['confinement']['H'],
#                                                A=2.5)
#     # tau_E = 1.2 * ureg.second
#     tau_Es3[i] = tau_E.magnitude
#     s_cond = ave_factor**2 * popcon.get_s_loss(n_bar, Ts3[i], energy_confinement_time=tau_E)

#     dWdt = (s_heating - s_brem - s_cond) * plasma_volume
#     dWdts3[i] = dWdt.to(ureg.MW).magnitude


# fig3, ax3 = plt.subplots(nrows=1, ncols=2)
# ax3[0].plot(Ts3, dWdts3)
# ax3[0].set_ylabel('$dW/dt$ [MJ/m^3]')

# ax3[1].plot(Ts3, tau_Es3)
# ax3[1].set_ylabel('tau_E [s]')
# ax3[1].set_yscale('log')
# for a in ax3:
#     a.set_xlabel('T0 [keV]')
#     a.grid()

# rs = np.linspace(0, inputs['minor_radius'])

# print(popcon.get_energy_confinement_time(method=inputs['confinement']['scaling'],
#                                                p_external=s_heating*plasma_volume,
#                                                plasma_current=inputs['plasma_current'],
#                                                major_radius=inputs['major_radius'],
#                                                minor_radius=inputs['minor_radius'],
#                                                kappa=inputs['areal_elongation'],
#                                                density=n_bar,
#                                                magnetic_field_on_axis=inputs['magnetic_field_on_axis'],
#                                                H=inputs['confinement']['H'],
#                                                A=2.5) * (s_heating*plasma_volume)**0.69)

# n0 = 2e20 * ureg.m**(-3)
# T0 = 34 * ureg.keV

# n_ave = 1.1e20 * ureg.m**(-3)
# T_ave = 15.7 * ureg.keV
# print('n_ave = {}'.format(popcon.get_n_ave(n0, inputs['n_edge_factor'], inputs['profile_alpha']['n'])))
# print('T_ave = {}'.format(popcon.get_n_ave(T0, inputs['T_edge']/T0, inputs['profile_alpha']['n'])))

# ns = popcon.get_parabolic_profile(n0, rs, inputs['minor_radius'],
#                                       inputs['n_edge_factor']*n0,
#                                       alpha=inputs['profile_alpha']['n'])
# Ts = popcon.get_parabolic_profile(T0, rs, inputs['minor_radius'],
#                                       inputs['T_edge'],
#                                       alpha=inputs['profile_alpha']['n'])
# p_fusion = popcon.get_p_fusion(ns, Ts, rs, inputs['areal_elongation'],
#                                    inputs['major_radius'], reaction=inputs['reaction'],
#                                    impurities=inputs['impurities'], f_DT=inputs['fuel_ratio'])
# s_fusion = popcon.get_s_fusion(n_ave, T_ave)
# print('s_fusion = {}'.format(s_fusion))
# print('s_fusion * volume = {}'.format(s_fusion * plasma_volume))
    
# p_brem = popcon.get_p_bremmstrahlung(ns, Ts, rs, inputs['areal_elongation'],
#                                          inputs['major_radius'], reaction=inputs['reaction'],
#                                          impurities=inputs['impurities'])
# print('P_fusion = {}'.format(p_fusion))
# print('P_brem = {}'.format(p_brem))



plt.show()