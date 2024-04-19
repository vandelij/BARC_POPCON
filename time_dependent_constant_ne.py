import popcon
import scipy.integrate
import numpy as np
import pint
ureg = pint.get_application_registry()
from matplotlib import pyplot as plt

T_at_0 = 0.1 * ureg.keV
n0 = 2.0e20 * ureg.m**(-3)

inputs = {}
kr_frac = 1.1e-3
inputs['reaction'] = 'DT'
# inputs['impurities'] =  [[36, kr_frac]]
inputs['impurities'] = None
inputs['fuel_ratio'] = 1.0
# inputs['impurities'] = None

inputs['major_radius'] = 7.1 * ureg.meter
inputs['inverse_aspect_ratio'] = 2/7.1
inputs['minor_radius'] = inputs['major_radius'] * inputs['inverse_aspect_ratio']
inputs['areal_elongation'] = 1.7
inputs['plasma_current'] = 17 * ureg.MA
inputs['magnetic_field_on_axis'] = 7.1 *ureg.tesla

inputs['confinement'] = {}
# Scaling options are 'ITER98y2', 'ITER97', or 'ITER89'
inputs['confinement']['scaling'] = 'ITER98y2'
inputs['confinement']['H'] = 1.00
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
inputs['profile_alpha']['n'] = 1.5
inputs['n_edge_factor'] = 0.1

inputs['P_aux_0'] = 0 * ureg.MW

inputs['minor_radius'] = inputs['major_radius'] * inputs['inverse_aspect_ratio']

def P_aux(t, P_aux_0):
    return P_aux_0

def get_dTdt(t, T0, inputs, n0):
    # print(T0)
    T0 *= ureg.keV
    rs = np.linspace(0, inputs['minor_radius'])
    ns = popcon.get_parabolic_profile(n0, rs, inputs['minor_radius'],
                                      inputs['n_edge_factor']*n0,
                                      alpha=inputs['profile_alpha']['n'])
    Ts = popcon.get_parabolic_profile(T0, rs, inputs['minor_radius'],
                                      inputs['T_edge'],
                                      alpha=inputs['profile_alpha']['n'])
    
    p_fusion = popcon.get_p_fusion(ns, Ts, rs, inputs['areal_elongation'],
                                   inputs['major_radius'], reaction=inputs['reaction'],
                                   impurities=inputs['impurities'], f_DT=inputs['fuel_ratio'])
    
    p_brem = popcon.get_p_bremmstrahlung(ns, Ts, rs, inputs['areal_elongation'],
                                         inputs['major_radius'], reaction=inputs['reaction'],
                                         impurities=inputs['impurities'])
    # print('p_fusion = {}'.format(p_fusion))
    # print('p_brem = {}'.format(p_brem))
    p_aux = P_aux(t, inputs['P_aux_0'])

    # Using maximum temperature rather than average for calculating P_ohmic
    # p_ohmic = popcon.get_p_ohmic_neoclassical(inputs['plasma_current'],
    #                                           T0,
    #                                           inputs['inverse_aspect_ratio'],
    #                                           inputs['major_radius'],
    #                                           inputs['areal_elongation'])
    p_ohmic = 0

    p_heating = 0.2013 * p_fusion + p_aux + p_ohmic
    # print('p_heating = {}'.format(p_heating))

    # Using maximum number density rather than average
    tau_E = popcon.get_energy_confinement_time(method=inputs['confinement']['scaling'],
                                               p_external=p_heating,
                                               plasma_current=inputs['plasma_current'],
                                               major_radius=inputs['major_radius'],
                                               minor_radius=inputs['minor_radius'],
                                               kappa=inputs['areal_elongation'],
                                               density=n0/2,
                                               magnetic_field_on_axis=inputs['magnetic_field_on_axis'],
                                               H=inputs['confinement']['H'],
                                               A=2.5)
    # tau_E = 1.2 * ureg.second
    
    p_cond = popcon.get_p_total_loss(ns, Ts, rs, inputs['major_radius'],
                                     inputs['areal_elongation'], 
                                     energy_confinement_time=tau_E,
                                     reaction=inputs['reaction'],
                                     impurities=inputs['impurities'])
    # print('p_cond = {}'.format(p_cond))
    V_plasma = 2 * np.pi**2 * inputs['major_radius'] * inputs['minor_radius']**2 * inputs['areal_elongation']

    if inputs['P_SOL_method'] == 'total':
        dWdt = (p_heating - p_brem - p_cond)
    elif inputs['P_SOL_method'] == 'partial':
        dWdt = (p_heating - p_cond)
    return dWdt

# sol = scipy.integrate.solve_ivp(get_dTdt, 
#                                 [0, 2], 
#                                 np.array([T_at_0.magnitude]),
#                                 args=(inputs, n0),
#                                 method='Radau')
# V_plasma = 2 * np.pi**2 * inputs['major_radius'] * inputs['minor_radius']**2 * inputs['areal_elongation']
# Ts = (sol.y.squeeze() *ureg.MJ / ureg.meter**(3) / (3 * n0)).to(ureg.keV)
# print(sol)

# print(sol.y)
# fig, ax = plt.subplots()
# ax.plot(sol.t, Ts)
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('T0 [keV]')


Ts2 = np.linspace(0, 50)

dWdts = np.zeros(Ts2.shape)
for i in range(len(dWdts)):
    dWdts[i] = get_dTdt(0, Ts2[i], inputs, n0).magnitude

# print(Ts.shape)
# print(dWdts.shape)

fig2, ax2 = plt.subplots()
ax2.plot(Ts2, dWdts)
ax2.set_xlabel('T0 [keV]')
ax2.set_ylabel('$dW/dt$ [MJ/m^3]')

rs = np.linspace(0, inputs['minor_radius'])

n0 = 2e20 * ureg.m**(-3)
T0 = 34 * ureg.keV
print('n_ave = {}'.format(popcon.get_n_ave(n0, inputs['n_edge_factor'], inputs['profile_alpha']['n'])))
print('T_ave = {}'.format(popcon.get_n_ave(T0, inputs['T_edge']/T0, inputs['profile_alpha']['n'])))

ns = popcon.get_parabolic_profile(n0, rs, inputs['minor_radius'],
                                      inputs['n_edge_factor']*n0,
                                      alpha=inputs['profile_alpha']['n'])
Ts = popcon.get_parabolic_profile(T0, rs, inputs['minor_radius'],
                                      inputs['T_edge'],
                                      alpha=inputs['profile_alpha']['n'])
p_fusion = popcon.get_p_fusion(ns, Ts, rs, inputs['areal_elongation'],
                                   inputs['major_radius'], reaction=inputs['reaction'],
                                   impurities=inputs['impurities'], f_DT=inputs['fuel_ratio'])
    
p_brem = popcon.get_p_bremmstrahlung(ns, Ts, rs, inputs['areal_elongation'],
                                         inputs['major_radius'], reaction=inputs['reaction'],
                                         impurities=inputs['impurities'])
print('P_fusion = {}'.format(p_fusion))
print('P_brem = {}'.format(p_brem))



plt.show()