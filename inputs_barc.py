import popcon
import pint
ureg = pint.get_application_registry()
from matplotlib import pyplot as plt

inputs = {}

inputs['reaction'] = 'DT'
inputs['impurities'] = [[36, 1.1e-3]]
# inputs['impurities'] = None

inputs['major_radius'] = 4.5 * ureg.meter
inputs['inverse_aspect_ratio'] = 0.3
inputs['areal_elongation'] = 1.8
inputs['plasma_current'] = 18 * ureg.MA
inputs['magnetic_field_on_axis'] = 10.5 *ureg.tesla

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
inputs['T_min'] = 3 * ureg.keV
inputs['T_max'] = 30 * ureg.keV
inputs['num_T_points'] = 20
inputs['profile_alpha'] = {}
inputs['profile_alpha']['T'] = 1.75
inputs['T_edge'] = 0.1 * ureg.keV

# Electron Density Inputs
inputs['n_min'] = 0.01e20 * ureg.meter**(-3)
inputs['n_max'] = 7e20 * ureg.meter**(-3)
inputs['num_n_points'] = 20
inputs['profile_alpha']['n'] = 1.1
inputs['n_edge_factor'] = 0.2

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
plot_inputs['contours']['sepOS_density_fraction'] = {'levels': [0.1, 1.0, 5.0, 10.0, 20.0], 'colors':'blue'}
plot_inputs['contours']['bernert_density_fraction'] = {'levels': [0.1, 1.0, 5.0, 10.0, 20.0], 'colors':'grey'}
plot_inputs['contours']['P_LH_fraction'] = {'levels': [0.1, 1.0, 5.0, 10.0, 20.0], 'colors':'gold'}

plot_inputs['plot_ignition'] = True

print('q*={}'.format(popcon.get_q_star(inputs['major_radius']*inputs['inverse_aspect_ratio'],
                                       inputs['major_radius'],
                                       inputs['areal_elongation'],
                                       inputs['magnetic_field_on_axis'],
                                       inputs['plasma_current'])))
output = popcon.get_all_parameters(inputs)

print(output['Q'].max())
print(output['Q'])

print('SepOS Density Fraction')
print(output['sepOS_density_fraction'])

print('Bernert Density Fraction')
print(output['bernert_density_fraction'])

print('P_LH_fraction')
print(output['P_LH_fraction'])

fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=[12, 9])
for a in ax2:
    for i,T in enumerate(output['electron_temperature']):
        a.plot(output['electron_density'], output['energy_confinement_time'][i,:], label='T={:.1f}'.format(T.to(ureg.keV)))
    a.legend()
    a.set_xlabel('Electron Density')
    a.set_ylabel('Energy Confinement Time')
ax2[1].set_yscale('log')
fig2.savefig('energy_confinement_times_ARC.png')

fig, ax = popcon.plot_popcon(output, plot_inputs)


plt.show()

