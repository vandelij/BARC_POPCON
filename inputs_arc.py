import popcon
import pint
ureg = pint.get_application_registry()
from matplotlib import pyplot as plt

inputs = {}

inputs['reaction'] = 'DT'
inputs['impurities'] = None

inputs['major_radius'] = 3.3 * ureg.meter
inputs['inverse_aspect_ratio'] = 1.13/3.3
inputs['areal_elongation'] = 1.84
inputs['plasma_current'] = 7.8 * ureg.MA
inputs['magnetic_field_on_axis'] = 9.2 *ureg.tesla

inputs['confinement'] = {}
# Scaling options are 'ITER98y2', 'ITER97', or 'ITER89'
inputs['confinement']['scaling'] = 'ITER98y2'
inputs['confinement']['H'] = 1.0
inputs['A'] = 2.5

# Method for calculating P_SOL
inputs['P_SOL_method'] = 'total'

inputs['num_r_points'] = 50

# Electron Temperature Inputs
inputs['T_min'] = 1 * ureg.keV
inputs['T_max'] = 25 * ureg.keV
inputs['num_T_points'] = 12
inputs['profile_alpha'] = {}
inputs['profile_alpha']['T'] = 1.5

# Electron Density Inputs
inputs['n_min'] = 1e19 * ureg.meter**(-3)
inputs['n_max'] = 5e20 * ureg.meter**(-3)
inputs['num_n_points'] = 10
inputs['profile_alpha']['n'] = 1.5

######## Plot Inputs ##########
plot_inputs = {}
plot_inputs['filename'] = 'popcon_test1.png'
plot_inputs['contours'] = {}
plot_inputs['contours']['P_fusion'] = {'levels': [1, 10, 100, 1000, 1e4]}
# plot_inputs['contours']['P_ohmic'] = {'levels': [1, 10, 100, 1000, 1e4]}
plot_inputs['contours']['P_auxillary'] = {'levels': [1, 10, 20, 30, 40, 50, 100, 1000, 1e4]}
# plot_inputs['contours']['P_SOL'] = {'levels': [10, 100, 1000, 1e4]}
# plot_inputs['contours']['P_radiation'] = {'levels': [10, 100, 1000, 1e4]}
# plot_inputs['contours']['energy_confinement_time'] = {'levels':[0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 10.0]}
plot_inputs['contours']['Q'] = {'levels': [0.1, 1, 2, 4, 8, 10, 20, 40, 100]}
plot_inputs['contours']['greenwald_fraction'] = {'levels': [1.0]}

output = popcon.get_all_parameters(inputs)

print(output['Q'])

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

