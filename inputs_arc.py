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
inputs['confinement']['scaling'] = 'ITER98y2'
# inputs['confinement']['scaling'] = 'ITER89'
inputs['confinement']['iteration_threshold'] = 0.01
inputs['confinement']['H'] = 1.0
inputs['A'] = 2.5


inputs['num_r_points'] = 50

# Electron Temperature Inputs
inputs['T_min'] = 4 * ureg.keV
inputs['T_max'] = 25 * ureg.keV
inputs['num_T_points'] = 8
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
plot_inputs['contours']['P_auxillary'] = {'levels': [10, 100, 1000, 1e4]}
plot_inputs['contours']['P_SOL'] = {'levels': [10, 100, 1000, 1e4]}
plot_inputs['contours']['P_radiation'] = {'levels': [10, 100, 1000, 1e4]}
plot_inputs['contours']['Q'] = {'levels': [0.1, 1, 10, 100]}

output = popcon.get_all_parameters(inputs)

print(output['P_fusion'])

fig, ax = popcon.plot_popcon(output, plot_inputs)

plt.show()

