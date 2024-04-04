import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import copy
import os
from scipy import interpolate
import pickle
import matplotx

import cfspopcon
from cfspopcon.unit_handling import ureg
PROJECT_NAME="BARC"

## Keep R_0 and B_0 constant but vary minor radius (a) by changing the inverse aspect ratio
R_0 = 6.5 * ureg.meter
blanket_thicknesses = [1 * ureg.meter, 2 * ureg.meter]
B_peak = 21 * ureg.tesla

max_greenwald_frac = 0.85
q_level = 15

epsilons = np.arange(0.15, 0.46, 0.05)
# ratios = [1.6]

# modes = ['neg_tri', 'pos_tri_H', 'pos_tri_L']
modes = ['neg_tri', 'pos_tri_H']

mu_0 = 4*np.pi * 1e-7 *ureg.henry / ureg.meter

data_filename = 'max_P_fusion_Q={}_gf={:.0f}%_R0={:.0f}cm.pkl'.format(q_level, 
                                                                  max_greenwald_frac*100,
                                                                  R_0.magnitude*100)
print(data_filename)
if os.path.isfile(data_filename):
    with open(data_filename, 'rb') as file:
        max_P_fusion = pickle.load(file)
else:
    max_P_fusion = np.zeros((len(modes), len(blanket_thicknesses), len(epsilons)))

    for i,mode in enumerate(modes):

        os.listdir(f"{mode}")
        input_parameters, algorithm, points = cfspopcon.read_case(f"{mode}")

        plot_style = cfspopcon.read_plot_style(f"plot_popcon.yaml")

        # Calculate q_star of MANTA to keep it constant
        a_MANTA = input_parameters["major_radius"] * input_parameters['inverse_aspect_ratio']
        q_star_MANTA = 2 * np.pi * a_MANTA**2 * input_parameters["magnetic_field_on_axis"] * input_parameters["areal_elongation"] \
                        / (mu_0 * input_parameters["major_radius"] * input_parameters["plasma_current"])
        print('q* = {}'.format(q_star_MANTA.to_reduced_units()))

        print(algorithm.validate_inputs(input_parameters))

        for j,blanket_thickness in enumerate(blanket_thicknesses):

            for k,epsilon in enumerate(epsilons):

                new_input_parameters = copy.copy(input_parameters)

                new_input_parameters['inverse_aspect_ratio'] = epsilon
                new_input_parameters['major_radius'] = R_0

                a = new_input_parameters["major_radius"] * new_input_parameters['inverse_aspect_ratio']
                R_peak = new_input_parameters["major_radius"] - blanket_thickness - a
                new_input_parameters["magnetic_field_on_axis"] = B_peak * R_peak / new_input_parameters["major_radius"]

                # Calculate the plasma current, keeping q* constant
                I_p = 2 * np.pi * a**2 * new_input_parameters["magnetic_field_on_axis"] * new_input_parameters["areal_elongation"] \
                            / (mu_0 * new_input_parameters["major_radius"] * q_star_MANTA)
                new_input_parameters["plasma_current"] = I_p

                if algorithm.validate_inputs(new_input_parameters):

                    dataset = xr.Dataset(new_input_parameters)

                    algorithm.update_dataset(dataset, in_place=True)

                    greenwald_mask = dataset.greenwald_fraction <= max_greenwald_frac

                    fig1, ax1 = plt.subplots()
                    q_contours = ax1.contour(dataset.dim_average_electron_temp, dataset.dim_average_electron_density[greenwald_mask], dataset.Q[greenwald_mask, :], levels=[q_level])

                    # print(q_contours.collections)
                    # print(q_contours.collections[0].get_paths())

                    try:
                        q_path = q_contours.collections[0].get_paths()[0].vertices

                        # print(q_path)

                        # print(dataset.dim_average_electron_density[greenwald_mask].shape)
                        # print(dataset.dim_average_electron_temp.shape)
                        # print(dataset.P_fusion[greenwald_mask, :].shape)
                        f = interpolate.interp2d(dataset.dim_average_electron_temp, dataset.dim_average_electron_density[greenwald_mask], dataset.P_fusion[greenwald_mask, :])
                        
                        p_fusion = []
                        for T, n in q_path:
                            p_fusion += [f(n,T)[0]]
                        # print(p_fusion)

                        # print(np.max(p_fusion))
                        max_P_fusion[i,j,k] = np.max(p_fusion)
                    except:
                        print(q_contours.collections[0].get_paths())
                        print('No Q={} contours for {}, b={}, epsilon={}'.format(q_level, 
                                                                                   mode, 
                                                                                   blanket_thickness,
                                                                                   epsilon))

                    plt.close(fig1)

                    fig, ax = cfspopcon.plotting.make_plot(dataset, plot_style, points=points, title=PROJECT_NAME, output_dir=None)

                    ax.set_title('{} $\epsilon={:.2f}$, $R_0={:.1f}$ m, $B_0={:.1f}$ T, $I_p={:.1f}$ MA, $b={:.1f}$ m'.format(mode, 
                                                                                                                    epsilon,
                                                                                                                    new_input_parameters['major_radius'].magnitude,
                                                                                                                    new_input_parameters["magnetic_field_on_axis"].magnitude,
                                                                                                                    I_p.to(ureg.amp).magnitude/1e6,
                                                                                                                    blanket_thickness.magnitude))
                    fig.savefig('{}/popcon_epsilon={:.0f}_R0={:.0f}cm_b={:.0f}cm.png'.format(mode, 
                                                                                epsilon*100,
                                                                                new_input_parameters['major_radius'].magnitude*100,
                                                                                blanket_thickness.magnitude * 100))
                    plt.close(fig)
    with open(data_filename, 'wb') as file:
        pickle.dump(max_P_fusion, file)     

colors = ['tab:blue', 'tab:orange', 'tab:green']
fmts = ['.-', '.--']
mode_labels = ['NT', 'PT-H', 'PT-L']
fig2, ax2 = plt.subplots(1,1, figsize=[13,7])

for i,mode in enumerate(modes):
    for j,blanket_thickness in enumerate(blanket_thicknesses):
        ax2.plot(epsilons, max_P_fusion[i,j,:], 
                 fmts[j], 
                 color=colors[i], 
                 label='{}, b={:.1f} m'.format(mode_labels[i], blanket_thickness.magnitude))
ax2.set_xlabel('$\epsilon=a/R_0$', fontsize=12)
ax2.set_ylabel('Maximum Fusion Power [MW]', fontsize=12)
ax2.set_xlim(epsilons[0] - 0.02, epsilons[-1] + 0.02)
ax2.set_title('$Q={}$, $n/n_G = {:.2f}$, $R_0={:.1f}$ m'.format(q_level, max_greenwald_frac, R_0.magnitude), fontsize=12)

legend_labels = []
for blanket_thickness in blanket_thicknesses:
    legend_labels += ['Blanket Thickness = {:.1f} m'.format(blanket_thickness.magnitude)]
ax2.legend(legend_labels)
matplotx.line_labels(ax=ax2)
ax2.grid(alpha=0.3)
ax2.spines.top.set_visible(False)
ax2.spines.right.set_visible(False)
fig2.savefig('max_P_fusion_Q={}_gf={:.0f}%_R0={:.0f}cm.png'.format(q_level, 
                                                                   max_greenwald_frac*100,
                                                                   R_0.magnitude * 100))

plt.show()
