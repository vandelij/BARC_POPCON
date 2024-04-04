import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import copy
from scipy import interpolate
import matplotx
import os
import pint
ureg = pint.get_application_registry()
import pickle

sys.path.append('/home/cdunn314/barc/popcon/')
import popcon

Rs = [4.0, 4.5, 5.0, 5.5]
epsilons = [0.27, 0.3, 0.33]
He_frac = 1e-2
krypton_fractions = np.arange(0, 7.5e-3, 5e-4)
# Rs = [6.5]
# ratios = [1.6]

max_greenwald_frac = 1.0
q_level = 10

# blanket_thicknesses = [1 * ureg.meter, 2 * ureg.meter]
# blanket_thicknesses = [1 * ureg.meter]
blanket_thickness = 1 * ureg.meter

B_peak = 22 * ureg.tesla

# scalings = ['ITER89', 'ITER97']
scalings = ['ITER89']

mu_0 = 4*np.pi * 1e-7 *ureg.henry / ureg.meter

inputs = {}

inputs['reaction'] = 'DT'

# inputs['inverse_aspect_ratio'] = 1.2/4.2
inputs['areal_elongation'] = 1.8
inputs['q_star'] = 2.5
# inputs['plasma_current'] = 14 * ureg.MA
# inputs['magnetic_field_on_axis'] = 11.5 *ureg.tesla

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
inputs['T_min'] = 3.0 * ureg.keV
inputs['T_max'] = 35 * ureg.keV
inputs['num_T_points'] = 60
inputs['profile_alpha'] = {}
inputs['profile_alpha']['T'] = 1.5
inputs['T_edge'] = 0.1 * ureg.keV

# Electron Density Inputs
inputs['n_min'] = 0.1e20 * ureg.meter**(-3)
inputs['n_max'] = 6e20 * ureg.meter**(-3)
inputs['num_n_points'] = 40
inputs['profile_alpha']['n'] = 1.1
inputs['n_edge_factor'] = 0.47

######## Plot Inputs ##########
plot_inputs = {}
# plot_inputs['filename'] = 'popcon_ARCH.png'
plot_inputs['contours'] = {}
plot_inputs['contours']['P_fusion'] = {'levels': [100, 250, 500, 1000, 1500, 2000, 3000], 'colors':'black'}
plot_inputs['contours']['P_fusion/A'] = {'levels': [1, 5, 10, 20], 
                                         'colors': 'gray',
                                         'linestyles': 'dashed'}
# plot_inputs['contours']['P_ohmic'] = {'levels': [1, 10, 100, 1000, 1e4]}
plot_inputs['contours']['P_auxillary'] = {'levels': [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80], 'colors':'red'}
# plot_inputs['contours']['P_SOL'] = {'levels': [10, 100, 1000, 1e4]}
# plot_inputs['contours']['P_radiation'] = {'levels': [10, 100, 1000, 1e4]}
# plot_inputs['contours']['energy_confinement_time'] = {'levels':[0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 10.0]}
plot_inputs['contours']['Q'] = {'levels': [1, 10, 20, 40, 1000], 'colors':'tab:purple'}
plot_inputs['contours']['peak_greenwald_fraction'] = {'levels': [1.0], 'colors':'green'}
plot_inputs['contours']['P_LH_fraction'] = {'levels': [1.0], 'colors':'blue'}

plot_inputs['plot_ignition'] = True


max_P_fusion = np.zeros((len(scalings), len(epsilons), len(Rs), len(krypton_fractions)))

max_P_fusion2 = np.zeros((len(scalings), len(epsilons), len(Rs), len(krypton_fractions)))
max_P_fusion_flux = np.zeros((len(scalings), len(epsilons), len(Rs), len(krypton_fractions)))
max_P_fusion_Qs = np.zeros((len(scalings), len(epsilons), len(Rs), len(krypton_fractions)))

for i, scaling in enumerate(scalings):

    new_input_parameters = copy.deepcopy(inputs)

    new_input_parameters['confinement']['scaling'] = scaling

    impurity_str = '_fHe={:.0e}'.format(He_frac)


    parent_dir = '{}_qa={:.0f}_k={:.0f}{}'.format(new_input_parameters['confinement']['scaling'].lower(),
                                                    new_input_parameters['q_star']*100,
                                                    new_input_parameters['areal_elongation']*100,
                                                    impurity_str)
    if not os.path.isdir(parent_dir):
        os.mkdir(parent_dir)
    for j,epsilon in enumerate(epsilons):
        epsilon_dir = 'epsilon={:.0f}'.format(epsilon*100)
        epsilon_path = os.path.join(parent_dir, epsilon_dir)
        if not os.path.isdir(epsilon_path):
            os.mkdir(epsilon_path)

        for k,R in enumerate(Rs):
            R_directory = 'R={:.0f}cm'.format(R*100)
            overall_directory = os.path.join(epsilon_path, R_directory)
            if not os.path.isdir(overall_directory):
                os.mkdir(overall_directory)

            for m,krypton_frac in enumerate(krypton_fractions):


                # new_input_parameters["major_radius"] = input_parameters["major_radius"] * ratio
                # new_input_parameters["magnetic_field_on_axis"] = input_parameters["magnetic_field_on_axis"] * ratio
                new_input_parameters["major_radius"] = R * ureg.meter
                new_input_parameters["inverse_aspect_ratio"] = epsilon
                a = new_input_parameters["major_radius"] * new_input_parameters['inverse_aspect_ratio']
                # print(new_input_parameters['major_radius'].units)
                # print(blanket_thickness.units)
                # print(a.units)
                R_peak = new_input_parameters["major_radius"] - blanket_thickness - a
                new_input_parameters["magnetic_field_on_axis"] = B_peak * R_peak / new_input_parameters["major_radius"]

                I_p = 2 * np.pi * a**2 * new_input_parameters["magnetic_field_on_axis"] * new_input_parameters["areal_elongation"] \
                            / (mu_0 * new_input_parameters["major_radius"] * new_input_parameters['q_star'])

                # if I_p > 15 * ureg.MA:
                #     I_p = 15 * ureg.MA
                new_input_parameters["plasma_current"] = I_p

                new_input_parameters['impurities'] = [[36, krypton_frac],
                                                       [2, He_frac]]
                inputs['impurity_label'] = {'Kr':krypton_frac,
                                            'He':He_frac}

                # Calculate I_p given the same q* as MANTA

                d_filename = 'e={:.0f}_fkr={:.0f}_I={:.0f}MA_b={:.0f}cm.pkl'.format( new_input_parameters["inverse_aspect_ratio"]*100, 
                                                                                    krypton_frac*1e6,
                                                                            I_p.to(ureg.amp).magnitude/1e6,
                                                                            blanket_thickness.magnitude*100)
                d_filepath = os.path.join(overall_directory, d_filename)
                print(d_filepath)
                
                if os.path.isfile(d_filepath):
                    with open(d_filepath, 'rb') as file:
                        output = pickle.load(file)
                else:
                    output = popcon.get_all_parameters(new_input_parameters)
                    with open(d_filepath, 'wb') as file:
                        pickle.dump(output, file)

                greenwald_mask = output['peak_greenwald_fraction'] <= max_greenwald_frac
                # print(greenwald_mask.shape)
                # print(output['Q'][:,greenwald_mask].transpose().shape)

                xmesh, ymesh = np.meshgrid(output['electron_temperature'].to(ureg.keV).magnitude, 
                        output['electron_density'][greenwald_mask].to(ureg.meter**(-3)).magnitude)

                fig2, ax2 = plt.subplots()
                q_contours = ax2.contour(xmesh, ymesh, output['Q'][:,greenwald_mask].transpose(), levels=[q_level])

                # print(q_contours.collections)
                # print(q_contours.collections[0].get_paths())

                try:
                    q_path = q_contours.collections[0].get_paths()[0].vertices

                    # print(q_path)

                    # print(dataset.dim_average_electron_density[greenwald_mask].shape)
                    # print(dataset.dim_average_electron_temp.shape)
                    # print(dataset.P_fusion[greenwald_mask, :].shape)
                    f = interpolate.interp2d(output['electron_temperature'], output['electron_density'][greenwald_mask], output['P_fusion'][:, greenwald_mask].transpose())
                    
                    p_fusion = []
                    for T, n in q_path:
                        p_fusion += [f(T,n)[0]]
                    # print(p_fusion)

                    # print(np.max(p_fusion))
                    max_P_fusion[i,j,k,m] = np.max(p_fusion)
                except:
                    print(q_contours.collections[0].get_paths())
                    print('No Q={} contours for {} R_0={} m'.format(q_level, scaling, R))

                plt.close(fig2)

                p_fusion_2 = output['P_fusion'][:, greenwald_mask]

                greenwald_mask_arr = np.array([list(greenwald_mask)]*len(output['electron_temperature']))

                Q_mask = np.logical_or(output['Q'] >= q_level, output['Q']<0)

                # ignition_mask = output['ignition_fraction'] < 1.0

                L_mode_mask = output['P_LH_fraction'] <= 1.0

                mask_2 = np.logical_and(greenwald_mask_arr, Q_mask)

                # mask_2 = np.logical_and(mask_2, ignition_mask)
                mask_2 = np.logical_and(mask_2, L_mode_mask)

                if mask_2.any():

                    ind = output['P_fusion'][mask_2].argmax()
                    max_p_f = output['P_fusion'][mask_2][ind]
                    print('R={:.1f}, b={:.1f}, f_Kr={:.0e}'.format(R, blanket_thickness, krypton_frac))
                    print(max_p_f)
                    max_p_f_Q = output['Q'][mask_2][ind]
                    print(max_p_f_Q)

                    max_P_fusion2[i,j,k,m] = max_p_f.magnitude
                    max_P_fusion_Qs[i,j,k,m] = max_p_f_Q
                    max_P_fusion_flux[i,j,k,m] = output['P_fusion/A'][mask_2][ind].magnitude



                

                # print(dataset['P_fusion'])



                # print(dataset['PRD'])
                plot_inputs['title'] = '$R_0={:.1f}$ m, $f_Kr={:.0f}appm$, $B_0={:.1f}$ T, $I_p={:.1f}$ MA, $b={:.1f}$ m'.format( R, 
                                                                                                    krypton_frac*1e6, 
                                                                                                    new_input_parameters["magnetic_field_on_axis"].magnitude,
                                                                                                    I_p.to(ureg.amp).magnitude/1e6,
                                                                                                    blanket_thickness.magnitude)
                plot_filename = 'popcon_fKr={:.0f}appm_b={:.0f}cm.png'.format(krypton_frac*1e6,
                                                                        blanket_thickness.magnitude*100)
                plot_inputs['filename'] = os.path.join(overall_directory, plot_filename)

                fig1, ax1 = popcon.plot_popcon(output, plot_inputs)

                plt.close(fig1)

# with open(data_filename, 'wb') as file:
#     pickle.dump(max_P_fusion, file)

colors = ['tab:blue', 'tab:orange', 'tab:green']
fmts = ['.-', '.--']
mode_labels = ['NT', 'PT-H', 'PT-L']

for i,scaling in enumerate(scalings):
    for j,epsilon in enumerate(epsilons):
        fig_e, ax_e = plt.subplots(1,1, figsize=[13,7])
        fig_f, ax_f = plt.subplots(1,1, figsize=[13,7])

        # p_contours = ax2.contourf(xmesh, ymesh, max_P_fusion2[i,j,:,:].transpose(), 
        #                           levels=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500],
        #                           cmap='plasma')
        # cbar = plt.colorbar(p_contours, levels=p_contours.levels)
        # cbar.set_label('Maximum P_fusion [MW]')

        for k,R in enumerate(Rs):
            ax_e.plot(krypton_fractions, max_P_fusion2[i,j,k,:], label='R={:.1f}m'.format(R))
            ax_f.plot(krypton_fractions, max_P_fusion_flux[i,j,k,:], label='R={:.1f}m'.format(R))
            for m,krypton_frac in enumerate(krypton_fractions):
                if max_P_fusion_Qs[i,j,k,m]<0:
                    ax_e.text(krypton_frac, max_P_fusion2[i,j,k,m]*1.01, 'Ignited')
                    ax_f.text(krypton_frac, max_P_fusion_flux[i,j,k,m]*1.01, 'Ignited')
                else:
                    ax_e.text(krypton_frac, max_P_fusion2[i,j,k,m]*1.01, 'Q={:.0f}'.format(max_P_fusion_Qs[i,j,k,m]))
                    ax_f.text(krypton_frac, max_P_fusion_flux[i,j,k,m]*1.01, 'Q={:.0f}'.format(max_P_fusion_Qs[i,j,k,m]))


        ax_e.set_xlabel('Krypton Fraction', fontsize=12)

        axes = [ax_e, ax_f]
        ax_e.set_ylabel('Maximum P_fusion [MW]', fontsize=12)
        ax_f.set_ylabel('Maximum P_fusion/Surface Area [MW/$m^2$]', fontsize=12)
        for ax in axes:
            ax.legend()
            ax.grid(alpha=0.3)
            ax.spines.top.set_visible(False)
            ax.spines.right.set_visible(False)
            ax.set_title(scaling + ' $f_{G,max}=$' + '{}'.format(max_greenwald_frac) \
                        + ', $n_{He}/n_e=$' + '{}'.format(He_frac) \
                        + ' epsilon={:.2f}'.format(epsilon)) 
        fig_e.savefig(os.path.join(epsilon_path, 'max_P_f_vs_krypton_{}_gf={:.0f}%_fHe={:.0e}.png'.format(scaling, max_greenwald_frac*100, He_frac)))
        fig_f.savefig(os.path.join(epsilon_path, 'max_P_f_flux_vs_krypton_{}_gf={:.0f}%_fHe={:.0e}.png'.format(scaling, max_greenwald_frac*100, He_frac)))
plt.show()
