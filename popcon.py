import numpy as np
from scipy import integrate
import matplotlib
from matplotlib import pyplot as plt
import pint
ureg = pint.get_application_registry()

def get_fusion_reactivity(T_e, reaction='DT'):
    # Coefficients from Table 1.3 of
    # Atzeni, Stefano, and JÜrgen Meyer-Ter-Vehn, ‘Nuclear fusion reactions’,
    # The Physics of Inertial Fusion: BeamPlasma Interaction, Hydrodynamics,
    # Hot Dense Matter, International Series of Monographs on Physics
    # (Oxford, 2004; online edn, Oxford Academic, 1 Jan. 2008),
    # https://doi.org/10.1093/acprof:oso/9780198562641.003.0001,
    # accessed 15 Feb. 2023.

    C = {}
    C['DT'] = [6.6610, 643.41e-16, 15.136e-3,
        75.189e-3, 4.6064e-3, 13.500e-3, -0.10675e-3, 0.01366e-3]
    
    T = T_e.to(ureg.keV)
    T = T.magnitude
    
    zi = 1 - (C[reaction][2] * T + C[reaction][4] * (T**2) + C[reaction][6] * (T**3)) \
            /(1 + C[reaction][3] * T + C[reaction][5] * (T**2) + C[reaction][7] * (T**3))
    xi = C[reaction][0]/(T**(1/3))

    reactivity = C[reaction][1] * (zi**(-5/6)) * (xi**2) * np.exp(-3*(zi**(1/3)) * xi) * (ureg.cm)**3 / ureg.second

    return reactivity

def get_s_fusion(n_e, T_e, impurities=None, reaction='DT'):

    """ Calculates fusion power density for a given fusion reaction
        n_e: float or iterable(float) = electron density in m^-3
        T_e: float or iterable(float) = electron temperature in keV 
        impurities: None or array of shape (n,2) for n impurities with a format [Z_i, f_i],
                    where Z_i is the atomic number of the impurity, and f_i = n_impurity/n_e
        reaction: str (default='DT') = type of fusion reaction
    """

    # Energy of fusion reaction
    E_f = {}
    E_f['DT'] = 17.6 * ureg.MeV

    # Calculate reactivity

    reactivity = get_fusion_reactivity(T_e, reaction=reaction)


    ### Get dilution of n_e from impurities factor
    if impurities:
        impurity_factor = 1
        for Z_i, f_i in impurities:
            impurity_factor -= f_i * Z_i
    else:
        impurity_factor = 1
    
    # Calculate fusion power density
    s_fusion = E_f[reaction].to(ureg.joule) * 1/4 * n_e.to((ureg.m)**(-3))**2 * impurity_factor**2 * reactivity.to((ureg.m**(3))/ureg.second)

    s_fusion = s_fusion.to(ureg.MW/(ureg.m**3))

    # print(s_fusion.units)

    return s_fusion


def get_z_eff(reaction='DT', impurities=None):

    """ Calculates Z-effective
        impurities: None or array of shape (n,2) for n impurities with a format [Z_i, f_i],
                    where Z_i is the atomic number of the impurity, and f_i = n_impurity/n_e
        reaction: str (default='DT') = type of fusion reaction
    """

    if reaction=='DD' or reaction=='DT':
        z_eff = 1
        if impurities:
            for Z_i, f_i in impurities:
                z_eff -= f_i * Z_i
                z_eff += f_i * Z_i**2
    else:
        raise ValueError('{} reaction not implemented yet.'.format(reaction))
    
    return z_eff


def get_s_bremsstrahlung(n_e, T_e, reaction='DT', impurities=None):
    """ Calculates bremsstrahlung power density for a given fusion reaction
        n_e: float or iterable(float) = electron density in m^-3
        T_e: float or iterable(float) = electron temperature in keV 
        impurities: None or array of shape (n,2) for n impurities with a format [Z_i, f_i],
                    where Z_i is the atomic number of the impurity, and f_i = n_impurity/n_e
        reaction: str (default='DT') = type of fusion reaction
    """
    z_eff = get_z_eff(reaction=reaction, impurities=impurities)

    s_brem = (5.35e-37 * (z_eff * (n_e.to(ureg.meter**(-3)))**2 * (T_e.to(ureg.keV))**(1/2)).magnitude) * (ureg.watt * (ureg.meter**(-3)))
    s_brem = s_brem.to(ureg.MW / ureg.m**3)
    return s_brem


def get_ln_lambda(T_e, n_e, reaction='DT'):

    # constants
    epsilon_0 = 8.854188e-12 * ureg.coulomb**2 / (ureg.newton * (ureg.meter)**2)
    e = 1.60217663e-19 * ureg.coulomb

    if reaction=='DT' or reaction=='DD':
        # Taken from Equation 9.35 of Freidberg's Plasma Physics and Fusion Energy, 2007
        # ISBN: 978-0-521-73317-5

        Lambda = 12 * np.pi * epsilon_0**(3/2) * (T_e.to(ureg.joule))**(3/2) / \
                ((n_e.to(ureg.meter**(-3)))**(1/2) * e**3)
        ln_lambda = (np.log(Lambda)).magnitude
    return ln_lambda


def get_spitzer_resistivity(T_e, n_e, impurities=None, reaction='DT'):
    ''' Spitzer resisitivity of unmagnitized (or resistivity parallel to B)
        from the MFE Plasma Forulary Section 5.6.1'''
    
    ln_lambda = get_ln_lambda(T_e, n_e, reaction=reaction)
    z_eff = get_z_eff(reaction=reaction, impurities=impurities)

    spitzer_resistivity = ((1.65e-9 * z_eff * ln_lambda) / (T_e.to(ureg.keV))**(3/2)).magnitude
    spitzer_resistivity = spitzer_resistivity * ureg.ohm * ureg.meter

    return spitzer_resistivity
    
def get_p_ohmic_classical(T_e, n_e, plasma_current, major_radius, minor_radius, areal_elongation,
                          impurities=None, reaction='DT'):

    spitzer_resistivity = get_spitzer_resistivity(T_e, n_e, impurities=impurities, reaction=reaction)

    resistance = spitzer_resistivity * (2 * major_radius / (minor_radius**2 * areal_elongation))
    p_ohmic = resistance * plasma_current**2
    p_ohmic = p_ohmic.to(ureg.MW)
    return p_ohmic


def get_p_ohmic_neoclassical(plasma_current, T_e, epsilon, major_radius, areal_elongation):
    """ Calculated using Equation 15.10 of 
    Freidberg's Plasma Physics and Fusion Energy, 2007, ISBN: 978-0-521-73317-5"""
    T_k = T_e.to(ureg.keV)
    R_0 = major_radius.to(ureg.m)
    I_M = plasma_current.to(ureg.MA)
    minor_radius = major_radius * epsilon

    p_ohmic = ((5.6e-2) / (1 - 1.31 * epsilon**(1/2) + 0.46 * epsilon)) \
            * (R_0 * I_M**2) / (minor_radius**2 * areal_elongation * T_k**(3/2))
    p_ohmic = (p_ohmic.magnitude) * ureg.MW

    return p_ohmic

def get_areal_integral(f, n_es, T_es, rs, areal_elongation,
                      **kwargs):

    units = f(n_es[0], T_es[0], **kwargs).units
    rs = rs.to(ureg.meter)
    r_integration = integrate.trapezoid(f(n_es, T_es, **kwargs)*rs, 
                                       x=rs)
    # print(r_integration.units)
    # r_integration *= units * ureg.meter**2
    # print('r_integration: {}'.format(r_integration))
    areal_integral = r_integration * 2 * np.pi * areal_elongation

    return areal_integral


def get_p_fusion(n_es, T_es, rs, areal_elongation, major_radius, reaction='DT', impurities=None):

    linear_fusion_power = get_areal_integral(get_s_fusion,
                                             n_es,
                                             T_es,
                                             rs,
                                             areal_elongation,
                                             reaction=reaction,
                                             impurities=impurities)
    
    # print(linear_fusion_power.units)
    p_fusion = linear_fusion_power * 2 * np.pi * major_radius

    # print(p_fusion.units)
    
    return p_fusion


def get_p_bremmstrahlung(n_es, T_es, rs, areal_elongation, major_radius, reaction='DT', impurities=None):

    linear_brem_power = get_areal_integral(get_s_bremsstrahlung,
                                             n_es,
                                             T_es,
                                             rs,
                                             areal_elongation,
                                             reaction=reaction,
                                             impurities=impurities)
    p_bremmstrahlung = linear_brem_power * 2 * np.pi * major_radius
    
    return p_bremmstrahlung


def get_p_loss(n_es, T_es,
              energy_confinement_time=1.0*ureg.second,
              reaction='DT', impurities=None):
    # Calculate total plasma pressure
    electron_pressures = n_es * T_es
    if reaction=='DT'  or reaction=='DD':
        fusion_reactant_ion_pressures = n_es * T_es
    
    impurity_ion_pressures = np.zeros(np.shape(n_es))
    if impurities:
        for Z_i, f_i in impurities:
            impurity_ion_pressures += f_i * electron_pressures
    total_pressures = electron_pressures + fusion_reactant_ion_pressures + impurity_ion_pressures

    p_loss = 3/2 * total_pressures / energy_confinement_time
    # print(p_loss.units)

    return p_loss


def get_p_sol(n_es, T_es, rs, areal_elongation, minor_radius,
              major_radius, energy_confinement_time, 
              method='total', p_radiation=None,
              reaction='DT', impurities=None):

    # Calculate volumetric average power loss (dU/dt) of plasma

    p_total_loss_linear = get_areal_integral(get_p_loss, n_es, T_es, rs, areal_elongation,
                               energy_confinement_time=energy_confinement_time,
                               reaction=reaction, impurities=impurities)
    p_total_loss = p_total_loss_linear * 2 * np.pi * major_radius 
    # print('p_total_loss:{}'.format(p_total_loss))

    if 'total' in method.lower():
        p_sol = p_total_loss
    elif 'partial' in method.lower():

        p_sol = p_total_loss - p_radiation
    else:
        raise ValueError(f'The method={method} is not implemented in get_p_sol.')

    return p_sol


def get_parabolic_profile(volumetric_average, rs, minor_radius, extrapolation_frac=0.1,
                          alpha=1.5):
    """ Computes 1-D parabolic profile based on rho=r/a, the volumetric
    average of the profile, and a specificed alpha (>1)"""
    a = minor_radius * (1 + extrapolation_frac)
    y_0 = volumetric_average * (alpha + 1)
    ys = y_0 * (1 - (rs/a)**2)**(alpha)
    return ys



def return_func(parameter):
    return parameter


def get_p_auxillary(p_fusion, p_radiation, p_ohmic, p_sol, alpha_fraction=0.2013):
    
    p_auxillary = p_sol + p_radiation - p_ohmic -  alpha_fraction * p_fusion

    return p_auxillary


def get_Q_scientific(p_fusion, p_auxillary, p_ohmic):
    Q = p_fusion / (p_auxillary + p_ohmic)
    return Q


def get_energy_confinement_time(method='ITER98y2', p_external=None, plasma_current=None,
                                major_radius=None, minor_radius=None, kappa=None,
                                density=None, magnetic_field_on_axis=None, H=1.0, A=2.5):
    if method.upper()=='ITER98Y2':
        # H-Mode Scaling from ITER98y2
        tau_E = H * 0.145 * plasma_current.to(ureg.MA).magnitude**0.93 \
                          * major_radius.to(ureg.meter).magnitude**1.39 \
                          * minor_radius.to(ureg.meter).magnitude**0.58 \
                          * kappa**0.78 \
                          * (density.to(ureg.meter**(-3)).magnitude/1e20)**0.41 \
                          * magnetic_field_on_axis.to(ureg.tesla).magnitude**0.15 \
                          * A**0.19 \
                          * p_external.to(ureg.MW).magnitude**(-0.69)
    elif method.upper()=='ITER89':
        # L-Mode Scaling from ITER89
        tau_E = H * 0.048 * plasma_current.to(ureg.MA).magnitude**0.85 \
                          * major_radius.to(ureg.meter).magnitude**1.2 \
                          * minor_radius.to(ureg.meter).magnitude**0.3 \
                          * kappa**0.5 \
                          * (density.to(ureg.meter**(-3)).magnitude/1e20)**0.1 \
                          * magnetic_field_on_axis.to(ureg.tesla).magnitude**0.2 \
                          * A**0.5 \
                          * p_external.to(ureg.MW).magnitude**(-0.5)
    elif method.upper()=='ITER97':
        # L-Mode Scaling from ITER97
        epsilon = minor_radius / major_radius
        tau_E = H * 0.023 * plasma_current.to(ureg.MA).magnitude**0.96 \
                          * major_radius.to(ureg.meter).magnitude**1.83 \
                          * kappa**0.64 \
                          * (density.to(ureg.meter**(-3)).magnitude/1e19)**0.40 \
                          * magnetic_field_on_axis.to(ureg.tesla).magnitude**0.03 \
                          * A**0.20 \
                          * epsilon**(-0.06) \
                          * p_external.to(ureg.MW).magnitude**(-0.73)
    # Add units
    tau_E = tau_E * ureg.second

    return tau_E


def get_greenwald_density(plasma_current, minor_radius):
    n_G = plasma_current.to(ureg.MA) / (np.pi * (minor_radius.to(ureg.meter))**2)
    n_G = n_G.magnitude * 1e20 * (ureg.meter**(-3))
    return n_G


def get_all_parameters(inputs):
    """ inputs: dict"""
    inputs['minor_radius'] = inputs['major_radius'] * inputs['inverse_aspect_ratio']
    if 'num_r_points' not in inputs.keys():
        inputs['num_r_points'] = 50
    rs = np.linspace(0, inputs['minor_radius'], inputs['num_r_points'])

    volumetric_temperatures = np.linspace(inputs['T_min'].to(ureg.keV), inputs['T_max'].to(ureg.keV), inputs['num_T_points'])
    volumetric_densities = np.linspace(inputs['n_min'].to(ureg.meters**(-3)), inputs['n_max'].to(ureg.meters**(-3)), inputs['num_n_points'])


    output = {'electron_temperature': volumetric_temperatures,
              'electron_density': volumetric_densities,
              'P_fusion': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'P_radiation': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'P_ohmic': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'P_SOL': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'P_auxillary': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'energy_confinement_time': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'Q': np.zeros((inputs['num_T_points'], inputs['num_n_points']))}
    
    output['greenwald_density'] = get_greenwald_density(inputs['plasma_current'], inputs['minor_radius'])
    output['greenwald_fraction'] = volumetric_densities / output['greenwald_density']

    # Get power balance parameters
    for i,T_e in enumerate(volumetric_temperatures):
        # T_es = get_parabolic_profile(T_e, rs, inputs['minor_radius'], alpha=inputs['profile_alpha']['T'])
        T_e_units = T_e.units
        T_es = np.array([T_e.magnitude]*len(rs)) * T_e_units
        for j,n_e in enumerate(volumetric_densities):
            n_e_units = n_e.units
            n_es = np.array([n_e.magnitude]*len(rs)) * n_e_units
            # n_es = get_parabolic_profile(n_e, rs, inputs['minor_radius'], alpha=inputs['profile_alpha']['n'])

            # Calculate powers
            # print(get_p_fusion(n_es, T_es, rs, inputs['areal_elongation'],
            #                                        inputs['major_radius'], reaction=inputs['reaction'],
            #                                        impurities=inputs['impurities']))
            output['P_fusion'][i,j] = get_p_fusion(n_es, T_es, rs, inputs['areal_elongation'],
                                                   inputs['major_radius'], reaction=inputs['reaction'],
                                                   impurities=inputs['impurities']).to(ureg.MW).magnitude
            output['P_radiation'][i,j] = get_p_bremmstrahlung(n_es, T_es, rs, inputs['areal_elongation'],
                                                              inputs['major_radius'], reaction=inputs['reaction'],
                                                              impurities=inputs['impurities']).to(ureg.MW).magnitude
            output['P_ohmic'][i,j] = get_p_ohmic_neoclassical(inputs['plasma_current'], T_e, inputs['inverse_aspect_ratio'],
                                                              inputs['major_radius'], inputs['areal_elongation']).to(ureg.MW).magnitude
            
            ## Iterate to get energy confinement time
            continue_loop = True
            energy_confinement_times = [1.0 * ureg.second]
            iter = 0
            while continue_loop:
                output['P_SOL'][i,j] = get_p_sol(n_es, T_es, rs, inputs['areal_elongation'], 
                                                 inputs['minor_radius'], inputs['major_radius'],
                                                energy_confinement_times[-1], method='total', 
                                                p_radiation=output['P_radiation'][i,j],
                                                reaction=inputs['reaction'], impurities=inputs['impurities']).to(ureg.MW).magnitude
                output['P_auxillary'][i,j] = get_p_auxillary(output['P_fusion'][i,j], output['P_radiation'][i,j],
                                                              output['P_ohmic'][i,j], output['P_SOL'][i,j])
                p_external = output['P_auxillary'][i,j]*ureg.MW + output['P_ohmic'][i,j]*ureg.MW

                # Recalculate energy confinement time
                new_confinement_time = get_energy_confinement_time(method=inputs['confinement']['scaling'], 
                                                                         p_external=p_external, plasma_current=inputs['plasma_current'],
                                                                         major_radius=inputs['major_radius'], minor_radius=inputs['minor_radius'], 
                                                                         kappa=inputs['areal_elongation'], density=n_e,
                                                                         magnetic_field_on_axis=inputs['magnetic_field_on_axis'], 
                                                                         H=inputs['confinement']['H'], A=inputs['A'])
                energy_confinement_times += [(new_confinement_time - energy_confinement_times[-1])*0.2 + energy_confinement_times[-1]]
                # print(energy_confinement_times[-1])
                iter += 1
                if energy_confinement_times[-1] is np.NaN:
                    print(energy_confinement_times)
                    raise Exception('energy confinement time is NaN')
                if (energy_confinement_times[-1] - energy_confinement_times[-2])/energy_confinement_times[-2] < inputs['confinement']['iteration_threshold']:
                    output['energy_confinement_time'][i,j] = energy_confinement_times[-1].to(ureg.second).magnitude
                    print('converged')
                    continue_loop = False
                if iter >=40:
                    print('DID NOT CONVERGE')
                    continue_loop = False
            # Calculate Q_scientific
            output['Q'][i,j] = get_Q_scientific(output['P_fusion'][i,j], output['P_auxillary'][i,j],
                                                output['P_ohmic'][i,j])
            # print('\n')
            # break
        # break
    output['P_fusion'] *= ureg.MW
    output['P_radiation'] *= ureg.MW
    output['P_SOL'] *= ureg.MW
    output['P_auxillary'] *= ureg.MW
    output['P_ohmic'] *= ureg.MW
    output['energy_confinement_time'] *= ureg.second
    return output
            


def initialize_plot_inputs(plot_inputs, outputs):

    if 'figsize' not in plot_inputs.keys():
        plot_inputs['figsize'] = [10, 8]
    if 'dpi' not in plot_inputs.keys():
        plot_inputs['dpi'] = 150
    if 'filename' not in plot_inputs.keys():
        plot_inputs['filename'] = 'popcon.png'

    if 'contours' not in plot_inputs.keys():
        plot_inputs['contours'] = {'P_fusion':{}, 'P_radiation':{}, 'P_ohmic':{}, 'P_SOL':{}, 
                                    'P_auxillary':{}, 'energy_confinement_time':{}, 'Q':{}}
        
    colors = ['black', 'tab:blue', 'tab:orange', 'tab:green','tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    color_index = 0
    for contour in plot_inputs['contours'].keys():
        keys = list(plot_inputs['contours'][contour].keys())
        if 'colors' not in keys:
            plot_inputs['contours'][contour]['colors'] = colors[color_index]
            color_index += 1
        if 'linewidth' not in keys:
            plot_inputs['contours'][contour]['linewidth'] = 1.0
        if 'vmin' not in keys:
            if len(pint.UnitRegistry.Quantity(outputs[contour].min()).dimensionality) > 0:
                plot_inputs['contours'][contour]['vmin'] = outputs[contour].min().magnitude
            else:
                plot_inputs['contours'][contour]['vmin'] = outputs[contour].min()
        if 'vmax' not in keys:
            if len(pint.UnitRegistry.Quantity(outputs[contour].max()).dimensionality) > 0:
                plot_inputs['contours'][contour]['vmax'] = outputs[contour].max().magnitude
            else:
                plot_inputs['contours'][contour]['vmax'] = outputs[contour].max()
        if 'levels' not in keys:
            plot_inputs['contours'][contour]['levels'] = 4
        if 'alpha' not in keys:
            plot_inputs['contours'][contour]['alpha'] = 1.0
        if 'label_fmt' not in keys:
            plot_inputs['contours'][contour]['label_fmt'] = '%.1f'
    
    return plot_inputs

               

    
def plot_popcon(outputs, plot_inputs, figsize=[10,8], dpi=150, savename='popcon.png'):
    """ contours: list of outputs key names that will be plotted"""

    plot_inputs = initialize_plot_inputs(plot_inputs, outputs)
    # print(plot_inputs)

    xmesh, ymesh = np.meshgrid(outputs['electron_temperature'].to(ureg.keV).magnitude, 
                               outputs['electron_density'].to(ureg.meter**(-3)).magnitude)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)

    plot_colors = []

    for contour in plot_inputs['contours'].keys():
        if contour.lower()=='greenwald_fraction':
            parameter = np.zeros(outputs['P_fusion'].shape)
            for j in range(len(outputs['electron_density'])):
                parameter[:,j] = outputs['greenwald_fraction'][j]
        else:
            parameter = outputs[contour]
        if len(pint.UnitRegistry.Quantity(parameter).dimensionality) > 0:
            parameter = parameter.magnitude

        # if contour=='P_fusion':
        #     cntr = ax.contourf(xmesh, ymesh, parameter.transpose(), cmap='plasma', levels=10,
        #                        norm=matplotlib.colors.LogNorm())
        #     cbar = plt.colorbar(cntr)
        #     cbar.set_label(label=contour)


        conlines = ax.contour(xmesh, ymesh, parameter.transpose(), 
                          levels=plot_inputs['contours'][contour]['levels'],
                          colors=plot_inputs['contours'][contour]['colors'],
                          alpha=plot_inputs['contours'][contour]['alpha'])
        ax.clabel(conlines, conlines.levels, inline=True, 
                  fmt=plot_inputs['contours'][contour]['label_fmt'])
        plot_colors += [plot_inputs['contours'][contour]['colors']]

    for i,contour in enumerate(plot_inputs['contours'].keys()):
        ax.plot([-2, -1], [-2, -1], color=plot_colors[i], label=contour)
        ax.set_xlim(outputs['electron_temperature'][0].magnitude, outputs['electron_temperature'][-1].magnitude)
        ax.set_ylim(outputs['electron_density'][0].magnitude, outputs['electron_density'][-1].magnitude)
        ax.legend()
    
    fig.savefig(plot_inputs['filename'])

    return fig, ax

        
        

        


