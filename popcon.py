import numpy as np
import scipy
from scipy import integrate
import matplotlib
from matplotlib import pyplot as plt
import warnings
import pint
ureg = pint.get_application_registry()

def get_vol_integral(f, n_es, T_es, rs, major_radius, areal_elongation, triangularity=0, **kwargs):

    dVs = 2*np.pi * major_radius * areal_elongation * np.pi * (rs[1:]**2 - rs[:-1]**2)

    ## Using plasma volume with triangularity from: 
    ## A. E. Costley, J. Hugill, and P. F. Buxton, 
    ## “On the power and size of tokamak fusion pilot plants and reactors,” 
    ## Nucl. Fusion, vol. 55, no. 3, p. 033001, Mar. 2015, 
    ## doi: 10.1088/0029-5515/55/3/033001.

    # As = major_radius / rs
    # dVs = (2 * np.pi**2 * areal_elongation * (As - triangularity) \
    #        + 16 * np.pi * areal_elongation * triangularity / 3) \
    #        * rs**3

    ds = f(n_es, T_es, **kwargs)
    dfs = (ds[1:] + ds[:-1])/2 * dVs
    vol_integral = dfs.sum()
    
    # for i,r in enumerate(rs[1:]):
    #     dV = 2*np.pi * major_radius * areal_elongation * np.pi * (rs[i]**2 - rs[i-1]**2)
    #     # dV_units = dV.units
    #     # dV = dV.magnitude
    #     # Get average of the function evalutated at the inner and outer rho
    #     ave_power_density = (f(n_es[i], T_es[i], **kwargs) + f(n_es[i-1], T_es[i-1], **kwargs))/2
    #     # ave_power_density_units = ave_power_density.units
    #     # ave_power_density = ave_power_density.magnitude
    #     dfs += [ave_power_density * dV]
    # vol_integral = np.sum(dfs)
    # vol_integral *= ave_power_density_units * dV_units
    return vol_integral


def return_func(parameter1, parameter2):
    return parameter1


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

def get_s_fusion(n_e, T_e, impurities=None, reaction='DT', f_DT=1.0, f_He=0.0):

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
    dilution_factor = 1 - 2*f_He
    if impurities:
        for element in impurities.keys():
            Z_i = impurities[element][0]
            f_i = impurities[element][1]
            dilution_factor -= f_i * Z_i
    
    # Calculate fusion power density
    s_fusion = E_f[reaction].to(ureg.joule) * (f_DT/((1 + f_DT)**2)) * n_e.to((ureg.m)**(-3))**2 * dilution_factor**2 * reactivity.to((ureg.m**(3))/ureg.second)

    s_fusion = s_fusion.to(ureg.MW/(ureg.m**3))

    # print(s_fusion.units)

    return s_fusion


def get_z_eff(reaction='DT', impurities=None, f_He=0.0):

    """ Calculates Z-effective
        impurities: None or array of shape (n,2) for n impurities with a format [Z_i, f_i],
                    where Z_i is the atomic number of the impurity, and f_i = n_impurity/n_e
        reaction: str (default='DT') = type of fusion reaction
    """

    if reaction=='DD' or reaction=='DT':
        z_eff = 1
        z_eff += (f_He * 2**2 - f_He*2)
        if impurities:
            for element in impurities.keys():
                Z_i = impurities[element][0]
                f_i = impurities[element][1]
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

def get_noble_lrad(T_es, impurity):

    T_r = T_es.to(ureg.keV).magnitude
    
    # Check if all temperatures are in the valid range
    z = np.where((T_r < 0.1) | (T_r > 100))
    if len(z[0]) > 0:
        print('Bad T out of range!!!')
        return 0

    lrad  = np.zeros(len(T_r))  # For every temperature, calculate lrad and z_avg
    z_avg = np.zeros(len(T_r))
    
    if 'Kr' in impurity:
        t_range = np.array([[0.1, 0.447], [0.447, 2.364], [2.364, 100]])
        a       = np.array([[-3.4512E+01, -2.1484E+01, -4.4723E+01, -4.0133E+01, -1.3564E+01],
                            [-3.1399E+01, -5.0091E-01, 1.9148E+00, -2.5865E+00, -5.2704E+00],
                            [-2.9954E+01, -6.3683E+00, 6.6831E+00, -2.9674E+00, 4.8356E-01]])
        
        z_range = np.array([[0.1, 0.447], [0.447, 4.117], [4.117, 100]])
        b       = np.array([[7.7040E+01, 3.0638E+02, 5.6890E+02, 4.6320E+02, 1.3630E+02],
                            [2.4728E+01, 1.5186E+00, 1.5744E+01, 6.8446E+01, -1.0279E+02],
                            [2.5368E+01, 2.3443E+01, -2.5703E+01, 1.3215E+01, -2.4682E+00]])
        
    elif 'Ar' in impurity: 
        t_range = np.array([[0.1, 0.6], [0.6, 3], [3, 100]])
        a       = np.array([[-3.2155E+01, 6.5221E+00, 3.0769E+01, 3.9161E+01, 1.5353E+01],
                            [-3.2530E+01, 5.4490E-01, 1.5389E+00, -7.6887E+00, 4.9806E+00],
                            [-3.1853E+01, -1.6674E+00, 6.1339E-01, 1.7480E-01, -8.2260E-02]])
        
        z_range = np.array([[0.1, 0.6], [0.6, 3], [3, 100]])
        b       = np.array([[1.3171E+01, -2.0781E+01, -4.3776E+01, -1.1595E+01, 6.8717E+00],
                            [1.5986E+01, 1.1413E+00, 2.5023E+00, 1.8455E+00, -4.8830E-02],
                            [1.4948E+01, 7.9986E+00, -8.0048E+00, 3.5667E+00, -5.9213E-01]])     
        
    elif 'Xe' in impurity: 
        t_range = np.array([[0.1, 0.5], [0.5, 2.5], [2.5, 10], [10, 100]])
        a       = np.array([[-2.9303E+01, 1.4351E+01, 4.7081E+01, 5.9580E+01, 2.5615E+01],
                            [-3.1113E+01, 5.9339E-01, 1.2808E+00, -1.1628E+01, 1.0748E+01],
                            [-2.5813E+01, -2.7526E+01, 4.8614E+01, -3.6885E+01, 1.0069E+01],
                            [-2.2138E+01, -2.2592E+01, 1.9619E+01, -7.5181E+00, 1.0858E+00]])
        
        z_range = np.array([[0.1, 0.3], [0.3, 1.5], [1.5, 8], [8, 100]])
        b       = np.array([[3.0532E+02, 1.3973E+03, 2.5189E+03, 1.9967E+03, 5.8178E+02],
                            [3.2616E+01, 1.6271E+01, -4.8384E+01, -2.9061E+01, 8.6824E+01],
                            [4.8066E+01, -1.7259E+02, 6.6739E+02, -9.0008E+02, 4.0756E+02],
                            [-5.7527E+01, 2.4056E+02, -1.9931E+02, 7.3261E+01, -1.0019E+01]]) 

    for i in range(0, len(t_range)):                                                    # For each pair of points in t_range
        T_r_indicies = np.where((T_r >= t_range[i, 0]) & (T_r <= t_range[i, 1]))[0]        # Find the range of Ti that are between the t range values
        if len(T_r_indicies) > 0 :                                                       # If there's more than one Ti in the current t range values
            coefficients_wrong_order = a[i,:]
            coefficients_correct_order = coefficients_wrong_order[::-1]
            lrad[T_r_indicies] = np.polyval(coefficients_correct_order, np.log10(T_r[T_r_indicies]))           # Construct polynomial out of the "a" coefficients, and evaluate for the given Ti

            
        Z_indicies  = np.where((T_r >= z_range[i, 0]) & (T_r <= z_range[i, 1]))[0]        # Find the range of Ti that are between the z range values 
        if len(Z_indicies) > 0 :                                                        # If there's more than one Ti in the current z range values
            coefficients_wrong_order = b[i,:]
            coefficients_correct_order = coefficients_wrong_order[::-1]
            z_avg[Z_indicies] = np.polyval(coefficients_correct_order, np.log10(T_r[Z_indicies]))            # Construct polynomial out of the "b" coefficients and evaluate for the given Ti
            
    lrad = 10**lrad

    return lrad, z_avg


def get_p_line_rad(n_es, T_es, impurities, 
                   rs, major_radius, areal_elongation, 
                   reaction='DT', f_DT=1.0):
    s_fusion = get_s_fusion(n_es, T_es, impurities=None, 
                            reaction=reaction, f_DT=f_DT,
                            f_He=0.0)
    s_line_rad = np.zeros(n_es.shape)
    z_dilutes = {}
    for element in impurities.keys():
        try:
            lrad, z_avg = get_noble_lrad(T_es, element)
        except:
            print('T_es: {}'.format(T_es))
            print('element: {}'.format(element))
            print(get_noble_lrad(T_es, element))
        f_imp = impurities[element][1]
        s_line_rad += (n_es.to(ureg.m**(-3)).magnitude)**2 \
              * lrad * f_imp * ureg.W / ureg.m**(3)
        # Calculate numerator of fusion power weighted volume average z
        z_dilution_numerator = z_avg * s_fusion

        # Calculate fusion power weighted average z
        z_dilute_num = get_vol_integral(return_func, z_dilution_numerator, None,
                                    rs, major_radius, areal_elongation)
        z_dilute_den = get_vol_integral(get_s_fusion, n_es, T_es,
                                        rs, major_radius, areal_elongation,
                                        triangularity=0, 
                                        impurities=None, f_He=0.0,
                                        reaction=reaction, f_DT=f_DT)
        z_dilute = z_dilute_num / z_dilute_den
        z_dilutes[element] = z_dilute.magnitude
    
    # Calculate volume averaged line radiation power
    p_line_rad = get_vol_integral(return_func, s_line_rad, None,
                                  rs, major_radius, areal_elongation)
    p_line_rad.to(ureg.MW)
    
    return p_line_rad, z_dilutes
     


def get_ln_lambda(T_e, n_e, reaction='DT'):

    # constants
    epsilon_0 = 8.854188e-12 * ureg.coulomb**2 / (ureg.newton * (ureg.meter)**2)
    e = 1.60217663e-19 * ureg.coulomb

    if reaction=='DT' or reaction=='DD':
        # Taken from Equation 9.35 of Freidberg's Plasma Physics and Fusion Energy, 2007
        # ISBN: 978-0-521-73317-5

        Lambda = 12 * np.pi * epsilon_0**(3/2) * (T_e.to(ureg.joule))**(3/2) / \
                ((n_e.to(ureg.meter**(-3)))**(1/2) * e**3)
        # print(Lambda)
        ln_lambda = (np.log(Lambda)).magnitude
    return ln_lambda


def get_spitzer_conductance(T_es, n_es, rs, 
                             major_radius, minor_radius,
                             areal_elongation,
                             impurities=None, reaction='DT',
                             f_He=0.0):
    ''' Spitzer resisitivity of unmagnitized (or resistivity parallel to B)
        from the MFE Plasma Forulary Section 5.6.1'''
    
    ln_lambda = get_ln_lambda(T_es, n_es, reaction=reaction)
    z_eff = get_z_eff(reaction=reaction, impurities=impurities, f_He=f_He)


    spitzer_conductivities = ((T_es.to(ureg.keV).magnitude)**(3/2)) / (1.65e-9  *  z_eff * ln_lambda)
    conductivity_ave = get_volume_average(rs, spitzer_conductivities, 
                                          major_radius, minor_radius,
                                          areal_elongation) * (1 / (ureg.ohm * ureg.m))
    effective_conductance = conductivity_ave * (areal_elongation * minor_radius**2) / (2 * major_radius)
    effective_conductance.to(ureg.ohm**(-1))
    return effective_conductance


def get_p_ohmic_classical(T_es, n_es, rs, plasma_current, major_radius, minor_radius, areal_elongation,
                          impurities=None, reaction='DT', f_He=0.0):

    conductance = get_spitzer_conductance(T_es, n_es, rs, 
                                            major_radius, minor_radius,
                                            areal_elongation,
                                            impurities=impurities, reaction=reaction,
                                            f_He=f_He)
    resistance = 1 / conductance
    p_ohmic = resistance * plasma_current**2
    p_ohmic = p_ohmic.to(ureg.MW)
    return p_ohmic


def get_p_ohmic_neoclassical(plasma_current, T_e, epsilon, major_radius, areal_elongation):
    """ Calculated using Equation 15.10 of 
    Freidberg's Plasma Physics and Fusion Energy, 2007, ISBN: 978-0-521-73317-5"""
    T_k = T_e.to(ureg.keV).magnitude
    R_0 = major_radius.to(ureg.m).magnitude
    I_M = plasma_current.to(ureg.MA).magnitude
    minor_radius = (major_radius * epsilon).magnitude

    p_ohmic = ((5.6e-2) / (1 - 1.31 * epsilon**(1/2) + 0.46 * epsilon)) \
            * (R_0 * I_M**2) / (minor_radius**2 * areal_elongation * T_k**(3/2))
    p_ohmic = (p_ohmic) * ureg.MW

    return p_ohmic

def get_current_density_profile(plasma_current, rs, minor_radius, inverse_aspect_ratio, areal_elongation):
    # Equation 15.9 in Freidberg's Plasma Physics and Fusion Energy, 2007, ISBN: 978-0-521-73317-5
    j_0 = plasma_current * 3 / (np.pi * minor_radius**2 * areal_elongation * (1 - 1.31 * inverse_aspect_ratio**(1/2) + 0.46 * inverse_aspect_ratio))
    # Equation 15.8 in Freidberg's Plasma Physics and Fusion Energy, 2007, ISBN: 978-0-521-73317-5
    js = j_0 * (1 - (rs/minor_radius)**2)**2 * (1 - inverse_aspect_ratio**(1/2) * (rs/minor_radius)**(1/2))**2
    return js

def get_s_ohmic_neoclassical(js, T_es):
    s_ohmic = js**2 * (3.3e-8 * ureg.ohm * ureg.meter) / (T_es.to(ureg.keV).magnitude)**(3/2)
    return s_ohmic



def get_p_ohmic_neoclassical_2(plasma_current, rs, T_es, minor_radius, inverse_aspect_ratio, areal_elongation):
    js = get_current_density_profile(plasma_current, rs, minor_radius, inverse_aspect_ratio, areal_elongation)
    p_ohmic = get_vol_integral(get_s_ohmic_neoclassical, js, T_es, rs, minor_radius/inverse_aspect_ratio, areal_elongation)
    return p_ohmic.to(ureg.MW)


def get_p_fusion(n_es, T_es, rs, areal_elongation, major_radius, reaction='DT', impurities=None,
                 f_DT=1.0, f_He=0.0, z_dilutes=None):

    # linear_fusion_power = get_areal_integral(get_s_fusion,
    #                                          n_es,
    #                                          T_es,
    #                                          rs,
    #                                          areal_elongation,
    #                                          reaction=reaction,
    #                                          impurities=impurities)
    
    # # print(linear_fusion_power.units)
    # p_fusion = linear_fusion_power * 2 * np.pi * major_radius

    # If line radiation is calculated, use those z_dilution values to 
    # calculate the dilution of the fusion fuel.
    if isinstance(z_dilutes, dict):
        p_fusion = get_vol_integral(get_s_fusion,
                                    n_es,
                                    T_es,
                                    rs,
                                    major_radius,
                                    areal_elongation,
                                    reaction=reaction,
                                    impurities=None,
                                    f_DT=f_DT,
                                    f_He=0.0)
        # print('pure p_fusion = {}'.format(p_fusion))
        # print('z_dilutes = {}'.format(z_dilutes))
        dilution_factor = 1.0 - 2 * f_He
        for element in impurities.keys():
            # print('{}: z_dilute={}, f_Z={}'.format(element, z_dilutes[element], impurities[element][1]))
            dilution_factor -= z_dilutes[element] * impurities[element][1]
        p_fusion *= dilution_factor**2
        # print('dilution_factor = {}'.format(dilution_factor**2))
        
    else:
        p_fusion = get_vol_integral(get_s_fusion,
                                    n_es,
                                    T_es,
                                    rs,
                                    major_radius,
                                    areal_elongation,
                                    reaction=reaction,
                                    impurities=impurities,
                                    f_DT=f_DT,
                                    f_He=f_He)
    # print(p_fusion.units)
    p_fusion = p_fusion.to(ureg.MW)
    
    return p_fusion


def get_ellipse_circumference(minor_radius, areal_elongation):
    # Using Ramanujan approximation
    b = minor_radius * areal_elongation
    circumference = np.pi * (3*(b + minor_radius) - np.sqrt((3*b + minor_radius)*(b + 3*minor_radius)))
    return circumference


def get_torus_surface_area(major_radius, minor_radius, areal_elongation, triangularity=0, g=0):
    """ Calculates surface area of elliptical torus"""
    circumference = get_ellipse_circumference(minor_radius, areal_elongation)
    surface_area = circumference * 2 * np.pi * major_radius
    return surface_area

def get_plasma_surface_area(major_radius, minor_radius, areal_elongation, triangularity, g=0):
    """ Calculates surface area of elongated, triangulated, toroidal plasma.
    g is the plasma-wall gap
    R_c is the radius of the central mechanical structure

    Based on formulas from:
    A. E. Costley, J. Hugill, and P. F. Buxton, 
    “On the power and size of tokamak fusion pilot plants and reactors,” 
    Nucl. Fusion, vol. 55, no. 3, p. 033001, Mar. 2015, 
    doi: 10.1088/0029-5515/55/3/033001.
    """

    A = major_radius / minor_radius
    # Radius of central mechanical structure
    R_c = ((A-1)/A) * major_radius - g
    
# def get_ellipse_circumference2()
def get_P_fusion_per_area(p_fusion, areal_elongation, major_radius, minor_radius):
    surface_area = get_torus_surface_area(major_radius, minor_radius, areal_elongation, triangularity=0)
    return p_fusion/surface_area


def get_p_bremmstrahlung(n_es, T_es, rs, areal_elongation, major_radius, reaction='DT', impurities=None):

    # linear_brem_power = get_areal_integral(get_s_bremsstrahlung,
    #                                          n_es,
    #                                          T_es,
    #                                          rs,
    #                                          areal_elongation,
    #                                          reaction=reaction,
    #                                          impurities=impurities)
    # p_bremmstrahlung = linear_brem_power * 2 * np.pi * major_radius
    
    p_bremmstrahlung = get_vol_integral(get_s_bremsstrahlung,
                                n_es,
                                T_es,
                                rs,
                                major_radius,
                                areal_elongation,
                                reaction=reaction,
                                impurities=impurities)
    p_bremmstrahlung = p_bremmstrahlung.to(ureg.MW)
    return p_bremmstrahlung


def get_total_pressure_factor(reaction='DT', impurities=None, f_He=0.0):
    electron_pressure_factor = 1
    if reaction=='DT' or reaction=='DD':
        fusion_ion_pressure_factor = 1
    
    impurity_ion_pressure_factor = 0
    # Add helium ash to impurity pressure factor, which should not be included in the impurities dictionary
    impurity_ion_pressure_factor += f_He
    fusion_ion_pressure_factor -= 2 * f_He
    if impurities:
        for element in impurities.keys():
            f_i = impurities[element][1]
            Z_i = impurities[element][0]
            impurity_ion_pressure_factor += f_i
            if reaction=='DT' or reaction=='DD':
                fusion_ion_pressure_factor -= f_i * Z_i
    total_pressure_factor = electron_pressure_factor + fusion_ion_pressure_factor + impurity_ion_pressure_factor
    return total_pressure_factor


def get_s_loss(n_es, T_es,
              energy_confinement_time=1.0*ureg.second,
              reaction='DT', impurities=None, f_He=0.0):
    # Calculate total plasma pressure
    electron_pressures = n_es * T_es
    if reaction=='DT'  or reaction=='DD':
        fusion_reactant_ion_pressures = n_es * T_es
    
    impurity_ion_pressures = np.zeros(np.shape(n_es)) * electron_pressures.units

    # Add helium ash to impurity pressures, which should not be included in the impurities dictionary
    impurity_ion_pressures += electron_pressures * f_He
    fusion_reactant_ion_pressures -= f_He * 2 * electron_pressures
    if impurities:
        for element in impurities.keys():
            f_i = impurities[element][1]
            Z_i = impurities[element][0]
            impurity_ion_pressures += f_i * electron_pressures
            if reaction=='DT' or reaction=='DD':
                fusion_reactant_ion_pressures -= f_i * Z_i * electron_pressures
    total_pressures = electron_pressures + fusion_reactant_ion_pressures + impurity_ion_pressures

    p_loss = 3/2 * total_pressures / energy_confinement_time
    # print(f'tau_E = {energy_confinement_time}')
    # print(f'p_loss = {p_loss}')
    p_loss = p_loss.to(ureg.MW / (ureg.meter**3))

    return p_loss


def get_plasma_energy_density(n_es, T_es, reaction='DT', impurities=None, f_He=0.0):
    # Calculate total plasma pressure
    electron_pressures = n_es * T_es
    if reaction=='DT'  or reaction=='DD':
        fusion_reactant_ion_pressures = n_es * T_es
    
    impurity_ion_pressures = np.zeros(np.shape(n_es))
    # Add helium ash to impurity pressures, which should not be included in the impurities dictionary
    impurity_ion_pressures += electron_pressures * f_He
    fusion_reactant_ion_pressures -= f_He * 2 * electron_pressures

    if impurities:
        for element in impurities.keys():
            f_i = impurities[element][1]
            Z_i = impurities[element][0]
            impurity_ion_pressures += f_i * electron_pressures
            if reaction=='DT' or reaction=='DD':
                fusion_reactant_ion_pressures -= f_i * Z_i * electron_pressures
    total_pressures = electron_pressures + fusion_reactant_ion_pressures + impurity_ion_pressures
    plasma_energy_density = 3/2 * total_pressures
    return plasma_energy_density

def get_plasma_energy(n_es, T_es, rs, major_radius, areal_elongation, reaction='DT', impurities=None):
    plasma_energy = get_vol_integral(get_plasma_energy_density, n_es, T_es, rs,
                                     major_radius, areal_elongation, reaction=reaction, impurities=impurities)
    return plasma_energy

def get_p_total_loss(n_es, T_es, rs, major_radius, areal_elongation,
                     energy_confinement_time=1.0*ureg.second,
                     reaction='DT',
                     impurities=None,
                     f_He=0.0):
    p_total_loss = get_vol_integral(get_s_loss,
                                    n_es,
                                    T_es,
                                    rs,
                                    major_radius,
                                    areal_elongation,
                                    energy_confinement_time=energy_confinement_time,
                                    reaction=reaction,
                                    impurities=impurities,
                                    f_He=f_He)
    return p_total_loss


def get_p_sol(n_es, T_es, rs, areal_elongation, minor_radius,
              major_radius, energy_confinement_time, 
              method='total', p_radiation=None,
              reaction='DT', impurities=None,
              f_He=0.0):

    # Calculate volumetric average power loss (dU/dt) of plasma

    # p_total_loss_linear = get_areal_integral(get_p_loss, n_es, T_es, rs, areal_elongation,
    #                            energy_confinement_time=energy_confinement_time,
    #                            reaction=reaction, impurities=impurities)
    # p_total_loss = p_total_loss_linear * 2 * np.pi * major_radius 
    p_total_loss = get_vol_integral(get_s_loss,
                                    n_es,
                                    T_es,
                                    rs,
                                    major_radius,
                                    areal_elongation,
                                    energy_confinement_time=energy_confinement_time,
                                    reaction=reaction,
                                    impurities=impurities,
                                    f_He=f_He)
    # print('p_total_loss:{}'.format(p_total_loss))

    if 'total' in method.lower():
        p_sol = p_total_loss
    elif 'partial' in method.lower():
        p_sol = p_total_loss - p_radiation
    else:
        raise ValueError(f'The method={method} is not implemented in get_p_sol.')

    p_sol = p_sol.to(ureg.MW)
    return p_sol


def get_parabolic_profile(y_0, rs, minor_radius, y_edge, extrapolation_frac=0.1,
                          alpha=1.5):
    """ Computes 1-D parabolic profile based on rho=r/a, the volumetric
    average of the profile, and a specificed alpha (>1)"""
    # a = minor_radius * (1 + extrapolation_frac)
    # y_0 = volumetric_average * (alpha + 1)
    # ys = y_0 * (1 - (rs/a)**2)**(alpha)

    ys = (y_0 - y_edge) * (1 - (rs/minor_radius)**2)**(alpha) + y_edge
    return ys



def get_volume_average(rs, ys, major_radius, minor_radius, areal_elongation):

    numerator = get_vol_integral(return_func, ys, np.ones(ys.shape), rs, major_radius, areal_elongation)
    # Calculate volume
    denominator = 2 * np.pi * major_radius * np.pi * minor_radius**2 * areal_elongation
    y_ave =  numerator / denominator
    return y_ave

def get_p_auxillary(p_fusion, p_radiation, p_ohmic, p_sol, alpha_fraction=0.2013):
    
    p_auxillary = p_sol + p_radiation - p_ohmic -  alpha_fraction * p_fusion

    p_auxillary = p_auxillary.to(ureg.MW)

    return p_auxillary


def get_Q_scientific(p_fusion, p_auxillary, p_ohmic):
    Q = p_fusion / (p_auxillary + p_ohmic)
    # Q = p_fusion / p_auxillary
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

def get_P_aux_from_tau_E(energy_confinement_time, 
                         method='ITER98y2', p_sol_method='total',
                         p_ohmic=None, p_alpha=None, p_radiation=None, 
                         plasma_current=None,
                         major_radius=None, minor_radius=None, kappa=None,
                         density=None, magnetic_field_on_axis=None, H=1.0, A=2.5):
    if method.upper()=='ITER98Y2':
        # H-Mode Scaling from ITER98y2
        p_heating = (H * 0.145 * plasma_current.to(ureg.MA).magnitude**0.93 \
                          * major_radius.to(ureg.meter).magnitude**1.39 \
                          * minor_radius.to(ureg.meter).magnitude**0.58 \
                          * kappa**0.78 \
                          * (density.to(ureg.meter**(-3)).magnitude/1e20)**0.41 \
                          * magnetic_field_on_axis.to(ureg.tesla).magnitude**0.15 \
                          * A**0.19 \
                          * energy_confinement_time.to(ureg.second).magnitude**(-1))**(1/0.69)
    elif method.upper()=='ITER89':
        # L-Mode Scaling from ITER89
        p_heating = (H * 0.048 * plasma_current.to(ureg.MA).magnitude**0.85 \
                          * major_radius.to(ureg.meter).magnitude**1.2 \
                          * minor_radius.to(ureg.meter).magnitude**0.3 \
                          * kappa**0.5 \
                          * (density.to(ureg.meter**(-3)).magnitude/1e20)**0.1 \
                          * magnetic_field_on_axis.to(ureg.tesla).magnitude**0.2 \
                          * A**0.5 \
                          * energy_confinement_time.to(ureg.second).magnitude**(-1))**(1/0.5)
    elif method.upper()=='ITER97':
        # L-Mode Scaling from ITER97
        epsilon = minor_radius / major_radius
        p_heating = (H * 0.023 * plasma_current.to(ureg.MA).magnitude**0.96 \
                          * major_radius.to(ureg.meter).magnitude**1.83 \
                          * kappa**0.64 \
                          * (density.to(ureg.meter**(-3)).magnitude/1e19)**0.40 \
                          * magnetic_field_on_axis.to(ureg.tesla).magnitude**0.03 \
                          * A**0.20 \
                          * epsilon**(-0.06) \
                          * energy_confinement_time.to(ureg.second).magnitude**(-1))**(1/0.73)
    p_heating = p_heating * ureg.MW

    if p_sol_method=='total':
        p_auxillary = p_heating - p_ohmic - p_alpha + p_radiation
    elif p_sol_method=='partial':
        p_auxillary = p_heating - p_ohmic - p_alpha

    return p_auxillary


def p_aux_root_func(energy_confinement_time, 
                    n_es, T_es, rs,
                    p_fusion, p_radiation, p_ohmic, reaction, impurities,
                    p_sol_method, scaling_method,
                    plasma_current,
                    major_radius, minor_radius, kappa,
                    density, magnetic_field_on_axis, H, A, alpha_fraction,
                    f_He):
    
    p_alpha = p_fusion * alpha_fraction
    p_sol = get_p_sol(n_es, T_es, rs, kappa, 
                    minor_radius, major_radius,
                    energy_confinement_time*ureg.second, method=p_sol_method, 
                    p_radiation=p_radiation,
                    reaction=reaction, impurities=impurities, f_He=f_He)
    
    p_aux_1 = get_p_auxillary(p_fusion, p_radiation,
                            p_ohmic, p_sol).to(ureg.MW).magnitude
    # print('p_aux_1 = {}'.format(p_aux_1))
    
    p_aux_2 = get_P_aux_from_tau_E(energy_confinement_time*ureg.second, 
                                   method=scaling_method, p_sol_method=p_sol_method, 
                                p_ohmic=p_ohmic, p_alpha=p_alpha, p_radiation=p_radiation,
                                plasma_current=plasma_current,
                                major_radius=major_radius, minor_radius=minor_radius, kappa=kappa,
                                density=density, magnetic_field_on_axis=magnetic_field_on_axis, 
                                H=H, A=A).to(ureg.MW).magnitude
    
    # print('p_aux_2 = {}'.format(p_aux_2))

    # return p_aux_1 - p_aux_2, p_aux_1, p_aux_2
    return p_aux_1 - p_aux_2

def get_new_upper_bound(times, ys):
    ## Check for sign changes in ys
    mask = np.diff(np.sign(ys)) != 0
    # Ensure first element is false and len(mask) = len(ys)
    mask = np.append(False, mask)

    # Find each time that a sign change occurred
    sign_change_times = times[mask]

    # Use the smallest time where a time change occured,
    # thereby, getting the smallest root for confinement time
    new_upper_bound = sign_change_times[0]

    return new_upper_bound


def get_converged_confinement_time(lower_bound=1e-4, upper_bound=1e3,
                    n_es=None, T_es=None, rs=None,
                    p_fusion=None, p_radiation=None, p_ohmic=None, reaction='DT', impurities=None,
                    p_sol_method='total', scaling_method='ITER98y2',
                    plasma_current=None,
                    major_radius=None, minor_radius=None, kappa=None,
                    density=None, magnetic_field_on_axis=None, H=1.0, A=2.5,
                    alpha_fraction=0.2013,
                    f_He=0.0):
    """ Uses the p_aux_root_func function to find the root of the difference of P_auxillary
    calculated at various energy confinement times using P_loss to calculate P_auxillary, and using
    the confinement time empircal scalings to calculate P_auxillary"""

    # Try to find the root without any changes to the bounds
    try:
        root  = scipy.optimize.toms748(p_aux_root_func,
                                        lower_bound,
                                        upper_bound,
                                        args=(n_es, T_es, rs,
                                        p_fusion, p_radiation, p_ohmic, reaction, impurities,
                                        p_sol_method, scaling_method,
                                        plasma_current,
                                        major_radius, minor_radius, kappa,
                                        density, magnetic_field_on_axis, H, A, alpha_fraction, f_He))
    except:
        # if root finding failed, try to adjust the upper bound as there may be multiple roots, but
        # we want to find the smallest one
        try:
            times = np.linspace(lower_bound, upper_bound, 200)
            ys = np.zeros(times.shape)

            for n,time in enumerate(times):
                ys[n] = p_aux_root_func(time, 
                                n_es, T_es, rs,
                                p_fusion, p_radiation, p_ohmic, reaction, impurities,
                                p_sol_method, scaling_method,
                                plasma_current,
                                major_radius, minor_radius, kappa,
                                density, magnetic_field_on_axis, H, A, alpha_fraction, f_He)
            new_upper_bound = get_new_upper_bound(times, ys)

        # if no roots were found, we may have not have good enough resolution, so try to 
        # find a sign change in y by using a finer step size for the times array
        except:
            try:
                # print('Using 2,000 times to find root')
                min_ind = np.argmin(np.abs(ys))
                # print(times[min_ind])
                new_lower_bound = times[min_ind]*1e-2
                new_upper_bound = times[min_ind]*2
                # times = np.logspace(np.log10(new_lower_bound), np.log10(new_upper_bound), 2000)
                times = np.linspace(new_lower_bound, new_upper_bound, 5000)
                ys = np.zeros(times.shape)
                p_aux_1s = np.zeros(times.shape)
                p_aux_2s = np.zeros(times.shape)

                for n,time in enumerate(times):
                    ys[n] = p_aux_root_func(time, 
                                    n_es, T_es, rs,
                                    p_fusion, p_radiation, p_ohmic, reaction, impurities,
                                    p_sol_method, scaling_method,
                                    plasma_current,
                                    major_radius, minor_radius, kappa,
                                    density, magnetic_field_on_axis, H, A, alpha_fraction, f_He)
                new_upper_bound = get_new_upper_bound(times, ys)
            except:

                show_plot = True

        # Once a new upper bound has been set, try to find the root again
        
        try:
            root  = scipy.optimize.toms748(p_aux_root_func,
                                            lower_bound,
                                            new_upper_bound,
                                            args=(n_es, T_es, rs,
                                            p_fusion, p_radiation, p_ohmic, reaction, impurities,
                                            p_sol_method, scaling_method,
                                            plasma_current,
                                            major_radius, minor_radius, kappa,
                                            density, magnetic_field_on_axis, H, A, alpha_fraction, f_He))
            print('density={:.2e}: energy_confinement_time = {:.3f}'.format(density, root))
        except:
            # If a root cannot be found, then use the time at which the minimum difference 
            # between the two calculated P_auxillaries occurs and call that the energy confinement time.
            # This method is NOT CORRECT, but may estimate the confinement time decently.
            warnings.warn('Root not found! Using minimum difference as confinement time.')
            min_ind = np.argmin(np.abs(ys))
            root = times[min_ind]
            print('density={:.2e}: energy_confinement_time = {:.3f}'.format(density, root))
        # show_plot=True

    # if show_plot:   
    #     fig, ax = plt.subplots(1, 1)
    #     ax.plot(times, ys)
    #     min_ind = np.argmin(np.abs(ys))
    #     ax.plot(times[min_ind], ys[min_ind], '.', ms=8)
    #     ax.set_ylim(-20, 20)
    #     ax.set_xlabel('Confinement Time')
    #     ax.set_ylabel('Y')

    #     # fig2, ax2 = plt.subplots(1,1)
    #     # ax2.plot(times, p_aux_1s, '.-', label='p_aux_1')
    #     # ax2.plot(times, p_aux_2s, '.-', label='p_aux_2')
    #     # ax2.set_yscale('log')
    #     # ax2.set_xscale('log')
    #     # ax2.legend()


    #     plt.show()

    energy_confinement_time = root * ureg.second

    return energy_confinement_time

    
def get_ignition_fraction(p_fusion, p_radiation, p_SOL, alpha_fraction=0.2013):
    ignition_fraction = p_fusion*alpha_fraction / (p_radiation + p_SOL)
    # ignition_fraction = ignition_fraction.magnitude()
    return ignition_fraction


def get_greenwald_density(plasma_current, minor_radius):
    n_G = plasma_current.to(ureg.MA) / (np.pi * (minor_radius.to(ureg.meter))**2)
    n_G = n_G.magnitude * 1e20 * (ureg.meter**(-3))
    return n_G


def get_peak_greenwald(plasma_current, minor_radius, n_edge_factor, alpha_n):
    n_G = get_greenwald_density(plasma_current, minor_radius)
    rs = np.linspace(0, minor_radius, 1000)
    norm_ns = get_parabolic_profile(1.0, rs, minor_radius, n_edge_factor, alpha=alpha_n)
    norm_line_ave = np.mean(norm_ns)
    n_G_peak = n_G / norm_line_ave
    return n_G_peak

def get_n0_from_n_ave(n_ave, n_edge_factor, alpha_n):
    n0 = n_ave * ((alpha_n + 1) / ((alpha_n + 1)*n_edge_factor + (1 - n_edge_factor)))
    return n0

def get_T_ave(T0, T_edge, alpha_T):
    T_ave = T_edge + (T0 - T_edge)/(alpha_T + 1)
    return T_ave


def get_T0_from_T_ave(T_ave, T_edge, alpha_T):
    T0 = (T_ave - T_edge) * (alpha_T + 1) + T_edge
    return T0



def get_n_ave(n_0, n_edge_factor, alpha_n):
    """ Calculates the volume averaged density 
    for a parabolic profile from the peak density"""
    n_ave =  n_0 * (((alpha_n + 1)*n_edge_factor + (1 - n_edge_factor)) / (alpha_n + 1))
    return n_ave


def get_p_LH_transition(n_ave, magnetic_field_on_axis, major_radius, minor_radius):

    p_LH = 1.38 * ureg.MW \
           * (n_ave.to(ureg.meter**(-3)).magnitude / 1e20)**0.77 \
           * (magnetic_field_on_axis.to(ureg.tesla).magnitude)**0.92 \
           * (major_radius.to(ureg.meter).magnitude)**1.23 \
           * (minor_radius.to(ureg.meter).magnitude)**0.76
    return p_LH


def get_p_LH_transition_2(n_e0, n_edge_factor, alpha_n, major_radius, inverse_aspect_ratio, areal_elongation,
                          B_toroidal_0,
                          plasma_current, zeff,
                          F_A=None, gamma=0.5):
    """ Get the L to H mode transition power based on:
    ITPA H-mode Power Threshold Database Working Group presented by T Takizuka
    2004 Plasma Phys. Control. Fusion 46 A227"""
    n_ave = get_n_ave(n_e0, n_edge_factor, alpha_n)
    
    minor_radius = major_radius * inverse_aspect_ratio
    A = 1 / inverse_aspect_ratio

    S = get_torus_surface_area(major_radius, minor_radius, areal_elongation)
    
    # Get toroidal field at outboard mid-plane
    B_tout = B_toroidal_0 * A / (A + 1)

    # Get poloidal field at outboard mid-plane
    mu_0 = 4*np.pi * 1e-7 *ureg.henry / ureg.meter
    B_pout = mu_0 * plasma_current / (get_ellipse_circumference(minor_radius, areal_elongation))

    # Get total outboard field
    B_out = np.sqrt(B_pout**2 + B_tout**2)

    if not F_A:
        f_A = 1 - (2/(1 + A))**(0.5)
        F_A = 0.1 * A / f_A

    P_LH = 0.072 * (B_out.to(ureg.tesla).magnitude)**(0.7) \
                * (n_ave.to(ureg.meter**(-3)).magnitude/(1e20))**(0.7) \
                * (S.to(ureg.meter**2).magnitude)**(0.9) \
                * (zeff/2)**(0.7) \
                * F_A**gamma
    return P_LH


def get_p_LH_transition_3(n_ave, magnetic_field_on_axis, major_radius, minor_radius, areal_elongation,
                          reaction='DT', impurities=None):

    surface_area = (2 * np.pi * major_radius) * (2 * np.pi * minor_radius) * np.sqrt((1 + areal_elongation**2)/2)
    n20 = n_ave.to(ureg.m**(-3)).magnitude / (1e20)
    B_0 = magnetic_field_on_axis.to(ureg.tesla).magnitude
    S = surface_area.to(ureg.m**2).magnitude
    zeff = get_z_eff(reaction=reaction, impurities=impurities)

    # p_LH = 0.0488 * n20**0.717 * B_0**0.803 * S**0.941 * zeff**0.7 *  ureg.MW
    p_LH = 0.0488 * n20**0.717 * B_0**0.803 * S**0.941 *  ureg.MW

    return p_LH






def get_q_star(minor_radius, major_radius, areal_elongation, magnetic_field_on_axis, plasma_current):
    mu_0 = 4*np.pi * 1e-7 *ureg.henry / ureg.meter
    
    q_star = 2 * np.pi * minor_radius**2 * magnetic_field_on_axis * areal_elongation \
           / (mu_0 * major_radius * plasma_current)
    q_star = q_star.to_reduced_units()
    return q_star


def get_SepOS_density_limit(p_SOL, q_cyl, magnetic_field_on_axis, n_edge_factor):
    n_sep_limit = 8.38 * (p_SOL.to(ureg.MW).magnitude)**0.4 \
                       * (q_cyl)**(-1.12) \
                       * (magnetic_field_on_axis.to(ureg.tesla).magnitude)**(0.73)

    n_0_limit = n_sep_limit * ureg.meter**(-3) * 1e19 / n_edge_factor
    return n_0_limit


def get_bernert_density_limit(p_heat, plasma_current, q_95, n_edge_factor):
    # print('q_95 = {}'.format(q_95.to_reduced_units()))
    n_sep_bernert_limit = 0.51 * (p_heat.to(ureg.MW).magnitude)**0.39 \
                       * (q_95.magnitude)**(-0.32) \
                       * (plasma_current.to(ureg.MA).magnitude)**(0.27)
    n_0_bernert_limit = n_sep_bernert_limit * ureg.meter**(-3) * 1e19 / n_edge_factor
    # print('n_0_limit = {}'.format(n_0_limit))
    return n_0_bernert_limit


def get_giacomin_density_limit(inverse_aspect_ratio, minor_radius, major_radius, p_SOL,
                               safety_factor, areal_elongation, magnetic_field_on_axis, alpha=1.0):
    A = 1 / inverse_aspect_ratio
    a = minor_radius.to(ureg.m).magnitude
    R = major_radius.to(ureg.m).magnitude
    try:
        P_SOL = p_SOL.to(ureg.MW).magnitude
    except:
        P_SOL = p_SOL
    q = safety_factor
    kappa = areal_elongation
    B_T = magnetic_field_on_axis.to(ureg.T).magnitude

    n_lim = alpha * A**(1/6) * a**(3/14) * P_SOL**(10/21) * R**(-43/42) * q**(-22/21) \
        * (1 + kappa**2)**(-1/3) * B_T**(2/3)
    
    n_lim *= 1e20 * ureg.m**(-3)
    
    return n_lim

def get_peak_giacomin_density(n_edge_factor, inverse_aspect_ratio, minor_radius, major_radius, p_SOL,
                               safety_factor, areal_elongation, magnetic_field_on_axis, alpha=1.0):
    n_edge = get_giacomin_density_limit(inverse_aspect_ratio, minor_radius, major_radius, p_SOL,
                                       safety_factor, areal_elongation, magnetic_field_on_axis,
                                       alpha=alpha)
    n_peak = n_edge / n_edge_factor
    return n_peak


def get_q_star_eff(major_radius, minor_radius, areal_elongation, plasma_current, magnetic_field_on_axis):
    
    R = major_radius.to(ureg.m).magnitude
    a = minor_radius.to(ureg.m).magnitude
    Ip = plasma_current.to(ureg.MA).magnitude
    B0 = magnetic_field_on_axis.to(ureg.T).magnitude

    q_star_eff = 5 * (1 + areal_elongation**2) * (B0/R) * (a**2/ (2 * Ip))
    return q_star_eff


def initialize_inputs(inputs):
    inputs['minor_radius'] = inputs['major_radius'] * inputs['inverse_aspect_ratio']
    inputs_keys = inputs.keys()
    if 'DT_ratio' not in inputs_keys:
        inputs['DT_ratio'] = 1.0
    if 'f_He' not in inputs_keys:
        inputs['f_He'] = 0.0
    return inputs


def get_all_parameters(inputs):
    """ inputs: dict"""
    
    inputs = initialize_inputs(inputs)

    q_star = get_q_star_eff(inputs['major_radius'], inputs['minor_radius'],
                            inputs['areal_elongation'], inputs['plasma_current'],
                            inputs['magnetic_field_on_axis'])
    
    if 'num_r_points' not in inputs.keys():
        inputs['num_r_points'] = 50
    rs = np.linspace(0, inputs['minor_radius'], inputs['num_r_points'])

    volumetric_temperatures = np.linspace(inputs['T_min'].to(ureg.keV), inputs['T_max'].to(ureg.keV), inputs['num_T_points'])
    volumetric_densities = np.linspace(inputs['n_min'].to(ureg.meters**(-3)), inputs['n_max'].to(ureg.meters**(-3)), inputs['num_n_points'])


    output = {'electron_temperature': volumetric_temperatures,
              'electron_density': volumetric_densities,
              'P_fusion': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'P_fusion/A': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'P_radiation': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'P_ohmic': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'P_SOL': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'P_auxillary': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'P_loss': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'P_LH': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'P_LH_fraction': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'energy_confinement_time': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'Q': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'ignition_fraction': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'sepOS_density_fraction': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'bernert_density_fraction': np.zeros((inputs['num_T_points'], inputs['num_n_points'])),
              'giacomin_ricci_fraction': np.zeros((inputs['num_T_points'], inputs['num_n_points']))}
    
    output['greenwald_density'] = get_greenwald_density(inputs['plasma_current'], inputs['minor_radius'])
    output['greenwald_fraction'] = volumetric_densities / output['greenwald_density']

    output['peak_greenwald_density'] = get_peak_greenwald(inputs['plasma_current'], inputs['minor_radius'],
                                                          inputs['n_edge_factor'], inputs['profile_alpha']['n'])
    output['peak_greenwald_fraction'] = volumetric_densities / output['peak_greenwald_density']

    # Get power balance parameters
    for i,T_e in enumerate(volumetric_temperatures):
        T_es = get_parabolic_profile(T_e, rs, inputs['minor_radius'], 
                                     inputs['T_edge'], alpha=inputs['profile_alpha']['T'])
        # T_e_units = T_e.units
        # T_es = np.array([T_e.magnitude]*len(rs)) * T_e_units
        for j,n_e in enumerate(volumetric_densities):
            # n_e_units = n_e.units
            # n_es = np.array([n_e.magnitude]*len(rs)) * n_e_units
            n_es = get_parabolic_profile(n_e, rs, inputs['minor_radius'], 
                                         n_e*inputs['n_edge_factor'] , alpha=inputs['profile_alpha']['n'])

            # Calculate powers
            # print(get_p_fusion(n_es, T_es, rs, inputs['areal_elongation'],
            #                                        inputs['major_radius'], reaction=inputs['reaction'],
            #                                        impurities=inputs['impurities']))
            if isinstance(inputs['impurities'], dict):
                p_radiation, z_dilutes = get_p_line_rad(n_es, T_es, inputs['impurities'],
                                                            rs, inputs['major_radius'], inputs['areal_elongation'],
                                                            reaction=inputs['reaction'], f_DT=inputs['DT_ratio'])
                # print(z_dilutes)
            else:
                p_radiation = get_p_bremmstrahlung(n_es, T_es, rs, inputs['areal_elongation'],
                                                              inputs['major_radius'], reaction=inputs['reaction'],
                                                              impurities=inputs['impurities'])
                z_dilutes = None
            output['P_radiation'][i,j] = p_radiation.to(ureg.MW).magnitude
            output['P_fusion'][i,j] = get_p_fusion(n_es, T_es, rs, inputs['areal_elongation'],
                                                   inputs['major_radius'], reaction=inputs['reaction'],
                                                   impurities=inputs['impurities'],
                                                   f_DT=inputs['DT_ratio'],
                                                   f_He=inputs['f_He'],
                                                   z_dilutes=z_dilutes).to(ureg.MW).magnitude
            output['P_fusion/A'][i,j] = get_P_fusion_per_area(output['P_fusion'][i,j]*ureg.MW,
                                                              inputs['areal_elongation'],
                                                              inputs['major_radius'],
                                                              inputs['minor_radius']).to(ureg.MW/(ureg.meter**2)).magnitude
            output['P_ohmic'][i,j] = get_p_ohmic_classical(T_es, n_es, rs, inputs['plasma_current'], inputs['major_radius'], inputs['minor_radius'],
                                                            inputs['areal_elongation'], 
                                                            impurities=inputs['impurities'],
                                                            reaction=inputs['reaction'],
                                                            f_He=inputs['f_He']).to(ureg.MW).magnitude
            # output['P_ohmic'][i,j] = get_p_ohmic_neoclassical_2(inputs['plasma_current'], rs, T_es, inputs['minor_radius'],
            #                                                     inputs['inverse_aspect_ratio'], inputs['areal_elongation']).to(ureg.MW).magnitude
            
            ## Iterate to get energy confinement time
            energy_confinement_time_guess = 1.0 * ureg.second

            tau_E = get_converged_confinement_time( 
                    lower_bound=inputs['confinement']['lower_bound'], 
                    upper_bound=inputs['confinement']['upper_bound'],
                    n_es=n_es, T_es=T_es, rs=rs,
                    p_fusion=output['P_fusion'][i,j]*ureg.MW, 
                    p_radiation=output['P_radiation'][i,j]*ureg.MW, 
                    p_ohmic=output['P_ohmic'][i,j]*ureg.MW, 
                    reaction=inputs['reaction'], 
                    impurities=inputs['impurities'],
                    p_sol_method=inputs['P_SOL_method'], 
                    scaling_method=inputs['confinement']['scaling'],
                    plasma_current=inputs['plasma_current'],
                    major_radius=inputs['major_radius'], 
                    minor_radius=inputs['minor_radius'], 
                    kappa=inputs['areal_elongation'],
                    density=n_e, 
                    magnetic_field_on_axis=inputs['magnetic_field_on_axis'], 
                    H=inputs['confinement']['H'], 
                    A=inputs['A'],
                    f_He=inputs['f_He'])
            
            output['energy_confinement_time'][i,j] = tau_E.to(ureg.second).magnitude

            output['P_SOL'][i,j] = get_p_sol(n_es, T_es, rs, inputs['areal_elongation'], 
                                             inputs['minor_radius'], inputs['major_radius'],
                                             tau_E, method=inputs['P_SOL_method'], 
                                             p_radiation=output['P_radiation'][i,j]*ureg.MW,
                                             reaction=inputs['reaction'], impurities=inputs['impurities'],
                                             f_He=inputs['f_He']).to(ureg.MW).magnitude
            output['P_loss'][i,j] = output['P_SOL'][i,j] + output['P_radiation'][i,j]
            
            output['P_auxillary'][i,j] = get_p_auxillary(output['P_fusion'][i,j]*ureg.MW, output['P_radiation'][i,j]*ureg.MW,
                                                         output['P_ohmic'][i,j]*ureg.MW, output['P_SOL'][i,j]*ureg.MW).to(ureg.MW).magnitude

                
            # Calculate Q_scientific
            output['Q'][i,j] = get_Q_scientific(output['P_fusion'][i,j], output['P_auxillary'][i,j],
                                                output['P_ohmic'][i,j])
            # Calculate ignition fraction
            output['ignition_fraction'][i,j] = get_ignition_fraction(output['P_fusion'][i,j], output['P_radiation'][i,j], output['P_SOL'][i,j])

            # Calculate L-mode to H-mode transition power
            # output['P_LH_fraction'][i,j] = (output['P_SOL'][i,j]) \
            #                                 / get_p_LH_transition(n_e, inputs['n_edge_factor'], inputs['profile_alpha']['n'],
            #                                           inputs['magnetic_field_on_axis'], inputs['major_radius'],
            #                                           inputs['minor_radius']).to(ureg.MW).magnitude 
            zeff = get_z_eff(inputs['reaction'], inputs['impurities'])

            # output['P_LH_fraction'][i,j] = (output['P_SOL'][i,j]) \
            #                               / get_p_LH_transition_2(n_e, inputs['n_edge_factor'], inputs['profile_alpha']['n'],
            #                                                       inputs['major_radius'], inputs['inverse_aspect_ratio'],
            #                                                       inputs['areal_elongation'], inputs['magnetic_field_on_axis'],
            #                                                       inputs['plasma_current'], zeff,
            #                                                       F_A=None, gamma=0.5)
            output['P_LH'][i,j] = get_p_LH_transition_3(np.mean(n_es), inputs['magnetic_field_on_axis'],
                                                                    inputs['major_radius'], inputs['minor_radius'],
                                                                    inputs['areal_elongation'],
                                                                    reaction=inputs['reaction'], impurities=inputs['impurities']).to(ureg.MW).magnitude
            output['P_LH_fraction'][i,j] = (output['P_SOL'][i,j]) \
                                            / get_p_LH_transition_3(np.mean(n_es), inputs['magnetic_field_on_axis'],
                                                                    inputs['major_radius'], inputs['minor_radius'],
                                                                    inputs['areal_elongation'],
                                                                    reaction=inputs['reaction'], impurities=inputs['impurities']).to(ureg.MW).magnitude
            output['sepOS_density_fraction'][i,j] = (n_e / get_SepOS_density_limit(output['P_SOL'][i,j] * ureg.MW,
                                                                       q_star,
                                                                       inputs['magnetic_field_on_axis'],
                                                                       inputs['n_edge_factor']).to(ureg.meter**(-3)) \
                                                                       ).magnitude
            # output['bernert_density_fraction'][i,j] = (n_e / get_bernert_density_limit(
            #                                                     (output['P_fusion'][i,j] + output['P_ohmic'][i,j] + output['P_auxillary'][i,j]) * ureg.MW,
            #                                                     inputs['plasma_current'],
            #                                                     q_star,
            #                                                     inputs['n_edge_factor']).to(ureg.meter**(-3)) \
            #                                                     ).magnitude
            output['giacomin_ricci_fraction'][i,j] = (n_e / get_peak_giacomin_density(inputs['n_edge_factor'],
                                                                                      inputs['inverse_aspect_ratio'],
                                                                                      inputs['minor_radius'],
                                                                                      inputs['major_radius'],
                                                                                      output['P_SOL'][i,j]*ureg.MW,
                                                                                      q_star,
                                                                                      inputs['areal_elongation'],
                                                                                      inputs['magnetic_field_on_axis'])).magnitude
            
            
            
            
            # print('\n')
            # break
        # break
    output['P_fusion'] *= ureg.MW
    output['P_fusion/A'] *= ureg.MW / (ureg.meter**2)
    output['P_radiation'] *= ureg.MW
    output['P_SOL'] *= ureg.MW
    output['P_auxillary'] *= ureg.MW
    output['P_ohmic'] *= ureg.MW
    # output['P_LH'] *= ureg.MW
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
    if 'legend_label' not in plot_inputs.keys():
        plot_inputs['legend_label'] = {'P_fusion': '$P_{fusion}$ [MW]',
                                       'P_fusion/A': '$P_{fusion}/A$ [MW/m${}^2$]',
                                       'P_radiation': '$P_{rad}$ [MW]',
                                       'P_ohmic': '$P_{ohmic}$ [MW]',
                                       'P_SOL': '$P_{SOL}$ [MW]',
                                       'P_auxillary': '$P_{aux}$ [MW]',
                                       'P_LH_fraction': '$P_{SOL}/P_{LH}$ [MW]',
                                       'energy_confinement_time': '$\tau_E$ [s]',
                                       'Q': '$Q_s$',
                                       'greenwald_fraction': '$n/n_G$',
                                       'peak_greenwald_fraction': '$n/n_G$',
                                       'sepOS_density_fraction': '$n/n_{SepOS}$',
                                       'bernert_density_fraction': '$n/n_{bernert}$',
                                       'giacomin_ricci_fraction': '$n/n_{GR}$'}
    if 'plot_ignition' not in plot_inputs.keys():
        plot_inputs['plot_ignition'] = False
    if 'title' not in plot_inputs.keys():
        plot_inputs['title'] = ''
        
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
        if 'linestyles' not in keys:
            plot_inputs['contours'][contour]['linestyles'] = 'solid'
    
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
        elif contour.lower()=='peak_greenwald_fraction':
            parameter = np.zeros(outputs['P_fusion'].shape)
            for j in range(len(outputs['electron_density'])):
                parameter[:,j] = outputs['peak_greenwald_fraction'][j]
        else:
            parameter = outputs[contour]
        # Check if parameter has units, and if so, only use the magnitude
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
                          alpha=plot_inputs['contours'][contour]['alpha'],
                          linestyles=plot_inputs['contours'][contour]['linestyles'])
        ax.clabel(conlines, conlines.levels, inline=True, 
                  fmt=plot_inputs['contours'][contour]['label_fmt'], fontsize=14)
        plot_colors += [plot_inputs['contours'][contour]['colors']]

    if plot_inputs['plot_ignition']:
        cntr = ax.contourf(xmesh, ymesh, outputs['ignition_fraction'].transpose(),
                           levels=[1.0, 100.0],
                           cmap='Reds',
                           alpha=0.5)
    for i,contour in enumerate(plot_inputs['contours'].keys()):
        if plot_inputs['contours'][contour]['linestyles'] == 'solid':
            fmt = '-'
        elif plot_inputs['contours'][contour]['linestyles'] == 'dashed':
            fmt = '--'
        ax.plot([-2, -1], [-2, -1], fmt, color=plot_colors[i], label=plot_inputs['legend_label'][contour])
        ax.set_xlim(outputs['electron_temperature'][0].magnitude, outputs['electron_temperature'][-1].magnitude)
        ax.set_xlabel('$T_{e0}$ [keV]', fontsize=20)
        ax.tick_params(axis='x', labelsize=16)

        ax.set_ylim(outputs['electron_density'][0].magnitude, outputs['electron_density'][-1].magnitude)
        ax.set_ylabel('$n_{e0}$ [m${}^{-3}$]', fontsize=20)
        ax.tick_params(axis='y', labelsize=16)
        ax.legend(fontsize=18, loc='lower right')
    ax.set_title(plot_inputs['title'])
    fig.savefig(plot_inputs['filename'])

    return fig, ax

        
        

        


