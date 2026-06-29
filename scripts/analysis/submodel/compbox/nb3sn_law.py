import numpy as np
import math
import json
import os
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_scaling_parameters(filename='scaling_parameters.json', wire_type='RRP MQXF 0.85 mm'):
    """Load scaling parameters from JSON file for specified wire type.

    `filename` is resolved relative to this module's directory so the loader
    works regardless of the caller's cwd (e.g. cross-variant overlays driven
    from sweep_range.py run from the repo root).
    """
    if not os.path.isabs(filename):
        filename = os.path.join(_MODULE_DIR, filename)
    with open(filename, 'r') as f:
        data = json.load(f)
    params = data['wire_types'][wire_type]['parameters']
    
    # Handle eps_l0 parameter - use eps_l0_free_strand if available, otherwise eps_l0
    if 'eps_l0_free_strand' in params:
        eps_l0_value = params['eps_l0_free_strand']['value']
    else:
        eps_l0_value = params['eps_l0']['value']
    
    return {
        'C0': params['C0']['value'],
        'Bc20': params['Bc20']['value'],
        'Tc0': params['Tc0']['value'],
        'C1': params['C1']['value'],
        'eps_l0': eps_l0_value,
        'eta': params['eta']['value'],
        'sigma': params['sigma']['value'],
        'nu': params['nu']['value'],
        'K': params['K']['value'],
        'p': params['p']['value'],
        'q': params['q']['value']
    }

def load_scaling_parameters_with_uncertainty(filename='scaling_parameters.json', wire_type='RRP MQXF 0.85 mm'):
    """Load scaling parameters with uncertainties from JSON file for specified wire type.

    `filename` is resolved relative to this module's directory so the loader
    works regardless of caller cwd. See `load_scaling_parameters` for context."""
    if not os.path.isabs(filename):
        filename = os.path.join(_MODULE_DIR, filename)
    with open(filename, 'r') as f:
        data = json.load(f)
    params = data['wire_types'][wire_type]['parameters']
    
    result = {}
    for key in ['C0', 'Bc20', 'Tc0', 'eps_l0', 'eta']:
        if key == 'eps_l0':
            param_data = params['eps_l0_free_strand']
        else:
            param_data = params[key]
        
        result[key] = {
            'value': param_data['value'],
            'uncertainty': param_data.get('uncertainty', 0)
        }
    
    # Add parameters without uncertainties
    for key in ['C1', 'sigma', 'nu', 'K', 'p', 'q']:
        result[key] = {'value': params[key]['value'], 'uncertainty': 0}
    
    return result

def calculate_critical_current_bounds(eps_1, eps_2, eps_3, eps_12, Bp, T, include_cable_field=False, num_strands=21, cable_scaling=0.5):
    """Calculate critical current with uncertainty bounds.
    
    Returns:
        tuple: (Ic_nominal, Ic_upper, Ic_lower) where upper/lower are based on parameter uncertainties
    """
    params_with_unc = load_scaling_parameters_with_uncertainty()
    
    # Nominal parameters
    nominal_params = {key: val['value'] for key, val in params_with_unc.items()}
    
    # Parameters for maximum Ic (optimistic bounds)
    max_params = nominal_params.copy()
    max_params['C0'] += params_with_unc['C0']['uncertainty'] or 0  # Higher C0 → higher Ic
    max_params['Bc20'] += params_with_unc['Bc20']['uncertainty'] or 0  # Higher Bc20 → higher Ic
    max_params['Tc0'] += params_with_unc['Tc0']['uncertainty'] or 0  # Higher Tc0 → higher Ic
    max_params['eta'] -= params_with_unc['eta']['uncertainty'] or 0  # Lower eta → higher Ic (corrected!)
    max_params['eps_l0'] += params_with_unc['eps_l0']['uncertainty'] or 0  # Less negative eps_l0 → higher Ic
    
    # Parameters for minimum Ic (conservative bounds)
    min_params = nominal_params.copy()
    min_params['C0'] -= params_with_unc['C0']['uncertainty'] or 0  # Lower C0 → lower Ic
    min_params['Bc20'] -= params_with_unc['Bc20']['uncertainty'] or 0  # Lower Bc20 → lower Ic
    min_params['Tc0'] -= params_with_unc['Tc0']['uncertainty'] or 0  # Lower Tc0 → lower Ic
    min_params['eta'] += params_with_unc['eta']['uncertainty'] or 0  # Higher eta → lower Ic (corrected!)
    min_params['eps_l0'] -= params_with_unc['eps_l0']['uncertainty'] or 0  # More negative eps_l0 → lower Ic
    
    # Calculate Ic for all three parameter sets
    Ic_nominal, _, _, _ = calculate_critical_current(eps_1, eps_2, eps_3, eps_12, Bp, T, 
                                                   nominal_params, include_cable_field, num_strands, cable_scaling)
    Ic_upper, _, _, _ = calculate_critical_current(eps_1, eps_2, eps_3, eps_12, Bp, T, 
                                                 max_params, include_cable_field, num_strands, cable_scaling)
    Ic_lower, _, _, _ = calculate_critical_current(eps_1, eps_2, eps_3, eps_12, Bp, T, 
                                                 min_params, include_cable_field, num_strands, cable_scaling)
    
    return Ic_nominal, Ic_upper, Ic_lower

def calculate_critical_current(eps_1, eps_2, eps_3, eps_12, Bp, T, params=None, include_cable_field=False, num_strands=21, cable_scaling=0.5):
    """Calculate critical current using Nb3Sn scaling law.
    
    Args:
        eps_1, eps_2, eps_3, eps_12: Strain components
        Bp: Applied magnetic field (T)
        T: Temperature (K)
        params: Scaling parameters (optional)
        include_cable_field: If True, includes cable self-field contribution (default: False)
        num_strands: Number of strands in cable for cable field calculation (default: 21)
        cable_scaling: Cable self-field scaling factor (default: 0.5)
    """
    if params is None:
        params = load_scaling_parameters()
    
    # Extract parameters
    C0 = params['C0']
    Bc20_max = params['Bc20']
    Tc0_max = params['Tc0']
    C1 = params['C1']
    eps_l0 = params['eps_l0']
    eta = params['eta']
    sigma = params['sigma']
    nu = params['nu']
    K = params['K']
    p = params['p']
    q = params['q']

    eps_T0 = -nu*eps_l0 + K
    
    eps_a = eps_3
    
    # #For axial strain only
    # J_2 = (1/3)*(eps_l0-eps_T0+(1+nu)*eps_a)**2
    # I_1 = (1-2*nu)*(eps_a)+eps_l0+2*eps_T0
    
    #Plane stress No eps_23, eps_31 
    eps_1_tot = eps_1 + eps_T0
    eps_2_tot = eps_2 + eps_T0
    eps_3_tot = eps_a + eps_l0
    
    I_1 = eps_1_tot + eps_2_tot + eps_3_tot
    
    # Calculate I_2 using strain tensor invariants
    # Reference: https://doc.comsol.com/5.5/doc/com.comsol.help.sme/sme_ug_theory.06.09.html
    # Build strain tensor e_t (3x3 matrix)
    # For plane stress with shear only in x-y plane:
    # e_t = [[eps_1_tot, eps_12,      0      ]
    #        [eps_12,     eps_2_tot,  0      ]
    #        [0,          0,          eps_3_tot]]

    
    # Build strain tensor e_t (3x3 matrix) for scalar inputs
    e_t = np.zeros((3, 3), dtype=float)
    e_t[0, 0] = eps_1_tot
    e_t[1, 1] = eps_2_tot
    e_t[2, 2] = eps_3_tot
    e_t[0, 1] = e_t[1, 0] = eps_12  # Shear in x-y plane
    e_tsquared = np.matmul(e_t, e_t)
    # Inline trace computations to avoid function call overhead (fastest for fixed 3x3)
    tr_e_t = e_t[0, 0] + e_t[1, 1] + e_t[2, 2]
    tr_e_tsquared = e_tsquared[0, 0] + e_tsquared[1, 1] + e_tsquared[2, 2]
    I_2 = 0.5 * (tr_e_t**2 - tr_e_tsquared)
    J_2 = (1.0/3.0)*I_1**2 - I_2 # Alternative J2 calculation using tensor trace
        
    
    seps = (np.exp(-C1*((J_2+3)/(J_2+1))*J_2) + np.exp(-C1*((I_1**2+3)/(I_1**2+1))*I_1**2))/2
    
    
    TC0_eps = Tc0_max*(seps**(1/3))
    t = T/TC0_eps
    BC2_Teps = Bc20_max*seps*(1-t**(3/2))
    b = Bp/BC2_Teps
    Ic_bTeps= (C0/Bp)*((seps)**sigma)*((1-t**(3/2))*(1-t**2))**(eta/2)*(b**p)*(1-b)**q
    
    # Iteratively include self-field until Ic change below threshold
    threshold = 1e-3  # A
    max_iter = 100
    radius = 0.85 * (850e-6)/2  # Strand radius in meters (for 0.85 mm diameter)
    mu0 = 4*math.pi*1e-7  # Permeability of free space (H/m)
    
    # Initial critical current calculation (without self-field)
    prev_Ic = Ic_bTeps
    B_strand_self = 0.0
    B_cable_self = 0.0
    
    for _ in range(max_iter):
        # Calculate self-fields from current Ic
        I_encl = prev_Ic
        B_strand_self = 0.85*mu0*I_encl/(2*math.pi*radius)
        
        # Calculate cable self-field if requested
        B_cable_self = 0.0
        if include_cable_field:
            # Approximate cable field contribution (empirical formula)
            # This represents the field from other strands in the cable
            B_cable_self = cable_scaling * I_encl * num_strands / 10000  # Tesla
            
        # Total self-field contribution
        B_total_self = B_strand_self + B_cable_self
            
        # Update Bp for next iteration
        Bp2 = Bp + B_total_self

        # Recalculate critical current with updated field
        b = Bp2/BC2_Teps
        Ic_new = (C0/Bp2)*((seps)**sigma)*((1-t**(3/2))*(1-t**2))**(eta/2)*(b**p)*(1-b)**q
        
        # Check convergence
        if abs(Ic_new - prev_Ic) < threshold:
            Ic_bTeps = Ic_new
            break
            
        prev_Ic = Ic_new
    else:
        # Max iterations reached; accept last value
        Ic_bTeps = prev_Ic
    
    # Ensure final self-field values correspond to final Ic
    I_encl = Ic_bTeps
    B_strand_self = mu0*I_encl/(2*math.pi*radius)
    B_cable_self = 0.0
    if include_cable_field:
        B_cable_self = cable_scaling * I_encl * num_strands / 10000
        
    return Ic_bTeps, seps, B_strand_self, B_cable_self

if __name__ == "__main__":
    # Create strain array from -0.3% to +0.4%
    eps_a_array = np.linspace(-0.3, 0.4, 100)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=4)
    ax.tick_params(which='major', length=7)
    # First set: T = 4.2 K with Bp = 12 T
    T1 = 4.2  # Temperature (K)
    Bp_values_T1 = [10,12,14,16]  # Applied field values for T1
    
    for Bp in Bp_values_T1:
        Ic_array = []
        for eps_a in eps_a_array:
            # For axial strain only: eps_1=0, eps_2=0, eps_3=eps_a, eps_12=0
            Ic, _, _, _ = calculate_critical_current(0, 0, eps_a, 0, Bp, T1)
            Ic_array.append(Ic)
        
        # Convert to numpy array for easier handling
        Ic_array = np.array(Ic_array)
        
        # Plot this field value
        plt.plot(eps_a_array, Ic_array, linewidth=2, label=f'Bp = {Bp:.0f} T, T = {T1} K')
    
    # Second set: T = 8 K with Bp = 10, 12, 14 T
    T2 = []#8.0  # Temperature (K)
    Bp_values_T2 =[]#[10, 12, 14]  # Applied field values for T2
    
    for Bp in Bp_values_T2:
        Ic_array = []
        for eps_a in eps_a_array:
            # For axial strain only: eps_1=0, eps_2=0, eps_3=eps_a, eps_12=0
            Ic, _, _, _ = calculate_critical_current(0, 0, eps_a, 0, Bp, T2)
            Ic_array.append(Ic)
        
        # Convert to numpy array for easier handling
        Ic_array = np.array(Ic_array)
        
        # Plot this field value
        plt.plot(eps_a_array, Ic_array, linewidth=2, label=f'Bp = {Bp:.0f} T, T = {T2} K')
    
    plt.xlabel('Applied Axial Strain ()')
    plt.ylabel('Critical Current Ic (A)')
    plt.title('Critical Current vs Strain for MQXF 0.85mm 108/127 strand')
    plt.grid(True, alpha=0.3)
    plt.legend()
    # adjust existing figure size (do not create a new figure)
    plt.gcf().set_size_inches(5, 5)
    plt.tight_layout()
    
    # Show and save the plot
    plt.savefig('critical_current_vs_strain.svg', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # Calculate Ic at zero strain (4.2 K, 12 T)
    print("\n" + "="*50)
    print("ZERO STRAIN CRITICAL CURRENT CALCULATION")
    print("="*50)
    T_zero = 4.2  # Temperature (K)
    Bp_zero = 12.0  # Magnetic field (T)
    
    radius = (850e-6)/2  # Strand radius in meters (for 0.85 mm diameter)
    mu0 = 4*math.pi*1e-7  # Permeability of free space (H/m)
    I_encl = 696  # Enclosed current in Amperes
    B_strand = mu0*I_encl/(2*math.pi*radius)
    B_cable = 0.6*I_encl*21/10000  # Approximate cable field contribution in T
    
    
    # All strain components = 0
    Ic_zero_strain, s_eps, _, _ = calculate_critical_current(
        eps_1=0, eps_2=0, eps_3=0, eps_12=0, 
        Bp=Bp_zero+B_strand+B_cable, T=T_zero
    )
    
    print(f"Conditions:")
    print(f"  Temperature: {T_zero} K")
    print(f"  Magnetic Field: {Bp_zero + B_strand + B_cable} T")
    print(f"  All strains: 0")
    print(f"\nResult:")
    print(f"  Ic (zero strain) = {Ic_zero_strain:.2f} A")
    print(f"  S-function (zero strain) = {s_eps:.2f}")
    print("="*50)
    
    
    