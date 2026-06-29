#!/usr/bin/env python3
"""
Simple Nb3Sn Critical Current Calculator

This script calculates the critical current (Ic) of a Nb3Sn strand
using hardcoded parameters for magnetic field (B), temperature (T), 
and strain components using the scaling law implemented in nb3sn_law.py.
"""

import numpy as np
from nb3sn_law import calculate_critical_current, load_scaling_parameters

def main():
    """Main function to calculate critical current with predefined parameters."""
    print("=" * 60)
    print("Nb3Sn Critical Current Calculator")
    print("=" * 60)
    print()
    print("This calculator uses the Nb3Sn scaling law to compute the critical")
    print("current (Ic) based on predefined magnetic field, temperature, and strain.")
    print()
    
    try:
        # Load scaling parameters
        params = load_scaling_parameters()
        print(f"Wire type: RRP MQXF 0.85 mm")
        print()
        
        # Define parameters (modify these values as needed)
        B_values = np.array([7.5, 8.75, 10.0, 12.0])  # Magnetic field values in Tesla
        T = 4.2         # Temperature in Kelvin
        eps_1 = 0.0     # Strain in direction 1
        eps_2 = 0.0     # Strain in direction 2
        eps_3 = 0.0     # Axial strain
        eps_12 = 0.0    # Shear strain
        
        # Self-field options
        include_cable_field = False  # Set to True to include cable self-field
        num_strands = 21            # Number of strands in cable
        
        print("Input Parameters:")
        print("-" * 40)
        print(f"Magnetic field (B):     {B_values} T")
        print(f"Temperature (T):        {T:.2f} K")
        print(f"Strain ε₁:              {eps_1:.6f}")
        print(f"Strain ε₂:              {eps_2:.6f}")
        print(f"Strain ε₃ (axial):      {eps_3:.6f}")
        print(f"Shear strain ε₁₂:       {eps_12:.6f}")
        print(f"Include cable field:    {include_cable_field}")
        print(f"Number of strands:      {num_strands}")
        print()

        
        # Calculate critical current for all B values efficiently
        results = []
        for B in B_values:
            Ic, s_eps, B_strand_self, B_cable_self = calculate_critical_current(
                eps_1=eps_1,
                eps_2=eps_2, 
                eps_3=eps_3,
                eps_12=eps_12,
                Bp=B,
                T=T,
                params=params,
                include_cable_field=include_cable_field,
                num_strands=num_strands
            )
            results.append((B, Ic, s_eps, B_strand_self, B_cable_self))
        
        print("=" * 80)
        print("RESULTS:")
        print("=" * 80)
        print(f"{'B (T)':<8} {'Ic (A)':<12} {'21*Ic (A)':<14} {'B_strand (T)':<12} {'B_cable (T)':<12} {'S-function':<12}")
        print("-" * 80)
        for B, Ic, s_eps, B_strand_self, B_cable_self in results:
            print(f"{B:<8.2f} {Ic:<12.2f} {21*Ic:<14.2f} {B_strand_self:<12.4f} {B_cable_self:<12.4f} {s_eps:<12.4f}")
        print("=" * 80)
        
    except FileNotFoundError:
        print("Error: Could not find scaling_parameters.json file.")
        print("Make sure the file is in the same directory as this script.")
    except Exception as e:
        print(f"Error in calculation: {e}")
        print("Please check the parameter values in the script.")

if __name__ == "__main__":
    main()