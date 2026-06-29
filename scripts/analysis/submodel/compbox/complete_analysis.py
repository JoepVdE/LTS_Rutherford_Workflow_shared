#!/usr/bin/env python3
"""
Complete Strain and Critical Current Analysis Pipeline

This single script performs comprehensive analysis combining:
1. Strain analysis with area-weighted calculations
2. Critical current correlation analysis  
3. Degradation vs strain analysis
4. Multi-set support (up to 16 sets)
5. All visualizations and data exports

Run this ONE file to get everything: python complete_analysis.py
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Any
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import warnings
warnings.filterwarnings('ignore')


SCRIPT_DIR = Path(__file__).resolve().parent
# compbox_stage.py points both env vars into the run folder
# (<run>/APDL/compbox/results/strain_analysis[/Plots]); the SCRIPT_DIR
# fallbacks only apply when running this script by hand.
DEFAULT_RESULTS_DIR = Path(
    os.environ.get("COMPBOX_RESULTS_DIR", str(SCRIPT_DIR / "strain_analysis_results"))
)
PLOT_OUTPUT_DIR = Path(
    os.environ.get("COMPBOX_PLOT_DIR", str(DEFAULT_RESULTS_DIR / "Plots"))
)


def ensure_directory(path: Path) -> Path:
    """Create directory (and parents) if it does not exist and return the Path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_plot(fig, filename: str, plot_dir: Path, *, description: str | None = None, dpi: int = 300) -> Path:
    """Persist a matplotlib figure to SVG and report the destination."""
    plot_dir = ensure_directory(plot_dir)
    output_path = plot_dir / filename
    fig.savefig(output_path, format='svg', bbox_inches='tight', dpi=dpi)
    label = description or "Plot"
    print(f"✅ {label} saved to: {output_path}")
    return output_path


def save_dataframe(
    df: pd.DataFrame,
    path: Path,
    description: str,
    *,
    index: bool = False,
    float_format: str | None = "%.8e",
) -> Path:
    """Persist a DataFrame to CSV with consistent messaging."""
    ensure_directory(path.parent)
    to_csv_kwargs: dict[str, Any] = {"index": index}
    if float_format is not None:
        to_csv_kwargs["float_format"] = float_format
    df.to_csv(path, **to_csv_kwargs)
    print(f"✅ {description} saved to: {path}")
    return path


# =============================================================================
# FILE LOCATION CONFIGURATION - MODIFY THIS SECTION TO CHANGE DATA SOURCE
# =============================================================================

# MAIN DATA DIRECTORY: Where all input files are located.
# compbox_stage.py sets COMPBOX_DATA_DIR to the submodel runfolder, which
# holds the strains_out_*.out dumps and the staged BOX9.txt measurement.
# Fallback is the cwd so the script stays usable by hand.
DATA_SOURCE_DIR = os.environ.get("COMPBOX_DATA_DIR", os.getcwd())

# Expected file structure within DATA_SOURCE_DIR:
# ├── BOX6.txt                    (Critical current vs pressure data)
# ├── strains_out_strand_1_set_1.out
# ├── strains_out_strand_2_set_1.out
# └── ...

# CSV outputs will be saved to: DEFAULT_RESULTS_DIR
# Plots will be saved to: PLOT_OUTPUT_DIR

print(f"🔧 CONFIGURED DATA SOURCE: {DATA_SOURCE_DIR}")
print(f"📄 Expected files: BOX9.txt (or BOX6.txt), strains_out_*.out")

# =============================================================================
# STRAIN ANALYSIS FUNCTIONS
# =============================================================================

def parse_strain_file(filepath):
    """Parse a strain_out file and return strain data."""
    try:
        data = np.loadtxt(filepath, skiprows=1)
        if data.size == 0:
            return None, None
        areas = data[:, 3]  # a_el
        ex = data[:, 4] # e_xx
        ey = data[:, 5] # e_yy
        ez = data[:, 6] # e_zz
        exy = data[:, 7] # e_xy
        v = 0.36
        e_avg = (1/(1+v))*np.sqrt(0.5*((ex-ey)**2 + (ey-ez)**2 + (ez-ex)**2 + 6*exy**2))
        strains = e_avg  # e_avg
        return areas, strains
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None


def extract_strand_set_info(filename):
    """Extract strand and set numbers from filename."""
    match = re.search(r'strand_(\d+)_set_(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def compute_weighted_average_strain(areas, strains):
    """Compute area-weighted average strain."""
    if areas is None or strains is None or len(areas) == 0:
        return None
    
    valid_mask = ~np.isnan(areas) & ~np.isnan(strains) & (areas > 0)
    if not np.any(valid_mask):
        return None
    
    valid_areas = areas[valid_mask]
    valid_strains = strains[valid_mask]
    
    weighted_avg = np.sum(valid_areas * valid_strains) / np.sum(valid_areas)
    return weighted_avg


def load_strain_data(data_base_dir: Path):
    """Load all strain_out files, returning per-element and per-strand datasets."""
    strain_files = sorted(Path(data_base_dir).glob("strains_out_*.out"))
    if not strain_files:
        print(f"❌ No strains_out files found in {data_base_dir}/ directory")
        print("   Expected files matching pattern: strains_out_*.out")
        return None, None

    print(f"📁 Found {len(strain_files)} strain_out files")

    plot_data: list[dict] = []
    weighted_averages: list[dict] = []

    for filepath in strain_files:
        filename = filepath.name
        strand, set_num = extract_strand_set_info(filename)
        if strand is None or set_num is None:
            continue

        areas, strains = parse_strain_file(filepath)
        if areas is None or strains is None:
            continue

        weighted_avg = compute_weighted_average_strain(areas, strains)
        if weighted_avg is None:
            continue

        weighted_averages.append({
            'strand': strand,
            'set': set_num,
            'filename': filename,
            'num_elements': len(areas),
            'total_area': np.sum(areas),
            'weighted_avg_strain': weighted_avg,
            'min_strain': np.min(strains),
            'max_strain': np.max(strains),
            'unweighted_avg_strain': np.mean(strains)
        })

        valid_mask = ~np.isnan(areas) & ~np.isnan(strains) & (areas > 0)
        for strain_val in strains[valid_mask]:
            plot_data.append({
                'strand': strand,
                'set': set_num,
                'strain': strain_val
            })

    if not plot_data:
        print("❌ No valid strain data for analysis")
        return None, None

    plot_df = pd.DataFrame(plot_data)
    avg_df = pd.DataFrame(weighted_averages)
    print(f"✅ Processed {len(plot_df)} strain values from {len(weighted_averages)} strand-set combinations")
    return plot_df, avg_df


# =============================================================================
# BOX6 DATA PARSING FUNCTIONS
# =============================================================================

def parse_box6_data(filepath):
    """Parse BOX6.txt file containing pressure and Ic data."""
    try:
        # Read the data line by line to handle inconsistent formatting
        data_rows = []
        with open(filepath, 'r') as f:
            for line in f:
                # Split by tab and clean up
                parts = line.strip().split('\t')
                # Remove empty parts and clean up
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) >= 3:  # Need at least pressure, ic_current, ic_permanent
                    data_rows.append(parts)
        
        if not data_rows:
            print(f"No valid data found in {filepath}")
            return None
        
        # Convert to DataFrame
        max_cols = max(len(row) for row in data_rows)
        
        # Pad shorter rows with NaN
        for row in data_rows:
            while len(row) < max_cols:
                row.append('NaN')
        
        # Create DataFrame with proper column names
        if max_cols >= 6:
            columns = ['pressure_mpa', 'ic_current', 'ic_permanent', 'ratio1', 'ratio2', 'degradation']
        elif max_cols >= 3:
            columns = ['pressure_mpa', 'ic_current', 'ic_permanent']
            if max_cols > 3:
                columns.extend([f'col_{i}' for i in range(3, max_cols)])
        else:
            print(f"Insufficient columns in {filepath}")
            return None
        
        data = pd.DataFrame(data_rows, columns=columns[:max_cols])
        
        # Convert numeric columns
        numeric_cols = ['pressure_mpa', 'ic_current', 'ic_permanent']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Convert other potential numeric columns
        for col in data.columns:
            if col not in numeric_cols:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Clean the data - remove rows where essential data is missing
        data = data.dropna(subset=['pressure_mpa', 'ic_current'])
        
        # Calculate degradation if we have ic_permanent data
        if 'ic_permanent' in data.columns:
            # Calculate degradation as (ic_current - ic_permanent)/ic_current where available
            mask = (data['ic_permanent'].notna()) & (data['ic_current'] > 0)
            data.loc[mask, 'degradation_calculated'] = (data.loc[mask, 'ic_current'] - data.loc[mask, 'ic_permanent']) / data.loc[mask, 'ic_current']
        
        print(f"   Parsed {len(data)} rows with {data.shape[1]} columns")
        
        return data
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def load_box6_dataset(data_base_dir: Path):
    """Load the experimental Ic-vs-pressure table if available.

    Tries BOX9.txt (SMACC_HF campaign) first, then BOX6.txt (CD1). Both share
    the tab-separated format: pressure | Ic under load | Ic after unload | ...
    Override the filename with COMPBOX_EXPERIMENT_FILE.
    """
    env_name = os.environ.get("COMPBOX_EXPERIMENT_FILE")
    candidates = [env_name] if env_name else ["BOX9.txt", "BOX6.txt"]
    box6_file = None
    for name in candidates:
        p = Path(data_base_dir) / name
        if p.exists():
            box6_file = p
            break
    if box6_file is None:
        print(f"⚠️  None of {candidates} found in {data_base_dir} - skipping critical current analysis")
        return None

    print(f"📊 Loading critical current data from: {box6_file}")
    box6_df = parse_box6_data(box6_file)

    if box6_df is None or len(box6_df) == 0:
        print("❌ Failed to load BOX6 data")
        return None

    print(f"✅ Loaded {len(box6_df)} pressure/Ic data points")
    print(f"   Pressure range: {box6_df['pressure_mpa'].min():.1f} - {box6_df['pressure_mpa'].max():.1f} MPa")
    print(f"   Ic range: {box6_df['ic_current'].min():.0f} - {box6_df['ic_current'].max():.0f} A")
    return box6_df


# =============================================================================
# CORRELATION FUNCTIONS
# =============================================================================

def correlate_strain_pressure(strain_df, box6_df, set_mapping=None):
    """Correlate strain data with pressure/Ic data."""
    
    if set_mapping is None:
        unique_sets = sorted(strain_df['set'].unique())
        # Table rows in file order (ascending peak pressure); row 0 is the
        # baseline holding pressure (10 MPa for BOX6, 39.9 MPa for BOX9).
        file_order_pressures = list(box6_df['pressure_mpa'].values)
        baseline_pressure = file_order_pressures[0]

        if len(unique_sets) == 2 * len(file_order_pressures):
            # Interleaved LOAD/UNLOAD schedule (box9_loading.inp convention):
            # set 2k-1 = load peak of table row k, set 2k = unload to baseline.
            set_mapping = {}
            for k in range(1, len(file_order_pressures) + 1):
                set_mapping[2 * k - 1] = file_order_pressures[k - 1]
                set_mapping[2 * k] = baseline_pressure
            print(f"   Using interleaved load/unload set mapping "
                  f"({len(unique_sets)} sets = 2 x {len(file_order_pressures)} table rows)")
        else:
            # Fallback: sorted sets mapped to sorted pressures in order
            # (legacy behaviour, e.g. loading-only runs).
            pressure_points = sorted(file_order_pressures)
            set_mapping = {}
            for i, set_num in enumerate(unique_sets):
                if i < len(pressure_points):
                    set_mapping[set_num] = pressure_points[i]
                else:
                    # If more sets than pressure points, use last value
                    set_mapping[set_num] = pressure_points[-1]
            print(f"   WARNING: {len(unique_sets)} sets vs {len(file_order_pressures)} "
                  "table rows - not an interleaved pair count; using legacy "
                  "sorted set->pressure mapping. Check this is intended.")
    
    # Create correlation data
    correlation_data = []
    
    for set_num in strain_df['set'].unique():
        if set_num in set_mapping:
            pressure = set_mapping[set_num]
            
            # Find closest pressure point in BOX6 data
            pressure_idx = np.argmin(np.abs(box6_df['pressure_mpa'] - pressure))
            box6_row = box6_df.iloc[pressure_idx]
            
            # Get strain data for this set
            set_strains = strain_df[strain_df['set'] == set_num]
            
            for _, strain_row in set_strains.iterrows():
                correlation_data.append({
                    'set': set_num,
                    'strand': strain_row['strand'],
                    'pressure_mpa': pressure,
                    'actual_pressure_mpa': box6_row['pressure_mpa'],
                    'ic_current': box6_row['ic_current'],
                    'ic_permanent': box6_row.get('ic_permanent', np.nan),
                    'degradation': box6_row.get('degradation', np.nan),
                    'degradation_calculated': box6_row.get('degradation_calculated', np.nan),
                    'weighted_avg_strain': strain_row['weighted_avg_strain'],
                    'min_strain': strain_row['min_strain'],
                    'max_strain': strain_row['max_strain'],
                    'num_elements': strain_row['num_elements'],
                    'total_area': strain_row['total_area']
                })
    
    return pd.DataFrame(correlation_data), set_mapping


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_strain_ic_correlation_plots(correlation_df, plot_dir):
    """Create strain-Ic correlation analysis plots."""
    
    print("\n" + "=" * 60)
    print("CREATING STRAIN-IC CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Main correlation figure with only visualization subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Strain vs Pressure scatter plot
    ax1 = axes[0, 0]
    try:
        scatter = ax1.scatter(correlation_df['actual_pressure_mpa'], 
                             correlation_df['weighted_avg_strain'],
                             c=correlation_df['strand'], 
                             cmap='viridis', alpha=0.7, s=50)
        ax1.set_xlabel('Pressure (MPa)')
        ax1.set_ylabel('Weighted Average Strain')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Strand Number')
        
        # Add trend line
        if len(correlation_df) > 1:
            z = np.polyfit(correlation_df['actual_pressure_mpa'], 
                          correlation_df['weighted_avg_strain'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(correlation_df['actual_pressure_mpa'].min(), 
                                 correlation_df['actual_pressure_mpa'].max(), 100)
            ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
            ax1.legend()
            
    except Exception as e:
        print(f"❌ Error in strain vs pressure plot: {e}")
    
    # 2. Strain vs Ic Current
    ax2 = axes[0, 1]
    try:
        scatter2 = ax2.scatter(correlation_df['ic_current'], 
                              correlation_df['weighted_avg_strain'],
                              c=correlation_df['actual_pressure_mpa'], 
                              cmap='plasma', alpha=0.7, s=50)
        ax2.set_xlabel('Critical Current Ic (A)')
        ax2.set_ylabel('Weighted Average Strain')
        ax2.grid(True, alpha=0.3)
        
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Pressure (MPa)')
        
        # Correlation coefficient
        if len(correlation_df) > 1:
            corr_coef = correlation_df['ic_current'].corr(correlation_df['weighted_avg_strain'])
            ax2.text(0.05, 0.95, f'R = {corr_coef:.3f}', 
                    transform=ax2.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
    except Exception as e:
        print(f"❌ Error in strain vs Ic plot: {e}")
    
    # 3. Strain distribution by pressure level
    ax3 = axes[1, 0]
    try:
        # Create box plot of strain distribution by pressure
        pressure_levels = sorted(correlation_df['actual_pressure_mpa'].unique())
        strain_by_pressure = [correlation_df[correlation_df['actual_pressure_mpa'] == p]['weighted_avg_strain'].values 
                             for p in pressure_levels]
        
        if len(strain_by_pressure) > 0:
            bp = ax3.boxplot(strain_by_pressure, labels=[f'{p:.1f}' for p in pressure_levels], 
                           patch_artist=True)
            
            # Color the boxes
            cmap = plt.get_cmap('viridis')
            colors = cmap(np.linspace(0, 1, len(bp['boxes'])))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax3.set_xlabel('Pressure (MPa)')
            ax3.set_ylabel('Weighted Average Strain')
            ax3.grid(True, alpha=0.3)
            plt.setp(ax3.get_xticklabels(), rotation=45)
            
    except Exception as e:
        print(f"❌ Error in strain distribution plot: {e}")
    
    # 4. Pressure vs Ic Current relationship
    ax4 = axes[1, 1]
    try:
        # Get unique pressure-Ic combinations
        pressure_ic = correlation_df.groupby('actual_pressure_mpa')['ic_current'].first().reset_index()
        
        ax4.plot(pressure_ic['actual_pressure_mpa'], pressure_ic['ic_current'], 
                'bo-', linewidth=2, markersize=8, alpha=0.8)
        ax4.set_xlabel('Pressure (MPa)')
        ax4.set_ylabel('Critical Current Ic (A)')
        ax4.grid(True, alpha=0.3)
        
        # Add data points
        ax4.scatter(pressure_ic['actual_pressure_mpa'], pressure_ic['ic_current'], 
                   c='red', s=100, alpha=0.6, zorder=10)
        
        # Add correlation coefficient
        if len(pressure_ic) > 1:
            corr_coef = pressure_ic['actual_pressure_mpa'].corr(pressure_ic['ic_current'])
            ax4.text(0.05, 0.95, f'R = {corr_coef:.3f}', 
                    transform=ax4.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                   
    except Exception as e:
        print(f"❌ Error in pressure vs Ic plot: {e}")
    
    fig.tight_layout()

    try:
        save_plot(fig, "strain_ic_correlation_analysis.svg", plot_dir, description="Strain-Ic correlation analysis")
    except Exception as e:
        print(f"❌ Error saving correlation plot: {e}")
    finally:
        plt.close(fig)


def create_individual_set_analyses(avg_df, plot_df, plot_dir):
    """Create individual statistical analysis for each set."""
    
    print("\n" + "=" * 60)
    print("CREATING INDIVIDUAL SET STATISTICAL ANALYSES")
    print("=" * 60)
    
    # Get all available sets
    available_sets = sorted(avg_df['set'].unique())
    print(f"Creating individual analyses for {len(available_sets)} sets: {available_sets}")
    
    for set_num in available_sets:
        print(f"  Processing Set {set_num}...")
        
        # Filter data for this set
        set_avg_df = avg_df[avg_df['set'] == set_num].copy()
        set_plot_df = plot_df[plot_df['set'] == set_num].copy()
        
        if len(set_avg_df) == 0:
            print(f"  ⚠️  No data for Set {set_num}")
            continue
        
        # Create figure for this set
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        try:
            # 1. Histogram of strain values
            ax1 = axes[0, 0]
            if len(set_plot_df) > 0:
                ax1.hist(set_plot_df['strain'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.axvline(set_plot_df['strain'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
                ax1.axvline(set_plot_df['strain'].median(), color='orange', linestyle='--', linewidth=2, label='Median')
                ax1.set_xlabel('Strain')
                ax1.set_ylabel('Frequency')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # 2. Box plot by strand
            ax2 = axes[0, 1]
            if len(set_plot_df) > 0:
                strands = sorted(set_plot_df['strand'].unique())
                strain_by_strand = [set_plot_df[set_plot_df['strand'] == s]['strain'].values for s in strands]
                bp = ax2.boxplot(strain_by_strand, labels=strands, patch_artist=True)
                
                # Color the boxes
                cmap = plt.get_cmap('viridis')
                colors = cmap(np.linspace(0, 1, len(bp['boxes'])))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax2.set_xlabel('Strand Number')
                ax2.set_ylabel('Strain')
                ax2.grid(True, alpha=0.3)
                plt.setp(ax2.get_xticklabels(), rotation=45)
            
            # 3. Weighted averages scatter
            ax3 = axes[0, 2]
            ax3.scatter(set_avg_df['strand'], set_avg_df['weighted_avg_strain'], 
                       s=100, alpha=0.7, color='red', edgecolors='black')
            ax3.set_xlabel('Strand Number')
            ax3.set_ylabel('Weighted Average Strain')
            ax3.grid(True, alpha=0.3)
            
            # Add trend line
            if len(set_avg_df) > 1:
                z = np.polyfit(set_avg_df['strand'], set_avg_df['weighted_avg_strain'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(set_avg_df['strand'].min(), set_avg_df['strand'].max(), 100)
                ax3.plot(x_trend, p(x_trend), "b--", alpha=0.8, linewidth=2)
            
            # 4. Statistics table
            ax4 = axes[1, 0]
            ax4.axis('off')
            
            # Calculate statistics
            stats_data = []
            for _, row in set_avg_df.iterrows():
                strand_data = set_plot_df[set_plot_df['strand'] == row['strand']]['strain']
                stats_data.append([
                    f"Strand {int(row['strand'])}",
                    f"{row['weighted_avg_strain']:.3e}",
                    f"{strand_data.mean():.3e}",
                    f"{strand_data.std():.3e}",
                    f"{strand_data.min():.3e}",
                    f"{strand_data.max():.3e}",
                    f"{len(strand_data)}"
                ])
            
            # Create table
            table = ax4.table(cellText=stats_data,
                            colLabels=['Strand', 'Weighted Avg', 'Mean', 'Std', 'Min', 'Max', 'Count'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            
            # 5. Overall statistics
            ax5 = axes[1, 1]
            ax5.axis('off')
            
            overall_stats = f"""Set {set_num} Overall Statistics:

Total Strain Values: {len(set_plot_df):,}
Number of Strands: {len(set_avg_df)}

Strain Statistics:
  Mean: {set_plot_df['strain'].mean():.3e}
  Median: {set_plot_df['strain'].median():.3e}
  Std Dev: {set_plot_df['strain'].std():.3e}
  Min: {set_plot_df['strain'].min():.3e}
  Max: {set_plot_df['strain'].max():.3e}
  Range: {set_plot_df['strain'].max() - set_plot_df['strain'].min():.3e}

Weighted Average Statistics:
  Mean: {set_avg_df['weighted_avg_strain'].mean():.3e}
  Std Dev: {set_avg_df['weighted_avg_strain'].std():.3e}
  Min: {set_avg_df['weighted_avg_strain'].min():.3e}
  Max: {set_avg_df['weighted_avg_strain'].max():.3e}"""
            
            ax5.text(0.05, 0.95, overall_stats, transform=ax5.transAxes, 
                    verticalalignment='top', fontsize=10, fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            
            # 6. Strain range comparison
            ax6 = axes[1, 2]
            strand_means = set_avg_df.groupby('strand')['weighted_avg_strain'].mean()
            ax6.bar(strand_means.index, strand_means.values, alpha=0.7, color='green', edgecolor='black')
            ax6.set_xlabel('Strand Number')
            ax6.set_ylabel('Weighted Average Strain')
            ax6.grid(True, alpha=0.3)
            plt.setp(ax6.get_xticklabels(), rotation=45)
            
        except Exception as e:
            print(f"  ❌ Error creating plots for Set {set_num}: {e}")
        
        fig.tight_layout()

        try:
            save_plot(
                fig,
                f"strain_statistical_summary_set_{set_num}.svg",
                plot_dir,
                description=f"Set {set_num} analysis",
            )
        except Exception as e:
            print(f"  ❌ Error saving Set {set_num} plot: {e}")
        finally:
            plt.close(fig)
    
    print(f"✅ Individual set analyses completed for {len(available_sets)} sets")


def create_strain_distribution_by_pressure(avg_df, plot_df, set_mapping, plot_dir):
    """Create strain distribution boxplot with pressure labels."""

    print("\n" + "=" * 60)
    print("CREATING STRAIN DISTRIBUTION BY PRESSURE PLOT")
    print("=" * 60)

    available_sets = sorted(avg_df['set'].unique())
    print(f"Creating strain distribution plot for {len(available_sets)} pressure conditions")

    fig, ax = plt.subplots(figsize=(8, 6))

    set_data: list[np.ndarray] = []
    pressure_labels: list[str] = []
    all_strains: list[float] = []

    for set_num in available_sets:
        set_strains = plot_df[plot_df['set'] == set_num]['strain'].values
        if len(set_strains) > 0:
            set_data.append(set_strains)
            all_strains.extend(set_strains)
            pressure = set_mapping.get(set_num, 0)
            pressure_labels.append(f"{int(round(pressure))}")

    if not set_data:
        print("❌ No data available for strain distribution plot")
        plt.close(fig)
        return

    all_strains_array = np.array(all_strains)
    p5 = np.percentile(all_strains_array, 5)
    p95 = np.percentile(all_strains_array, 95)

    bp = ax.boxplot(
        set_data,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker='o', markersize=3, alpha=0.3),
    )

    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    strain_range = p95 - p5
    y_padding = strain_range * 0.5
    lower_ylim = float(max(0.0, p5 - y_padding))
    upper_ylim = float(p95 + y_padding)
    ax.set_ylim(lower_ylim, upper_ylim)

    # Use plain decimal formatting for y-axis (like the heatmap)
    from matplotlib.ticker import FuncFormatter
    def format_func(value, tick_number):
        return f'{value:.3f}'
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))

    ax.set_xlabel('Average Transverse Stress (MPa)', fontsize=16)
    ax.set_ylabel('Simulated Von-Mises Strain (-)', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, len(pressure_labels) + 1))
    ax.set_xticklabels(pressure_labels, rotation=45, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Add horizontal colorbar below similar to strain_heatmap_by_pressure
    # Calculate median strain for each pressure to match colorbar with box colors
    median_strains = [np.median(data) for data in set_data]
    strain_min = min(median_strains)
    strain_max = max(median_strains)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=strain_min, vmax=strain_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.15, shrink=0.8)
    cbar.set_label('Simulated Von-Mises Strain (-)', fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    fig.tight_layout()

    try:
        save_plot(fig, "strain_distribution_by_pressure.svg", plot_dir, description="Strain distribution by pressure plot")
    except Exception as e:
        print(f"❌ Error saving strain distribution plot: {e}")
    finally:
        plt.close(fig)


def create_strain_heatmap_by_pressure(avg_df, set_mapping, plot_dir):
    """Create strain heatmap with pressure on x-axis."""
    
    print("\n" + "=" * 60)
    print("CREATING STRAIN HEATMAP BY PRESSURE")
    print("=" * 60)
    
    available_sets = sorted(avg_df['set'].unique())
    print(f"Creating strain heatmap for {len(available_sets)} pressure conditions")
    
    fig, ax = plt.subplots(figsize=(8, 9))
    
    try:
        # Create pivot table with strand vs set
        pivot_data = avg_df.pivot(index='strand', columns='set', values='weighted_avg_strain')
        
        # Create pressure labels for x-axis
        pressure_labels = []
        for set_num in pivot_data.columns:
            pressure = set_mapping.get(set_num, 0)
            pressure_labels.append(f'{int(round(pressure))}')
        
        # Create heatmap
        im = ax.imshow(pivot_data.values, cmap='viridis', aspect='auto', interpolation='nearest')

        ax.set_xticks(range(len(pivot_data.columns)))
        ax.set_xticklabels(pressure_labels, fontsize=14)
        ax.set_yticks(range(len(pivot_data.index)))
        ax.set_yticklabels(pivot_data.index, fontsize=14)

        ax.set_xlabel('Average Transverse Stress (MPa)', fontsize=16)
        ax.set_ylabel('Strand Number', fontsize=16)

        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.15, shrink=0.8)
        cbar.set_label('Simulated Von-Mises Strain (-)', fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        fig.tight_layout()

        try:
            save_plot(fig, "strain_heatmap_by_pressure.svg", plot_dir, description="Strain heatmap by pressure")
        except Exception as e:
            print(f"❌ Error saving heatmap: {e}")
        finally:
            plt.close(fig)

    except Exception as e:
        print(f"❌ Error creating strain heatmap: {e}")
        plt.close(fig)





# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def main():
    """Complete analysis pipeline."""

    print("=" * 80)
    print("COMPLETE STRAIN AND CRITICAL CURRENT ANALYSIS PIPELINE")
    print("=" * 80)

    data_base_dir = Path(DATA_SOURCE_DIR)
    print(f"📁 Data source directory: {data_base_dir}")

    if not data_base_dir.exists():
        print(f"❌ Error: Data directory does not exist: {data_base_dir}")
        print("   Please modify DATA_SOURCE_DIR in the configuration section and try again.")
        return

    results_dir = ensure_directory(DEFAULT_RESULTS_DIR)
    plot_dir = ensure_directory(PLOT_OUTPUT_DIR)
    print(f"📁 CSV output directory: {results_dir}")
    print(f"🖼️ Plot output directory: {plot_dir}")

    print("\n" + "🔧 STEP 1: STRAIN ANALYSIS")
    print("=" * 50)

    plot_df, avg_df = load_strain_data(data_base_dir)
    if plot_df is None or avg_df is None:
        return

    save_dataframe(avg_df, results_dir / "strain_weighted_averages.csv", "Strain data")

    print("\n" + "🔧 STEP 2: CRITICAL CURRENT CORRELATION ANALYSIS")
    print("=" * 50)

    box6_df = load_box6_dataset(data_base_dir)
    correlation_df = None
    set_mapping: dict[int, float] = {}

    if box6_df is not None:
        print("🔗 Correlating strain and critical current data...")
        correlation_df, set_mapping = correlate_strain_pressure(avg_df, box6_df)
        print(f"✅ Created correlation data with {len(correlation_df)} data points")
        print(f"   Set-pressure mapping: {set_mapping}")

        save_dataframe(correlation_df, results_dir / "strain_ic_correlation_data.csv", "Strain-Ic correlation data")
        mapping_df = pd.DataFrame(list(set_mapping.items()), columns=['set', 'pressure_mpa'])
        save_dataframe(mapping_df, results_dir / "set_pressure_mapping.csv", "Set-pressure mapping", float_format=None)

        create_strain_ic_correlation_plots(correlation_df, plot_dir)

        print("\n" + "🔧 STEP 3: INDIVIDUAL SET ANALYSES")
        print("=" * 50)
        create_individual_set_analyses(avg_df, plot_df, plot_dir)

        print("\n" + "🔧 STEP 4: STRAIN DISTRIBUTION AND HEATMAP BY PRESSURE")
        print("=" * 50)
        if set_mapping:
            create_strain_distribution_by_pressure(avg_df, plot_df, set_mapping, plot_dir)
            create_strain_heatmap_by_pressure(avg_df, set_mapping, plot_dir)
        else:
            print("❌ Cannot create pressure-based plots: missing pressure data or mapping")

        print("\n" + "🔧 STEP 5: PRESSURE vs IC REDUCTION & STRAIN ANALYSIS")
        print("=" * 50)
        create_pressure_ic_strain_plot(correlation_df, box6_df, plot_dir)
    else:
        print("ℹ️  Skipping correlation-dependent analyses because BOX6.txt was not found.")

    print("\n" + "=" * 80)
    print("COMPLETE ANALYSIS SUMMARY")
    print("=" * 80)

    unique_sets = sorted(avg_df['set'].unique())
    unique_strands = sorted(avg_df['strand'].unique())

    print("📊 Strain Analysis:")
    print(f"   Total strain values analyzed: {len(plot_df):,}")
    print(f"   Strands: {len(unique_strands)} (Range: {min(unique_strands)}-{max(unique_strands)})")
    print(f"   Sets: {len(unique_sets)} (Available: {unique_sets})")
    print(f"   Strain range: {avg_df['weighted_avg_strain'].min():.3e} to {avg_df['weighted_avg_strain'].max():.3e}")

    if correlation_df is not None and box6_df is not None:
        print("\n📊 Critical Current Analysis:")
        print(f"   Pressure range: {box6_df['pressure_mpa'].min():.1f} - {box6_df['pressure_mpa'].max():.1f} MPa")
        print(f"   Ic range: {box6_df['ic_current'].min():.0f} - {box6_df['ic_current'].max():.0f} A")
        print(f"   Correlation data points: {len(correlation_df)}")

        try:
            strain_pressure_corr = correlation_df['weighted_avg_strain'].corr(correlation_df['actual_pressure_mpa'])
            strain_ic_corr = correlation_df['weighted_avg_strain'].corr(correlation_df['ic_current'])
            print("\n📈 Key Correlations:")
            print(f"   Strain vs Pressure: {strain_pressure_corr:.4f}")
            print(f"   Strain vs Ic:       {strain_ic_corr:.4f}")
        except Exception:
            pass
    else:
        print("\n⚠️  Critical current correlation metrics were skipped (BOX6.txt missing).")

    print("\n📋 Analysis Components Generated:")
    if correlation_df is not None:
        print("   📊 Strain-Ic correlation analysis")
        print(f"   📋 Individual statistical summaries for {len(unique_sets)} sets")
        print("   📦 Strain distribution by pressure (boxplot)")
        print("   🔥 Strain heatmap by pressure")
        print("   📊 Pressure vs Ic reduction & strain dual-axis plot")
    else:
        print("   ⚠️ Correlation-dependent plots were not generated")

    csv_files = sorted(results_dir.glob("*.csv"))
    plot_files = sorted(plot_dir.glob("*.svg"))

    print(f"\n� CSV data files in '{results_dir}': {len(csv_files)}")
    for file in csv_files:
        size_kb = file.stat().st_size / 1024
        print(f"   • {file.name} ({size_kb:.1f} KB)")

    print(f"\n🎨 Plot files in '{plot_dir}': {len(plot_files)}")
    for file in plot_files:
        size_kb = file.stat().st_size / 1024
        print(f"   • {file.name} ({size_kb:.1f} KB)")

    print(f"\n✅ Complete analysis finished! CSV outputs available in '{results_dir}'")
    print(f"�️ All plots saved in '{plot_dir}'")
    print("=" * 80)


def create_pressure_ic_strain_plot(correlation_df, box6_df, plot_dir):
    """
    Create a plot showing normalized critical current vs applied pressure with
    a secondary y-axis for simulated average strain with uncertainty bands.
    """
    print("\n" + "=" * 60)
    print("CREATING PRESSURE vs NORMALIZED CRITICAL CURRENT & STRAIN PLOT")
    print("=" * 60)
    
    fig = None

    try:
        # Sort by pressure to ensure first value is the lowest pressure
        box6_df_sorted = box6_df.sort_values('pressure_mpa').copy()
        
        # Normalize Ic values relative to the first (lowest pressure) measurement
        first_ic = box6_df_sorted['ic_current'].iloc[0]
        box6_df_sorted['ic_normalized'] = box6_df_sorted['ic_current'] / first_ic
        
        # Calculate strain statistics by pressure using actual 95% confidence intervals
        strain_stats = correlation_df.groupby('actual_pressure_mpa').agg({
            'weighted_avg_strain': ['mean', 'std', 'count', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)]
        }).reset_index()
        
        strain_stats.columns = ['pressure', 'strain_mean', 'strain_std', 'strain_count', 'strain_5th', 'strain_95th']
        strain_stats['strain_std'] = strain_stats['strain_std'].fillna(0)
        strain_stats['strain_lower'] = strain_stats['strain_5th']
        strain_stats['strain_upper'] = strain_stats['strain_95th']
        
        # Create the plot with dual y-axes
        fig, ax1 = plt.subplots(figsize=(13, 6))

        # Primary y-axis: Normalized critical current
        # Convert 4.5% error to normalized error (relative to first measurement)
        ic_error_normalized = 0.045  # 4.5% error in critical current measurement
        pressure_error = 5  # 5 MPa pressure error

        color1 = 'tab:red'
        ax1.set_xlabel('Applied Pressure (MPa)', fontsize=14)
        ax1.set_ylabel('Normalized Critical Current $I_c/I_{c0}$', color=color1, fontsize=14)

        # Plot normalized Ic with error bars
        ax1.errorbar(
            box6_df_sorted['pressure_mpa'],
            box6_df_sorted['ic_normalized'],
            yerr=ic_error_normalized,
            xerr=pressure_error,
            fmt='o-',
            color=color1,
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=2,
            label='$I_c/I_{c0}$ (measured, ±4%)',
        )

        ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
        ax1.grid(True, alpha=0.3)

        # Secondary y-axis: Strain with uncertainty bands
        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel('Simulated Von-Mises Strain (-)', color=color2, fontsize=14)

        # Plot strain mean line
        ax2.plot(
            strain_stats['pressure'],
            strain_stats['strain_mean'],
            'o-',
            color=color2,
            linewidth=2,
            markersize=8,
            label='Von-Mises Strain (simulated)',
        )

        # Plot 95% confidence interval bands
        ax2.fill_between(
            strain_stats['pressure'],
            strain_stats['strain_lower'],
            strain_stats['strain_upper'],
            alpha=0.3,
            color=color2,
            label='95% Confidence Interval (simulated)',
        )

        ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)

        # Expand both y-axis scales by 10%
        ic_ymin, ic_ymax = ax1.get_ylim()
        ic_range = ic_ymax - ic_ymin
        ax1.set_ylim(ic_ymin - 0.1 * ic_range, ic_ymax + 0.2 * ic_range)

        strain_ymin, strain_ymax = ax2.get_ylim()
        ax2.set_ylim(0, strain_ymax * 1.15)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc='upper center',
            fontsize=12,
            framealpha=0.95,
            fancybox=False,
            shadow=False,
            bbox_to_anchor=(0.5, 1.0),
        )

        fig.tight_layout()

        try:
            output_path = save_plot(
                fig,
                'pressure_normalized_ic_strain.svg',
                plot_dir,
                description="Pressure vs normalized Ic & strain plot",
            )
        except Exception as e:
            print(f"❌ Error saving pressure vs Ic reduction & strain plot: {e}")
            raise
        finally:
            if fig is not None:
                plt.close(fig)
                fig = None
        
        print(f"✅ Pressure vs normalized Ic & strain plot saved to: {output_path}")
        print(f"📊 Plotted {len(box6_df_sorted)} pressure points with error bars")
        print(f"📊 Strain data from {len(strain_stats)} pressure conditions")
        print(f"📊 Ic normalized relative to first measurement: {first_ic:.0f} A")
        
        return str(output_path)
        
    except Exception as e:
        print(f"❌ Error creating pressure vs Ic reduction & strain plot: {e}")
        import traceback
        traceback.print_exc()
        if fig is not None:
            plt.close(fig)
        return None


if __name__ == "__main__":
    main()
