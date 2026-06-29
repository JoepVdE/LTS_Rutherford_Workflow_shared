#!/usr/bin/env pvpython
# Run with: & "C:\LTS_Rutherford_Workflow\tools\paraview\ParaView-6.0.1-Windows-Python3.12-msvc2017-AMD64\bin\pvpython.exe" "<script_path>" <subfolder_name>
from paraview.simple import *
import os
import sys
import datetime
import csv


z_location_slice = 0.01
z_location_slice_max = 2*56/21 

# accept subfolder name as command-line argument
if len(sys.argv) < 2:
    print("Usage: pvpython extract_coordinates_stack_sort.py <subfolder_name>")
    sys.exit(1)
subfolder = sys.argv[1]
lsdyna_dir = os.path.join(os.getcwd(), subfolder, "LSDYNA")

# use the subfolder's LSDYNA directory for the d3plot file
FileName = os.path.join(lsdyna_dir, "d3plot")
d3plot = LSDynaReader(registrationName='d3plot', FileName=FileName)
extractSurface1 = ExtractSurface(registrationName='ExtractSurface1', Input=d3plot)

# redirect the downstream filters (e.g. Slice) to use the extracted surface


slice1 = Slice(registrationName='Slice1', Input=extractSurface1)

# for part_id in range(1, 22):
    # part_name = f'Part{part_id}'
    # # Properties modified on d3plot
    # d3plot.PartArrays = [part_name]

    # # create a new 'Slice'

    # for i in range(15):
        
    #     z_location = z_location_slice + i * (z_location_slice_max - z_location_slice) / 15
    #     # Properties modified on slice1.SliceType
    #     slice1.SliceType.Origin = [0.0, 0.0, z_location]
    #     slice1.SliceType.Normal = [0.0, 0.0, 1.0]
    #     UpdatePipeline(time=0.050, proxy=slice1)

    #     # save data
    #     # SaveData(f'C:/Users/vanden_j/OneDrive - ETH Zurich/Documents/ANSYS/LSDYNA/PyANSYS_read_D3plot/Result folder/betterresults/3012866/{foldername}/{part_name}_{z_location:.1f}mm.csv', proxy=slice1, UseScientificNotation=1)
        
    #     # save slice data to current results directory
    #     # create a subfolder named by z_location
    #     z_folder = os.path.join(results_dir, f"{z_location:.1f}mm")
    #     os.makedirs(z_folder, exist_ok=True)
    #     filepath = os.path.join(z_folder, f"{part_name}.csv")
    #     print('filepath', filepath)
    #     SaveData(filepath, proxy=slice1, ChooseArraysToWrite=1, PointDataArrays=['Deflected Coordinates'], UseScientificNotation=1)
        
        # read back the CSV so you can use the coords immediately

        # coords = []
        # with open(filepath, newline='') as csvfile:
        #     reader = csv.DictReader(csvfile)
        #     for row in reader:
        #         coords.append((
        #             float(row['Deflected Coordinates:0']),
        #             float(row['Deflected Coordinates:1'])
        #         ))
        #         alpha = 1
        #         alpha_shape = alphashape.alphashape(coords, alpha)

                
        
                
        # # print(f'coords: {coords}')
        # # now `coords` is a list of (x, y, z) tuples you can work with
        
        # # Step 3: Keep only boundary points
        # boundary_coords = []
        # for coord in coords:
        #     point = Point(coord)
        #     if alpha_shape.boundary.contains(point) or alpha_shape.boundary.touches(point):
        #         boundary_coords.append(coord)

        # # Step 4: Overwrite CSV with only boundary points
        # with open(filepath, mode='w', newline='') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     writer.writeheader()
        #     for coord in boundary_coords:
        #         writer.writerow({
        #             'Deflected Coordinates:0': coord[0],
        #             'Deflected Coordinates:1': coord[1]
        #         })
                
        
        # print(f'Processed {part_name} at z_location {z_location:.1f}mm')
# ...existing code...

# Load finish time and total part count from cable_parameters.json
cable_params_file = os.path.join(os.getcwd(), subfolder, "cable_parameters.json")
with open(cable_params_file, "r") as _f:
    import json as _json
    _cable_params = _json.load(_f)
finish_time = _cable_params["time"]
n_parts_total = _cable_params["N_Strands"] + 4  # wires + 4 plates
n_wire_parts = _cable_params["N_Strands"]  # exclude last 4 plates

stack_dir = os.path.join(os.getcwd(), subfolder, "stack")
os.makedirs(stack_dir, exist_ok=True)

stack_nr = 1  # Start stack numbering at 1

for i in range(15):
    z_location = z_location_slice + i * (z_location_slice_max - z_location_slice) / 15

    for part_id in range(1, n_wire_parts + 1):
        part_name = f'Part{part_id}'
        d3plot.PartArrays = [part_name]

        slice1.SliceType.Origin = [0.0, 0.0, z_location]
        slice1.SliceType.Normal = [0.0, 0.0, 1.0]
        UpdatePipeline(time=finish_time, proxy=slice1)

        # Save as ./stack/Stack_{stack_nr}_Part{part_id}.csv
        filename = f"Stack_{stack_nr}_Part{part_id}.csv"
        filepath = os.path.join(stack_dir, filename)
        print('filepath', filepath)
        SaveData(filepath, proxy=slice1, ChooseArraysToWrite=1, PointDataArrays=['Deflected Coordinates'], UseScientificNotation=1)

    stack_nr += 1  # Increment stack number for each z-location

print(f'All stacks saved in: {stack_dir}')
# print(f'Folder created: C:/Users/vanden_j/OneDrive - ETH Zurich/Documents/ANSYS/LSDYNA/PyANSYS_read_D3plot/Result folder/betterresults/3012866/{foldername}')