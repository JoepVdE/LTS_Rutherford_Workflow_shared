class ContactNodes:
    def __init__(self, connections_file, output_file, stack_nr, elem_per_strand=12*12 + 12*12*2,num_strands=21,n_stacks=10,elem_strand_offset_increase = 10000):
        """
        Initializes the ContactNodes class.
        Parameters:
        connections_file (str): Path to the file containing connections.
        output_file (str): Path to the output file for writing contact elements.
        elem_per_strand (int): Number of elements per strand. n_square = 12 * 12  # Total elements in the square core region  n_other = 12 * 12 * 2  # Total elements in the outer circular region
        """
        self.connections_file = connections_file
        self.output_file = output_file
        # self.elem_per_strand = elem_per_strand
        self.connection_list = []
        self.contact_element = 'CONTA172'
        self.friction = 0.2
        self.contact_type = 2
        self.stack_nr = stack_nr  # Stack number for the contact elements
        self.element_number = 20 + (self.stack_nr - 1) * 1100  # Reduced from 1e4 (need >1000 for target offset gap)


        self.num_strands = num_strands  # Number of strands
        self.n_stacks = n_stacks  # Number of stacks
        
        self.elem_strand_offset_increase = elem_strand_offset_increase
        self.elem_strand_offset_strack_nr = self.elem_strand_offset_increase*55  # Reduced from *100
        self.elem_strand_offset = self.elem_strand_offset_strack_nr + 1
        
        self.elem_per_strand = elem_strand_offset_increase


    def read_connections(self):
        """Reads the connections from the file and processes them."""
        try:
            with open(self.connections_file, "r") as file:
                connections = [line.strip() for line in file if line.strip() and line.strip()[0].isdigit()]
                for connection in connections:
                    numbers = connection.split(',')
                    if len(numbers) >= 2:
                        self.connection_list.append([int(numbers[0]), int(numbers[1])])
        except Exception as e:
            print(f"An error occurred while reading connections: {e}")

    def get_strand_element_indexes(self, strand_number,):
        """
        Calculate the element indexes for a given strand.

        Parameters:
        strand_number (int): The strand number (1-based index).

        Returns:
        tuple: A tuple containing the start and end indexes for the strand.
        """
        start_index = (strand_number - 1) * self.elem_per_strand + 1 + (self.stack_nr - 1) * self.elem_strand_offset_strack_nr
        end_index = strand_number * self.elem_per_strand + (self.stack_nr - 1) * self.elem_strand_offset_strack_nr
        return start_index, end_index

    def write_contacts(self):
        """Writes the contact elements to the output file."""
        try:
            with open(self.output_file, "w") as output_file:
                for parts in self.connection_list:
                    # Write contact element definition
                    output_file.write(f"et,{self.element_number},{self.contact_element}\n")
                    output_file.write(f"KEYOPT,{self.element_number},1,0\n")  # Ux, Uy
                    output_file.write(f"KEYOPT,{self.element_number},2,1\n")  # springs
                    output_file.write(f"KEYOPT,{self.element_number},9,1\n")  # Exclude penetration or gap
                    output_file.write(f"KEYOPT,{self.element_number},12,{self.contact_type}\n")  # Contact type
                    output_file.write(f"mp,mu,{self.element_number},{self.friction}\n")
                    
                    output_file.write(f"r,{self.element_number}\n\n")
                    output_file.write(f"et,{self.element_number + 1000},TARGE169\n")
                    output_file.write(f"real,{self.element_number}\n")
                    output_file.write(f"mat,{self.element_number}\n\n\n")

                    # Write element selection for the first strand
                    output_file.write("allsel \n")
                    output_file.write("asel,none \n")
                    output_file.write("cmsel,none \n")
                    
                    output_file.write(f"cmsel,a,i_{self.stack_nr}_{parts[0]}   \n")
                    output_file.write(f"cmsel,a,m_{self.stack_nr}_{parts[0]}   \n")
                    output_file.write(f"cmsel,a,o_{self.stack_nr}_{parts[0]}   \n")
                    output_file.write("allsel,belo,area\n")
                    output_file.write(f"type,{self.element_number}\n")

                    output_file.write("esurf\n")
                    
                    
                    # start_elem, end_elem = self.get_strand_element_indexes(parts[0])
                    # output_file.write(f"esel,s,,,{start_elem},{end_elem}\n")
                    # output_file.write("allsel,below,elem\n")
                    # output_file.write(f"type,{self.element_number}\n")
                    # output_file.write("eplot\n")
                    # output_file.write("esurf\n")
                    # output_file.write("allsel\n\n")

                    # Write element selection for the second strand
                    # start_elem, end_elem = self.get_strand_element_indexes(parts[1])
                    # output_file.write(f"esel,s,,,{start_elem},{end_elem}\n")
                    # Write element selection for the first strand
                    output_file.write("allsel \n")
                    output_file.write("asel,none \n")
                    output_file.write("cmsel,none \n")
                    
                    output_file.write(f"cmsel,a,i_{self.stack_nr}_{parts[1]}   \n")
                    output_file.write(f"cmsel,a,m_{self.stack_nr}_{parts[1]}   \n")
                    output_file.write(f"cmsel,a,o_{self.stack_nr}_{parts[1]}   \n")
                    output_file.write(f"type,{self.element_number + 1000}\n")

                    output_file.write("allsel,belo,area\n")
                    output_file.write("esurf\n")
                    
                    # output_file.write("allsel,below,elem\n")
                    # output_file.write(f"type,{self.element_number + 1000}\n")
                    # output_file.write("esurf\n")
                    output_file.write("allsel\n\n\n")

                    self.element_number += 1
        except Exception as e:
            print(f"An error occurred while writing contacts: {e}")

# Example usage
if __name__ == "__main__":
    contact_nodes = ContactNodes("connections.txt", "contacts_strands.txt")
    contact_nodes.read_connections()
    contact_nodes.write_contacts()