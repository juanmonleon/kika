Working with MCNP Input Files
=============================

This tutorial covers how to use KIKA to work with MCNP input files, including parsing, manipulation, and generation.

Parsing Input Files
-------------------

The first step in working with MCNP input files is to parse them into Python objects:

.. code-block:: python

   from kika import read_mcnp
   
   # Parse an MCNP input file
   input_file = read_mcnp('path/to/input.i')
   
   # Print a summary of the input file
   input_file.print_summary()

Accessing Input Components
--------------------------

Once parsed, you can access various components of the input file:

.. code-block:: python

   # Access cells
   cells = input_file.cells
   
   # Get a specific cell
   cell_5 = cells[5]
   print(f"Cell 5 material: {cell_5.material}")
   print(f"Cell 5 density: {cell_5.density}")
   
   # Access surfaces
   surfaces = input_file.surfaces
   
   # Get a specific surface
   surf_10 = surfaces[10]
   print(f"Surface 10 type: {surf_10.type}")
   print(f"Surface 10 parameters: {surf_10.params}")
   
   # Access materials
   materials = input_file.materials
   
   # Get a specific material
   mat_100 = materials.mat[100]
   print(f"Material 100 nuclides: {list(mat_100.nuclide.keys())}")

Working with Materials
----------------------

KIKA provides detailed access to material definitions:

.. code-block:: python

   # Print materials summary
   materials.print_summary()
   
   # Access a specific material
   mat = materials.mat[100]
   
   # Get all material IDs
   material_ids = list(materials.mat.keys())
   print(f"Available material IDs: {material_ids}")
   
   # Get material composition
   for zaid, nuclide in mat.nuclide.items():
       fraction_type = "Weight" if nuclide.fraction < 0 else "Atomic"
       print(f"ZAID: {zaid}, Element: {nuclide.element}, "
             f"Fraction: {abs(nuclide.fraction):.6e} ({fraction_type})")
   
   # Accessing nuclide details
   zaid = list(mat.nuclide.keys())[0]  # First nuclide
   nuclide = mat.nuclide[zaid]
   print(f"Element symbol: {nuclide.element}")
   print(f"Neutron library: {nuclide.nlib if nuclide.nlib else 'Using material default'}")
   print(f"Photon library: {nuclide.plib if nuclide.plib else 'Using material default'}")

Converting Between Atomic and Weight Fractions
----------------------------------------------

MCNP uses positive values for atomic fractions and negative values for weight fractions.
KIKA allows converting between these formats:

.. code-block:: python

   # Create a copy of a material
   new_material = mat.copy(new_id=mat.id+1000)
   
   # Convert to atomic fractions
   new_material.to_atomic_fraction()
   print(new_material)
   
   # Convert to weight fractions
   new_material.to_weight_fraction()
   print(new_material)
   
   # Convert all materials at once
   materials.to_atomic_fractions()  # Convert all to atomic fractions
   materials.to_weight_fractions()  # Convert all to weight fractions

Converting Natural Elements to Isotopes
---------------------------------------

KIKA provides functionality to convert materials containing natural elements (ZAIDs ending in '00') 
into their constituent isotopes based on natural abundances:

.. code-block:: python

   # Create a material with natural elements
   natural_mat = kika.mcnp.material.Mat(id=200, nlib="80c")
   
   # Add natural carbon (ZAID 6000)
   natural_mat.add_nuclide(zaid=6000, fraction=0.5)
   
   # Add natural oxygen (ZAID 8000)
   natural_mat.add_nuclide(zaid=8000, fraction=0.5)
   
   print("Material with natural elements:")
   print(natural_mat)
   
   # Convert all natural elements to isotopes
   natural_mat.expand_natural_elements()
   
   print("Material after converting natural elements to isotopes:")
   print(natural_mat)
   
   # You can also convert only specific natural elements:
   another_mat = kika.mcnp.material.Mat(id=300, nlib="80c")
   another_mat.add_nuclide(zaid=6000, fraction=0.3)  # Natural carbon
   another_mat.add_nuclide(zaid=8000, fraction=0.7)  # Natural oxygen
   
   # Convert only carbon to isotopes
   another_mat.expand_natural_elements(zaid_to_expand=6000)
   
   print("Material after converting only carbon to isotopes:")
   print(another_mat)

Creating New Materials
----------------------

You can create new materials from scratch using the KIKA API:

.. code-block:: python

   from kika.mcnp.material import Mat, Materials
   
   # Create a new material - water (H2O)
   water = Mat(id=100, nlib="80c")
   
   # Add hydrogen (atomic fraction 2/3)
   water.add_nuclide(zaid=1001, fraction=2/3)
   
   # Add oxygen (atomic fraction 1/3)
   water.add_nuclide(zaid=8016, fraction=1/3)
   
   # Create a materials collection with our new material
   new_materials = Materials()
   new_materials.add_material(water)
   
   # Convert to weight fractions
   water.to_weight_fraction()

Modifying Input Files
---------------------

You can also modify input files programmatically:

.. code-block:: python

   # Change a cell's material
   input_file.cells[5].material = 200
   
   # Modify a surface parameter
   input_file.surfaces[10].params[0] = 1.5
   
   # Add a nuclide to a material
   from kika.mcnp.material import Nuclide
   materials.mat[100].add_nuclide(zaid=1001, fraction=2.0)
   
   # TODO: Write the modified input to a new file
   # This functionality is planned for future releases

Working with Perturbations
--------------------------

KIKA provides tools for analyzing MCNP perturbation cards:

.. code-block:: python

   # Access perturbations
   perturbations = input_file.perturbation
   
   # Get a list of all perturbation IDs
   pert_ids = list(perturbations.pert.keys())
   print(f"Available perturbation IDs: {pert_ids}")
   
   # Access a specific perturbation
   pert_id = pert_ids[0]  
   pert = perturbations.pert[pert_id]
   print(pert)
   
   # Convert to pandas DataFrame for easier analysis
   pert_df = perturbations.to_dataframe()
   
   # Extract unique reaction types and energy bins
   reactions = perturbations.reactions
   energies = perturbations.pert_energies

Generating Perturbation Cards
-----------------------------

KIKA can help generate perturbation cards for sensitivity studies:

.. code-block:: python

   from kika.mcnp.pert_generator import generate_material_perturbations
   
   perturbations = generate_material_perturbations(
       material_id=300,
       cell_ids=[3, 5, 7, 9],
       zaid=26056,
       rxns=[1, 2, 102],  # MT numbers for reactions
       energy_groups=[1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 20],
       methods=[1, 2, 3, -1, -2, -3]
   )
   
   for pert in perturbations[:5]:  # Print first 5 perturbations
       print(pert)
