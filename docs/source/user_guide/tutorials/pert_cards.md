Setting up Sensitivity Analysis with PERT Cards
===============================================

This tutorial demonstrates how to prepare MCNP input files for sensitivity analysis using KIKA's perturbation card generation capabilities.

Overview
--------

The process of setting up sensitivity analysis involves:

1. Reading an existing MCNP input file
2. Creating a perturbed material for a specific nuclide
3. Generating PERT cards for sensitivity calculations
4. Preparing the input for MCNP execution

After completing these steps, you'll have an MCNP input file ready for sensitivity analysis calculations.

Setting up the Environment
--------------------------

Start by importing the necessary libraries:

.. code-block:: python

   import kika
   from pathlib import Path

   # Setup paths
   repo_root = Path.cwd().resolve().parent
   data_dir = repo_root / 'examples' / 'data'
   
   # Input file path
   inputfile = data_dir / 'inputfile_example_0.i'
   
   # Create a working copy of the input file
   working_file = data_dir / 'inputfile_example_0_working.i'
   
   with open(inputfile, 'r') as f_in, open(working_file, 'w') as f_out:
       f_out.write(f_in.read())
       print(f"Created working copy: {working_file}")

Reading the MCNP Input File
---------------------------

The first step is to read the MCNP input file to understand its structure and examine the materials:

.. code-block:: python

   # Read the input file
   input_data = kika.read_mcnp(working_file)
   
   # Check if our target material exists
   material_number = 300000
   if material_number in input_data.materials.mat:
       print(f"Material {material_number} found in the input file")
       
       # Display material composition
       print(input_data.materials.mat[material_number])

Creating a Perturbed Material
-----------------------------

For sensitivity analysis, we need to create a perturbed version of the material of interest:

.. code-block:: python

   # Perturb Fe-56 of the material
   kika.perturb_material(
       inputfile=working_file,
       material_number=300000,
       density=-7.85,
       nuclide=26056,
       pert_mat_id=None,  # Will use default: material_number*100 + 1
       in_place=True,
       format='atomic'  # Output in atomic fractions
   )

The ``perturb_material`` function creates a new material in the MCNP input file with a 100% increase in the specified nuclide (Fe-56 in this example). The new material ID is set to material_number*100 + 1 by default (30000001 in this case).

Generating PERT Cards
---------------------

Now we'll generate the PERT cards needed for sensitivity analysis:

.. code-block:: python

   # Define cells, reactions and energy grid
   cells = [3, 5, 7, 9]  # Cells where perturbation will be applied
   reactions = [1, 2, 4, 51, 102]  # MT numbers of reactions
   
   # We'll use the scale44 energy grid
   print(f"Energy grid has {len(kika.energyGrids.scale44)} points, creating {len(kika.energyGrids.scale44)-1} energy bins")
   
   # Generate PERT cards
   kika.generate_PERTcards(
       inputfile=working_file,
       cell=cells,
       density=0.01,  # 1% perturbation is typical for sensitivity analysis
       reactions=reactions,
       energies=kika.energyGrids.scale44,
       material=material_number * 100 + 1,  # Use the perturbed material ID
       order=2,  # Generate both first and second order perturbations
       errors=False,  # Don't generate error method cards
       in_place=True  # Add PERT cards to the input file
   )

Key parameters in the ``generate_PERTcards`` function:

- ``cell``: List of cells where the perturbation will be applied
- ``density``: Perturbation magnitude (typically 0.01 or 1%)
- ``reactions``: MT numbers of nuclear reactions to analyze
- ``energies``: Energy grid for perturbation (built-in options available through ``kika.energyGrids``)
- ``material``: Material ID of the perturbed material
- ``order``: Order of perturbation (1 for first-order, 2 for both first and second-order)

Examining the Generated PERT Cards
----------------------------------

After generating the cards, you can inspect them in the MCNP input file:

.. code-block:: python

   # Read back the file to examine the PERT cards
   modified_input_data = kika.read_mcnp(working_file)
   
   # Display the PERT cards
   print(modified_input_data.perturbation)

Running MCNP with Prepared Input
--------------------------------

The prepared input file with perturbed material and PERT cards is now ready for MCNP execution. After running MCNP, you'll have output files including a MCTAL file containing the perturbation results.

Analyzing Sensitivity Results
-----------------------------

After running MCNP with the prepared input file, you can analyze the results using KIKA's sensitivity analysis tools:

.. code-block:: python

   # Compute sensitivity coefficients for Fe-56
   sens_fe56 = kika.compute_sensitivity(
       inputfile="inputfile_example_0_working.i",
       mctalfile="inputfile_example_0.m", 
       tally=4,  # Tally number to analyze
       zaid=26056,  # Fe-56
       label="Fe-56 Sensitivity"
   )
   
   # Visualize sensitivity profiles
   sens_fe56.plot_sensitivity(
       energy="integral",
       reaction=[1, 2, 4, 51, 102],
       xlim=(0, 10)  # Limit x-axis to 0-10 MeV
   )

For more detailed information on analyzing sensitivity results, see the :doc:`sensitivity_analysis` tutorial.