Sensitivity Analysis with KIKA
===============================

This tutorial covers how to use KIKA for sensitivity analysis using MCNP perturbation calculations.

Overview of Sensitivity Analysis
--------------------------------

Sensitivity analysis using MCNP involves:

1. Setting up perturbation calculations in your MCNP input file
2. Running the MCNP calculation to get perturbation results
3. Processing the results to obtain sensitivity coefficients
4. Analyzing and visualizing the sensitivity profiles

KIKA provides tools for the last two steps in this process.

Computing Sensitivity Coefficients
----------------------------------

Start by computing sensitivity coefficients from MCNP perturbation results:

.. code-block:: python

   from kika.sensitivities import compute_sensitivity
   
   # Compute sensitivity data for a specific nuclide
   sensitivity = compute_sensitivity(
       input_path='path/to/input.i',
       mctal_path='path/to/mctal',
       tally=4,  # Tally number
       zaid=26056,  # ZAID for Fe-56
       label='Fe-56'  # Human-readable label
   )
   
   # Print sensitivity information
   print(sensitivity)

Accessing Sensitivity Data
--------------------------

Access the calculated sensitivity coefficients:

.. code-block:: python

   # Get available energy groups 
   print(f"Available energy groups: {sensitivity.energies}")
   
   # Get available reactions
   print(f"Available reactions: {sensitivity.reactions}")
   
   # Access sensitivity data for a specific energy group and reaction
   energy_group = '1.00e+00_3.00e+00'  # Example energy group
   reaction = 2  # MT=2 for elastic scattering
   
   sens_data = sensitivity.data[energy_group][reaction]
   print(sens_data)
   
   # Access sensitivity values and errors
   energy_boundaries = sens_data.energy_boundaries
   sens_values = sens_data.sensitivity
   error_values = sens_data.error
   
   # Calculate total sensitivity for this reaction in this energy range
   total_sens = sum(sens_values)
   print(f"Total sensitivity for reaction {reaction} in {energy_group}: {total_sens:.6e}")

Understanding SensitivityData Structure
---------------------------------------

The `SensitivityData` object contains sensitivity coefficients organized by detector energy range and reaction number:

.. code-block:: python

   # Basic attributes
   print(f"Nuclide: {sensitivity.nuclide}")
   print(f"Tally ID: {sensitivity.tally_id}")
   print(f"Available energy ranges: {sensitivity.energies}")
   print(f"Available reaction numbers: {sensitivity.reactions}")
   print(f"Number of perturbation energy bins: {len(sensitivity.pert_energies) - 1}")

   # Access the Coefficients object for a specific energy range and reaction
   first_energy = sensitivity.energies[0]
   first_reaction = sensitivity.reactions[0]
   sens_coeffs = sensitivity.data[first_energy][first_reaction]
   
   # Display coefficient properties
   print(f"Energy range: {sens_coeffs.energy}")
   print(f"Reaction number: {sens_coeffs.reaction}")
   print(f"Unperturbed result (R₀): {sens_coeffs.r0:.6e} ± {sens_coeffs.e0*100:.4f}% (relative)")
   print(f"Perturbation energy boundaries: {sens_coeffs.pert_energies}")

Visualizing Sensitivity Profiles
--------------------------------

Create visualizations of your sensitivity profiles:

.. code-block:: python

   # Plot sensitivity profile for an energy group and reaction
   sensitivity.plot_sensitivity(energy='1.00e+00_3.00e+00', reaction=2)
   
   # Plot multiple reactions for comparison
   sensitivity.plot_sensitivity(energy='1.00e+00_3.00e+00', reaction=[1, 2, 102])
   
   # Set energy limits for better visualization
   import matplotlib.pyplot as plt
   fig, ax = plt.subplots(figsize=(12, 6))
   sensitivity.plot_sensitivity(energy='1.00e+00_3.00e+00', reaction=2, ax=ax, xlim=[1e-11, 20])
   plt.grid(True, which='both', linestyle='--', alpha=0.7)

Working with Taylor Coefficients
--------------------------------

KIKA can analyze nonlinearity in sensitivity coefficients using Taylor coefficients:

.. code-block:: python

   # Check if Taylor coefficients are available
   if sensitivity.coefficients:
       # Access and display Taylor coefficient data
       first_energy = next(iter(sensitivity.coefficients.keys()))
       first_reaction = next(iter(sensitivity.coefficients[first_energy].keys()))
       taylor_coeffs = sensitivity.coefficients[first_energy][first_reaction]
       print(taylor_coeffs)
       
       # Visualize nonlinearity using Taylor ratios
       sensitivity.plot_ratio(
           energy=sensitivity.energies[2],
           reaction=[2],  # Plot only reaction MT=2
           top_n=5  # Show top 5 energy bins with highest nonlinearity
       )
       
       # Plot perturbed response for comparison
       sensitivity.plot_perturbed_response(
           energy=sensitivity.energies[2],
           reaction=[2],
           p_range=(-20, 20),  # Perturbation range in percent
           top_n=3,  # Show top 3 energy bins with highest nonlinearity
       )
       
       # Show the difference between approximations
       sensitivity.plot_second_order_contribution(
           energy=sensitivity.energies[2],
           reaction=[2],
           p_range=(-10, 10),
           top_n=3,
       )
   else:
       print("No Taylor coefficient data available. Second-order perturbations were not calculated.")

Converting to DataFrames
------------------------

For advanced analysis, convert sensitivity data to pandas DataFrames:

.. code-block:: python

   # Convert to DataFrame
   sens_df = sensitivity.to_dataframe()
   
   # Display the DataFrame
   print(sens_df.head())

Comparing Multiple Sensitivity Datasets
---------------------------------------

KIKA provides tools for comparing sensitivity profiles from different nuclides:

.. code-block:: python

   # Assuming you have multiple sensitivity datasets
   sens_fe56 = compute_sensitivity(
       input_path='path/to/fe56_input.i',
       mctal_path='path/to/fe56_mctal',
       tally=4,
       zaid=26056,
       label="Fe-56 Sensitivity"
   )
   
   sens_h1 = compute_sensitivity(
       input_path='path/to/h1_input.i',
       mctal_path='path/to/h1_mctal',
       tally=4,
       zaid=1001,
       label="H-1 Sensitivity"
   )
   
   # Compare sensitivity profiles
   from kika.sensitivities import plot_sens_comparison
   
   plot_sens_comparison(
       sens_list=[sens_fe56, sens_h1],
       energy=sens_fe56.energies[2],
       reactions=[1, 2],  # Compare total and elastic scattering reactions
       xlim=(0, 3)  # Limit x-axis to 0-3 MeV
   )

Creating SDF Files for SCALE
----------------------------

Generate Sensitivity Data Files (SDF) compatible with SCALE:

.. code-block:: python

   from kika.sensitivities.sdf import create_sdf_data
   
   # Create SDF data from sensitivity datasets
   sdf_data = create_sdf_data(
       sensitivity_data_list=[sens_fe56, sens_h1],
       energy=sens_fe56.energies[2],
       title="Example SDF Dataset"
   )
   
   # Create SDF data with specific reactions for each dataset
   sdf_filtered = create_sdf_data(
       sensitivity_data_list=[
           (sens_fe56, [1, 2]),  # Only include reactions 1 and 2 for Fe-56
           (sens_h1, [1])        # Only include reaction 1 for H-1
       ],
       energy=sens_fe56.energies[2],
       title="Filtered SDF Dataset"
   )
   
   # Access reaction data within the SDF object
   reaction_data = sdf_data.data[0]  # First reaction in the dataset
   print(f"Accessing first reaction: {reaction_data.nuclide} {reaction_data.reaction_name} (MT={reaction_data.mt})")
   print(f"Total sensitivity: {sum(reaction_data.sensitivity):.6e}")
   
   # Export SDF data to a file
   sdf_data.write_file(output_dir='path/to/output')
   
   # Group inelastic reactions (MT 51-91 into MT 4) and write to file
   sdf_data.group_inelastic_reactions(replace=True)
   sdf_data.write_file(output_dir='path/to/output')
