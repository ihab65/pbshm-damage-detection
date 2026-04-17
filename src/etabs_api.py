"""
etabs_api.py — The Simulation Engine

All functions interacting with comtypes and physical structural math.
Handles ETABS/SAP2000 API communication, material property manipulation,
modal analysis, flexibility matrix computation, and dataset generation.
"""

import os
import pickle
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import comtypes.client
except ImportError:
    pass


# ---------------------------------------------------------------------------
# API Connection
# ---------------------------------------------------------------------------

def start_API(verbose=True):
    """
    Starts the SAP2000 API and creates an application to control it.

    Returns
    -------
    SapModel or int
        The SapModel object on success, or 1 on failure.
    """
    helper = comtypes.client.CreateObject('SAP2000v1.Helper')
    helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)

    mySapObject = helper.GetObject("CSI.SAP2000.API.SapObject")
    try:
        mySapObject.ApplicationStart()
        if verbose:
            print("SAP2000 API started successfully")
    except:
        if verbose:
            print("SAP2000 API failed to connect")
        return 1

    SapModel = mySapObject.SapModel
    return SapModel


# ---------------------------------------------------------------------------
# Material Properties
# ---------------------------------------------------------------------------

def set_material(mat_name, E, SapModel):
    """Set isotropic material properties for a given material name."""
    SapModel.PropMaterial.SetMPIsotropic(mat_name, E, 0.3, 0.00001)


# ---------------------------------------------------------------------------
# Damage Pattern & Modal Analysis
# ---------------------------------------------------------------------------

def create_dp(SapModel, severity, group_name, E_i, mat_names):
    """
    Create a damage pattern by modifying material stiffness, run analysis,
    and retrieve mode shapes and frequencies.

    Parameters
    ----------
    SapModel : object
        The SAP2000 model object.
    severity : list
        List of severity values (0–1) for each element/zone.
    group_name : str
        Name of the joint group to retrieve results from.
    E_i : float
        Initial Young's modulus.
    mat_names : list
        List of material names corresponding to each zone.

    Returns
    -------
    tuple
        (Frequency, mode_shapes) arrays.
    """
    SapModel.SetModelIsLocked(False)
    # Edit materials
    for i, s in enumerate(severity):
        E = E_i * (1 - s)
        set_material(mat_names[i], E, SapModel)
    # Run analysis
    SapModel.Analyze.RunAnalysis()
    # Get results
    SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
    SapModel.Results.Setup.SetCaseSelectedForOutput("MODAL")

    NumberResults = 0
    Obj = ""
    Elm = ""
    LoadCase = ""
    StepType = ""
    StepNum = []
    U1 = []
    U2 = []
    U3 = []
    R1 = []
    R2 = []
    R3 = []
    # Retrieve mode shape results
    NumberResults, Obj, Elm, LoadCase, StepType, StepNum, U1, U2, U3, R1, R2, R3, n = \
        SapModel.Results.ModeShape(
            group_name, 2, NumberResults, Obj, Elm, LoadCase,
            StepType, StepNum, U1, U2, U3, R1, R2, R3
        )
    [_, _, _, _, _, _, Frequency, _, _] = SapModel.Results.ModalPeriod(
        0, '', '', [], [], [], [], []
    )
    U1 = np.array(U1).reshape((12, -1))
    U2 = np.array(U2).reshape((12, -1))
    U3 = np.array(U3).reshape((12, -1))
    mode_shapes = np.concatenate([U1.T, U2.T, U3.T], axis=0)

    return Frequency, mode_shapes


# ---------------------------------------------------------------------------
# Flexibility Matrix & Damage Indicator
# ---------------------------------------------------------------------------

def flexibility_matrix(Frequency, mode_shapes):
    """
    Compute the modal flexibility matrix.

    Parameters
    ----------
    Frequency : array-like
        Natural frequencies from modal analysis.
    mode_shapes : ndarray
        Mode shape matrix.

    Returns
    -------
    ndarray
        The flexibility matrix.
    """
    f = 2 * np.pi * np.array(Frequency)
    f = np.diag(f)

    t = mode_shapes
    np.dot(f, t.T)
    Flexibility = np.dot(t, np.dot(f, t.T))
    return Flexibility


def delta_fmax(Damaged_Flexibility, notDamaged_Flexibility):
    """
    Compute the maximum absolute change in flexibility between
    damaged and undamaged states (damage indicator).
    """
    return np.max(np.abs(Damaged_Flexibility - notDamaged_Flexibility), axis=1)


# ---------------------------------------------------------------------------
# Dataset Generation
# ---------------------------------------------------------------------------

def compute_and_save_not_damaged(SapModel, group_name, E_i, mat_names,
                                 n_elements, out_path):
    """
    Run the undamaged scenario and save the baseline flexibility matrix.
    """
    print("Running the not damaged scenario...")
    Frequency_0, mode_shapes_0 = create_dp(
        SapModel, n_elements * [0], group_name, E_i, mat_names
    )
    notDamaged_Flexibility = flexibility_matrix(Frequency_0, mode_shapes_0)

    with open(out_path, "wb") as f:
        pickle.dump(notDamaged_Flexibility, f)
    print(f"✅ notDamaged_Flexibility saved to {out_path}")


def create_dataset(batch_array, SapModel, group_name, E_i, mat_names,
                   not_damaged_path):
    """
    Generate damage indicator dataset from a batch of severity arrays.
    """
    # Load saved notDamaged_Flexibility
    with open(not_damaged_path, "rb") as f:
        notDamaged_Flexibility = pickle.load(f)

    DI = []
    for severity in tqdm(batch_array, desc='Generating batch'):
        Frequency, mode_shapes = create_dp(
            SapModel, severity, group_name, E_i, mat_names
        )
        Damaged_Flexibility = flexibility_matrix(Frequency, mode_shapes)
        Damage_indicator = delta_fmax(Damaged_Flexibility, notDamaged_Flexibility)
        DI.append(Damage_indicator)

    return DI


def run_batches(dataset_array, batch_size, SapModel, group_name, E_i,
                mat_names, not_damaged_file, output_dir):
    """
    Orchestrate batch processing: split dataset_array into batches,
    generate damage indicators, and save each batch to CSV.
    """
    all_csv_paths = []
    n_batches = int(np.ceil(len(dataset_array) / batch_size))

    for i in range(n_batches):
        batch = dataset_array[i * batch_size:(i + 1) * batch_size]
        print(f"\nProcessing batch {i + 1}/{n_batches} "
              f"with {len(batch)} samples...")

        # Generate dataset
        damage_indicators = create_dataset(
            batch, SapModel, group_name, E_i, mat_names, not_damaged_file
        )

        # Save batch to CSV
        batch_df = pd.DataFrame(damage_indicators)
        batch_path = os.path.join(output_dir, f"batch_{i + 1:03d}.csv")
        batch_df.to_csv(batch_path, index=False)
        all_csv_paths.append(batch_path)

    return all_csv_paths


def generate_unique_combinations(n=5000, step=0.05, n_elements=3):
    """
    Generate an (n × n_elements) array of unique rows,
    where each element is selected from np.arange(0, 0.9, step).

    Parameters
    ----------
    n : int
        Number of unique rows to return.
    step : float
        Step size for generating the value range.
    n_elements : int
        Number of elements (zones) per combination.

    Returns
    -------
    np.ndarray
        Array of shape (n, n_elements) with unique rows.
    """
    values = np.arange(0, 0.9, step)
    all_combinations = np.array(
        list(itertools.product(values, repeat=n_elements))
    )

    if n > len(all_combinations):
        raise ValueError(
            f"Requested {n} combinations, but only "
            f"{len(all_combinations)} are available."
        )

    np.random.shuffle(all_combinations)
    result = all_combinations[:n]
    return result
