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

def start_api(verbose=True):
    """
    Starts the ETABS API and creates an application to control it.

    Returns
    -------
    SapModel or int
        The SapModel object on success, or 1 on failure.
    """
    helper = comtypes.client.CreateObject('ETABSv1.Helper')

    myETABSObject = helper.GetObject("CSI.ETABS.API.ETABSObject")
    try:
        myETABSObject.ApplicationStart()
        if verbose:
            print("Etabs API started successfully")
    except:
        if verbose:
            print("Etabs API failed to connect")
        return 1
    SapModel = myETABSObject.SapModel

    return SapModel


def setup_sensor_group(SapModel, group_name, sensor_ids):
    """
    Creates a group in ETABS/SAP2000 and assigns the specified joint IDs to it.
    """
    print(f"Setting up group '{group_name}' with {len(sensor_ids)} sensors...")

    # 1. Unlock the model (assignments cannot be made to a locked model)
    SapModel.SetModelIsLocked(False)

    # 2. Define the new group
    SapModel.GroupDef.SetGroup(group_name)

    # 3. Assign each sensor joint to the group
    success_count = 0
    for j_id in sensor_ids:
        # The API requires the joint name as a string
        ret = SapModel.PointObj.SetGroupAssign(str(j_id), group_name)

        if ret == 0:
            success_count += 1
        else:
            print(f"  [Warning] Failed to add Joint ID '{j_id}' to group. Check if it exists.")

    print(f"Successfully added {success_count}/{len(sensor_ids)} joints to '{group_name}'.")


# ---------------------------------------------------------------------------
# Material Properties
# ---------------------------------------------------------------------------

def set_material(mat_name, E, SapModel):
    """Set isotropic material properties for a given material name."""
    SapModel.PropMaterial.SetMPIsotropic(mat_name, E, 0.3, 0.00001)


# ---------------------------------------------------------------------------
# Damage Pattern & Modal Analysis
# ---------------------------------------------------------------------------

def create_dp(SapModel, severity, group_name, E_i, mat_names, n_modes=12):
    """
    Create a damage pattern by modifying material stiffness, run analysis,
    and retrieve mode shapes and frequencies.

    Used by the flexibility-matrix pipeline to simulate damage scenarios.

    Parameters
    ----------
    SapModel : object
        The ETABS/SAP2000 model object.
    severity : list
        List of severity values (0–1) for each element/zone.
    group_name : str
        Name of the joint group to retrieve results from.
    E_i : float
        Initial Young's modulus.
    mat_names : list
        List of material names corresponding to each zone.
    n_modes : int
        Number of modes in the modal analysis (default 12).

    Returns
    -------
    tuple
        (Frequency, mode_shapes) arrays.
        mode_shapes has shape (3*n_joints, n_modes).
    """
    SapModel.SetModelIsLocked(False)
    # Edit materials according to damage severity
    for i, s in enumerate(severity):
        E = E_i * (1 - s)
        set_material(mat_names[i], E, SapModel)
    # Run analysis
    SapModel.Analyze.RunAnalysis()

    # Select only the Modal case for output
    SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()

    # Auto-detect the modal case name
    NumberNames, case_names, ret_code = SapModel.LoadCases.GetNameList()
    modal_case_name = "Modal"  # Default fallback
    if ret_code == 0:
        for name in case_names:
            case_type, sub_type, design_type, design_opt, auto, ret = \
                SapModel.LoadCases.GetTypeOAPI_1(name)
            if case_type == 3:  # 3 = Modal Cases
                modal_case_name = name
                break

    SapModel.Results.Setup.SetCaseSelectedForOutput(modal_case_name)

    # ETABS fix: tell the API to output ALL modes (SAP2000 does this by default)
    SapModel.Results.Setup.SetOptionModeShape(1, n_modes, True)

    # Group-based ModeShape extraction (ObjectElm=2 → GroupElm)
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

    NumberResults, Obj, Elm, LoadCase, StepType, StepNum, \
        U1, U2, U3, R1, R2, R3, n = SapModel.Results.ModeShape(
            group_name, 2, NumberResults, Obj, Elm, LoadCase,
            StepType, StepNum, U1, U2, U3, R1, R2, R3
        )

    [_, _, _, _, _, _, Frequency, _, _] = SapModel.Results.ModalPeriod(
        0, '', '', [], [], [], [], []
    )

    U1 = np.array(U1).reshape((n_modes, -1))
    U2 = np.array(U2).reshape((n_modes, -1))
    U3 = np.array(U3).reshape((n_modes, -1))
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


def extract_and_save_mode_shapes(SapModel, group_name, E_i, mat_names,
                                 n_elements, out_csv_path, n_modes=12):
    """
    Extract undamaged mode shapes for all candidate sensors and save to CSV.

    This is a dedicated extraction function — it does NOT reuse create_dp
    (which is designed for damage-pattern simulation + flexibility matrices).

    Key ETABS difference vs SAP2000:
        ETABS requires an explicit call to
            SapModel.Results.Setup.SetOptionModeShape(1, n_modes, True)
        to include ALL modes in the output. Without it, ETABS defaults to
        returning only 1 mode, whereas SAP2000 returns all modes by default.

    Parameters
    ----------
    SapModel : object
        The ETABS model object.
    group_name : str
        Name of the joint group containing all candidate sensors.
    E_i : float
        Initial (undamaged) Young's modulus.
    mat_names : list[str]
        Material names for each zone.
    n_elements : int
        Number of zones / elements.
    out_csv_path : str
        Path to save the mode shape CSV.
    n_modes : int
        Number of modes in the modal analysis (default 12).

    Saves
    -----
    CSV file with shape (n_modes, 3*n_sensors) ready for SNPO reshaping
    to (n_modes, 3, n_sensors).
    """
    print(f"Extracting mode shapes for group '{group_name}'...")

    # 1. Set undamaged material properties and run analysis
    SapModel.SetModelIsLocked(False)
    for i in range(n_elements):
        set_material(mat_names[i], E_i, SapModel)   # 0% damage → full E
    SapModel.Analyze.RunAnalysis()

    # 2. Configure output: select only the Modal case
    SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()

    # Auto-detect the modal case name
    NumberNames, case_names, ret_code = SapModel.LoadCases.GetNameList()
    modal_case_name = "Modal"  # Default fallback
    if ret_code == 0:
        for name in case_names:
            case_type, sub_type, design_type, design_opt, auto, ret = \
                SapModel.LoadCases.GetTypeOAPI_1(name)
            if case_type == 3:  # 3 = Modal Cases
                modal_case_name = name
                break

    SapModel.Results.Setup.SetCaseSelectedForOutput(modal_case_name)

    # 3. *** CRITICAL ETABS FIX ***
    #    Tell ETABS to output ALL mode shapes (modes 1 through n_modes).
    #    Without this call ETABS returns only 1 mode by default.
    #    SAP2000 does not need this — it returns all modes automatically.
    SapModel.Results.Setup.SetOptionModeShape(1, n_modes, True)

    # 4. Group-based ModeShape extraction (matching old SAP2000 approach)
    #    ObjectElm=2 → GroupElm: results for every joint in the group
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

    NumberResults, Obj, Elm, LoadCase, StepType, StepNum, \
        U1, U2, U3, R1, R2, R3, ret = SapModel.Results.ModeShape(
            group_name, 2, NumberResults, Obj, Elm, LoadCase,
            StepType, StepNum, U1, U2, U3, R1, R2, R3
        )

    print(f"   ModeShape returned {NumberResults} results (ret={ret})")

    if NumberResults == 0 or ret != 0:
        raise RuntimeError(
            f"ModeShape extraction failed for group '{group_name}' "
            f"(NumberResults={NumberResults}, ret={ret})."
        )

    # 5. Reshape flat arrays → (n_modes, n_joints) per direction
    U1 = np.array(U1).reshape((n_modes, -1))
    U2 = np.array(U2).reshape((n_modes, -1))
    U3 = np.array(U3).reshape((n_modes, -1))
    n_joints = U1.shape[1]

    # 6. Build the mode-shape matrix expected by SNPO:
    #    Concatenate directions → (n_modes, 3*n_joints)
    mode_shapes_for_snpo = np.concatenate([U1, U2, U3], axis=1)

    # 7. Save
    np.savetxt(out_csv_path, mode_shapes_for_snpo, delimiter=",")

    print(f"✅ Mode shapes successfully saved to {out_csv_path}")
    print(f"   Shape: {mode_shapes_for_snpo.shape}")
    print(f"   (n_modes={n_modes}, 3*n_sensors={3*n_joints}, "
          f"n_sensors={n_joints})")