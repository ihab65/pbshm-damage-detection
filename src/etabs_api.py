"""
etabs_api.py — The Simulation Engine

All functions interacting with comtypes and physical structural math.
Handles ETABS/SAP2000 API communication, material property manipulation,
modal analysis, flexibility matrix computation, and dataset generation.
"""

import os
import shutil
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
# Kill Switch — signal-based because comtypes __del__ swallows KeyboardInterrupt
# ---------------------------------------------------------------------------
import signal

_kill_requested = False


def _handle_sigint(signum, frame):
    """Signal handler that sets a flag instead of raising KeyboardInterrupt."""
    global _kill_requested
    _kill_requested = True
    print("\n⚠️  Ctrl+C received — will stop after the current scenario finishes...")


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

def launch_etabs(master_model_path, verbose=True):
    """
    Launch a brand-new ETABS process, copy the master model, and open it.
    Includes a pre-launch 'Ghostbuster' to prevent WinError -2146959355.
    """
    import time as _time
    import os
    import shutil
    import subprocess
    import comtypes.client

    master_model_path = os.path.abspath(master_model_path)
    if not os.path.isfile(master_model_path):
        raise FileNotFoundError(f"Master model not found: {master_model_path}")

    # --- 0. THE GHOSTBUSTER (Clear the RAM and COM ports) ---
    # Forcefully kill any lingering ETABS.exe processes from previous failed runs
    if verbose:
        print("   Sweeping background for zombie ETABS processes...")
    try:
        # /F = force, /IM = image name, /T = kill child processes
        subprocess.run(['taskkill', '/F', '/IM', 'ETABS.exe', '/T'], 
                       stdout=subprocess.DEVNULL, 
                       stderr=subprocess.DEVNULL)
        _time.sleep(2) # Give Windows a moment to release the file locks
    except Exception:
        pass # If it fails, it means no ghosts were found, which is fine!

    # --- 1. Copy the master .EDB to a temp working copy ---
    model_dir = os.path.dirname(master_model_path)
    temp_model_path = os.path.join(model_dir, "_temp_batch_run.EDB")
    shutil.copyfile(master_model_path, temp_model_path)
    if verbose:
        print(f"   Copied master model → {temp_model_path}")

    # --- 2. Launch a fresh ETABS process ---
    if verbose:
        print("   Spawning new ETABS COM Object...")
    try:
        # The strictly correct CSI method for instantiating a new process
        helper = comtypes.client.CreateObject('ETABSv1.Helper')
        myETABSObject = helper.CreateObjectProgID("CSI.ETABS.API.ETABSObject")
        myETABSObject.ApplicationStart()
    except Exception as e:
        raise RuntimeError(f"Failed to spawn ETABS COM Object: {e}")
    
    _time.sleep(8) # Give the heavy GUI time to load
    SapModel = myETABSObject.SapModel

    # --- 3. KICKSTART THE ENGINE ---
    SapModel.InitializeNewModel()

    # --- 4. Open the temp model (with Retry Logic) ---
    if verbose:
        print("   Attempting to open the model...")
        
    max_retries = 3
    ret = -1
    for attempt in range(1, max_retries + 1):
        ret = SapModel.File.OpenFile(temp_model_path)
        if ret == 0:
            break
            
        if verbose:
            print(f"   [Warning] OpenFile failed (Attempt {attempt}/{max_retries}). Retrying in 5s...")
        _time.sleep(5)

    if ret != 0:
        myETABSObject.ApplicationExit(False)
        raise RuntimeError(f"ETABS completely failed to open model after {max_retries} attempts.")

    if verbose:
        print(f"   ✅ Model opened successfully: {os.path.basename(temp_model_path)}")

    return SapModel, temp_model_path

def stop_api(verbose=True):
    """
    Gracefully shut down ETABS to free memory.

    Uses the COM helper to obtain the running ETABS object and calls
    ``ApplicationExit(False)`` (False = don't save).  Silently ignores
    errors if ETABS has already closed.
    """
    try:
        helper = comtypes.client.CreateObject('ETABSv1.Helper')
        myETABSObject = helper.GetObject("CSI.ETABS.API.ETABSObject")
        myETABSObject.ApplicationExit(False)
        if verbose:
            print("   ETABS shut down successfully.")
    except Exception as e:
        if verbose:
            print(f"   ETABS shutdown notice: {e}")


def _cleanup_temp_files(temp_model_path, verbose=True):
    """
    Remove the temp ``.EDB`` **and** all auxiliary files ETABS creates
    alongside it (e.g. ``.LOG``, ``.OUT``, ``.msh``, ``.K_*``, etc.).

    Works by globbing every file that shares the same stem as the EDB.
    """
    import glob as _glob

    if not temp_model_path:
        return

    stem = os.path.splitext(temp_model_path)[0]  # e.g. ...\_temp_batch_run
    pattern = stem + ".*"
    files = _glob.glob(pattern)

    if not files:
        return

    removed = 0
    for f in files:
        try:
            os.remove(f)
            removed += 1
        except Exception:
            pass

    if verbose:
        print(f"   Cleaned up {removed} temp file(s).")


def run_batches_with_reboot(dataset_array, batch_size, group_name, E_i,
                            mat_names, not_damaged_file, output_dir,
                            master_model_path, n_modes=12, cooldown=10):
    """
    Resume-safe batch runner that **restarts ETABS between every batch**
    to counteract the COM memory-leak slowdown.

    How it works
    ------------
    1. Scans *all* existing ``batch_*.csv`` files (from any previous run,
       regardless of the batch size used) and counts the total number of
       completed scenario rows.
    2. Slices ``dataset_array`` to skip those rows — so switching from
       ``batch_size=100`` to ``batch_size=20`` mid-campaign is perfectly
       safe.
    3. Uses a ``batch_part2_XXXX.csv`` naming convention so that old files
       are never overwritten.
    4. Between batches: kills ETABS → cleans up temp model → sleeps
       → launches a brand-new ETABS process with a fresh model copy.

    Parameters
    ----------
    dataset_array : ndarray
        The full master damage-combination array (n_total × n_zones).
    batch_size : int
        Number of scenarios per batch (e.g. 20).
    group_name : str
        ETABS joint-group name (e.g. ``"OPT_20"``).
    E_i : float
        Undamaged Young's modulus.
    mat_names : list[str]
        Material names for each zone.
    not_damaged_file : str
        Path to the pickled baseline flexibility matrix.
    output_dir : str
        Directory where batch CSVs are saved.
    master_model_path : str
        Absolute path to the clean master ``.EDB`` file.  A temporary
        copy is created for each batch and deleted afterwards.
    n_modes : int
        Number of modes for the modal analysis.
    cooldown : int
        Seconds to wait between ETABS shutdown and restart (default 10).

    Returns
    -------
    list[str]
        Paths of newly generated batch CSVs.
    """
    import glob
    import time

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Count ALL scenarios already on disk (any naming convention)
    # ------------------------------------------------------------------
    existing_files = sorted(glob.glob(os.path.join(output_dir, "batch_*.csv")))
    scenarios_completed = 0
    for f in existing_files:
        df = pd.read_csv(f)
        scenarios_completed += len(df)

    print(f"✅ Total scenarios already completed: {scenarios_completed}")

    # ------------------------------------------------------------------
    # 2. Slice the master array to get only what's left
    # ------------------------------------------------------------------
    remaining_array = dataset_array[scenarios_completed:]
    n_batches_left = int(np.ceil(len(remaining_array) / batch_size))

    if len(remaining_array) == 0:
        print("🎉 All scenarios are already complete!")
        return []

    print(f"📊 Remaining scenarios: {len(remaining_array)} → "
          f"{n_batches_left} batches of {batch_size}")

    # ------------------------------------------------------------------
    # 3. The reboot loop (with signal-based kill switch)
    #
    #    Why signal-based?  comtypes' _compointer_base.__del__ silently
    #    swallows KeyboardInterrupt, so a normal try/except never fires.
    #    Instead we install a SIGINT handler that sets _kill_requested,
    #    and check the flag between iterations.
    # ------------------------------------------------------------------
    import subprocess
    import sys

    global _kill_requested
    _kill_requested = False

    # Install our signal handler (save the original so we restore it later)
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)

    new_csv_paths = []
    temp_model_path = None

    # Find the highest existing part2 batch number so new files continue from there
    import re
    existing_part2 = glob.glob(os.path.join(output_dir, "batch_part2_*.csv"))
    max_part2_num = 0
    for fp in existing_part2:
        m = re.search(r'batch_part2_(\d+)\.csv$', os.path.basename(fp))
        if m:
            max_part2_num = max(max_part2_num, int(m.group(1)))

    try:  # outer try — ensures we always restore the original SIGINT handler

        for i in range(n_batches_left):
            batch_num = max_part2_num + i + 1
            batch_path = os.path.join(output_dir, f"batch_part2_{batch_num:04d}.csv")

            # Check kill flag at the TOP of each batch iteration
            if _kill_requested:
                break

            # Skip if this part2 batch already exists (crash recovery)
            if os.path.exists(batch_path):
                print(f"⏩ Skipping {os.path.basename(batch_path)} — already exists.")
                continue

            # --- Launch fresh ETABS with a clean model copy ---
            print(f"\n{'='*60}")
            print(f"⚙️  Batch {i + 1}/{n_batches_left}  |  "
                  f"Launching fresh ETABS...")
            print(f"{'='*60}")

            # 1. Launch
            SapModel, temp_model_path = launch_etabs(
                master_model_path, verbose=True
            )

            # 2. Ensure baseline exists
            if not os.path.exists(not_damaged_file):
                print("   Baseline not found — generating...")
                compute_and_save_not_damaged(
                    SapModel, group_name, E_i, mat_names,
                    len(mat_names), not_damaged_file, n_modes
                )

            # 3. Slice & compute
            batch = remaining_array[i * batch_size:(i + 1) * batch_size]
            print(f"   Running {len(batch)} scenarios...")

            damage_indicators = create_dataset(
                batch, SapModel, group_name, E_i, mat_names,
                not_damaged_file, n_modes
            )

            # Check kill flag AFTER compute — if set, discard partial results
            if _kill_requested:
                print("\n   Kill flag detected after compute — "
                      "discarding this batch's results.")
                SapModel = None
                break

            # 4. Save (only reached if fully completed AND no kill flag)
            batch_df = pd.DataFrame(damage_indicators)
            batch_df.to_csv(batch_path, index=False)
            new_csv_paths.append(batch_path)
            print(f"💾 Saved → {batch_path}")

            # 5. Graceful ETABS shutdown
            stop_api(verbose=True)
            SapModel = None

            # --- Clean up temp model + auxiliary files (normal path) ---
            _cleanup_temp_files(temp_model_path)

            # --- Cooldown before next reboot ---
            if i < n_batches_left - 1:
                print(f"   Cooling down {cooldown}s before next batch...")
                time.sleep(cooldown)

    finally:
        # Restore the original SIGINT handler
        signal.signal(signal.SIGINT, original_sigint)

    # ------------------------------------------------------------------
    # 🛑  Emergency cleanup if the kill switch was triggered
    # ------------------------------------------------------------------
    if _kill_requested:
        print("\n")
        print("=" * 60)
        print("🛑  KILL SWITCH ACTIVATED — Emergency shutdown")
        print("=" * 60)

        # Force-kill ETABS via taskkill (COM may be unresponsive)
        print("   Force-killing ETABS.exe...")
        subprocess.run(
            ['taskkill', '/F', '/IM', 'ETABS.exe', '/T'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # Clean up temp model + all auxiliary files
        time.sleep(2)
        _cleanup_temp_files(temp_model_path)

        print(f"\n   Batches completed before shutdown: "
              f"{len(new_csv_paths)}")
        print("🛑 Shutdown complete. Safe to close the terminal.")
        sys.exit(0)

    print(f"\n🏁 Finished — {len(new_csv_paths)} new batches generated.")
    return new_csv_paths


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

    # Auto-detect the modal case name BEFORE analysis
    NumberNames, case_names, ret_code = SapModel.LoadCases.GetNameList()
    modal_case_name = "Modal"  # Default fallback
    if ret_code == 0:
        for name in case_names:
            case_type, sub_type, design_type, design_opt, auto, ret = \
                SapModel.LoadCases.GetTypeOAPI_1(name)
            if case_type == 3:  # 3 = Modal Cases
                modal_case_name = name
                break

    # Force ETABS to compute exactly n_modes
    SapModel.LoadCases.ModalEigen.SetNumberModes(modal_case_name, n_modes, 1)

    # Edit materials according to damage severity
    for i, s in enumerate(severity):
        E = E_i * (1 - s)
        set_material(mat_names[i], E, SapModel)
    
    # Run analysis
    SapModel.Analyze.RunAnalysis()

    # Select only the Modal case for output
    SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
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

    [_, _, _, _, _, _, Frequency_raw, _, _] = SapModel.Results.ModalPeriod(
        0, '', '', [], [], [], [], []
    )

    if not StepNum:
        raise RuntimeError(f"No mode shapes returned for group '{group_name}'.")

    # Determine the actual number of modes returned by ETABS
    actual_n_modes = len(set(StepNum))
    if actual_n_modes != n_modes:
        print(f"  [Warning] Requested {n_modes} modes, but ETABS computed {actual_n_modes}.")

    Frequency = np.array(Frequency_raw[:actual_n_modes])

    # ETABS groups by Object, then by Mode. 
    # Reshaping to (-1, actual_n_modes) correctly assigns columns to modes.
    U1 = np.array(U1).reshape((-1, actual_n_modes))
    U2 = np.array(U2).reshape((-1, actual_n_modes))
    U3 = np.array(U3).reshape((-1, actual_n_modes))
    
    # Because U1 is now (n_joints, actual_n_modes), we no longer need .T
    mode_shapes = np.concatenate([U1, U2, U3], axis=0)

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
    omega = 2 * np.pi * np.array(Frequency)
    # Modal flexibility:  F = Φ diag(1/ωᵢ²) Φᵀ
    inv_omega2 = np.diag(1.0 / omega**2)

    Flexibility = mode_shapes @ inv_omega2 @ mode_shapes.T
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
                                 n_elements, out_path, n_modes=12):
    """
    Run the undamaged scenario and save the baseline flexibility matrix.
    """
    print("Running the not damaged scenario...")
    Frequency_0, mode_shapes_0 = create_dp(
        SapModel, n_elements * [0], group_name, E_i, mat_names, n_modes
    )
    notDamaged_Flexibility = flexibility_matrix(Frequency_0, mode_shapes_0)

    with open(out_path, "wb") as f:
        pickle.dump(notDamaged_Flexibility, f)
    print(f"✅ notDamaged_Flexibility saved to {out_path}")


def create_dataset(batch_array, SapModel, group_name, E_i, mat_names,
                   not_damaged_path, n_modes=12):
    """
    Generate damage indicator dataset from a batch of severity arrays.
    """
    # Load saved notDamaged_Flexibility
    with open(not_damaged_path, "rb") as f:
        notDamaged_Flexibility = pickle.load(f)

    DI = []
    for severity in tqdm(batch_array, desc='Generating batch'):
        # Check the kill switch between each scenario
        if _kill_requested:
            print("\n   ⚠️  Kill flag detected — stopping mid-batch.")
            break

        Frequency, mode_shapes = create_dp(
            SapModel, severity, group_name, E_i, mat_names, n_modes
        )
        Damaged_Flexibility = flexibility_matrix(Frequency, mode_shapes)
        Damage_indicator = delta_fmax(Damaged_Flexibility, notDamaged_Flexibility)
        DI.append(Damage_indicator)

    return DI


def run_batches(dataset_array, batch_size, SapModel, group_name, E_i,
                mat_names, not_damaged_file, output_dir, n_modes=12):
    """
    Orchestrate batch processing with CRASH RECOVERY: 
    Splits dataset_array into batches, generates damage indicators, 
    and skips any batch that already exists in the output_dir.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # --- FOOLPROOF BASELINE GENERATION ---
    if not os.path.exists(not_damaged_file):
        print(f"Baseline {not_damaged_file} not found. Generating implicitly with n={n_modes} modes...")
        compute_and_save_not_damaged(
            SapModel, group_name, E_i, mat_names, len(mat_names), not_damaged_file, n_modes
        )
        
    all_csv_paths = []
    n_batches = int(np.ceil(len(dataset_array) / batch_size))

    for i in range(n_batches):
        batch_path = os.path.join(output_dir, f"batch_{i + 1:03d}.csv")
        all_csv_paths.append(batch_path)
        
        # --- CRASH RECOVERY LOGIC ---
        # If the file already exists, ETABS already finished it. Skip it!
        if os.path.exists(batch_path):
            print(f" Skipping batch {i + 1}/{n_batches} | File already exists: {batch_path}")
            continue

        # If it doesn't exist, slice the array and run ETABS
        batch = dataset_array[i * batch_size:(i + 1) * batch_size]
        print(f"\n Processing batch {i + 1}/{n_batches} with {len(batch)} samples...")

        # Generate dataset
        damage_indicators = create_dataset(
            batch, SapModel, group_name, E_i, mat_names, not_damaged_file, n_modes
        )

        # Save batch to CSV
        batch_df = pd.DataFrame(damage_indicators)
        batch_df.to_csv(batch_path, index=False)
        print(f" Saved {batch_path}")

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

    # 1. Unlock and detect Modal Case
    SapModel.SetModelIsLocked(False)
    
    NumberNames, case_names, ret_code = SapModel.LoadCases.GetNameList()
    modal_case_name = "Modal"  # Default fallback
    if ret_code == 0:
        for name in case_names:
            case_type, sub_type, design_type, design_opt, auto, ret = \
                SapModel.LoadCases.GetTypeOAPI_1(name)
            if case_type == 3:  # 3 = Modal Cases
                modal_case_name = name
                break
                
    # Force ETABS to compute exactly n_modes
    SapModel.LoadCases.ModalEigen.SetNumberModes(modal_case_name, n_modes, 1)

    # 2. Set undamaged material properties and run analysis
    for i in range(n_elements):
        set_material(mat_names[i], E_i, SapModel)   # 0% damage → full E
    SapModel.Analyze.RunAnalysis()

    # 3. Configure output: select only the Modal case
    SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
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

    if NumberResults == 0 or ret != 0 or not StepNum:
        raise RuntimeError(
            f"ModeShape extraction failed for group '{group_name}' "
            f"(NumberResults={NumberResults}, ret={ret})."
        )

    # Determine the actual number of modes returned by ETABS
    actual_n_modes = len(set(StepNum))
    if actual_n_modes != n_modes:
        print(f"  [Warning] Requested {n_modes} modes, but ETABS computed {actual_n_modes}.")

    # ETABS groups by Object, then by Mode. 
    # Reshaping to (-1, actual_n_modes) correctly assigns columns to modes.
    U1 = np.array(U1).reshape((-1, actual_n_modes))
    U2 = np.array(U2).reshape((-1, actual_n_modes))
    U3 = np.array(U3).reshape((-1, actual_n_modes))
    
    n_joints = U1.shape[0]

    # 6. Build the mode-shape matrix expected by SNPO:
    #    Since U1 is (n_joints, actual_n_modes), SNPO expects (actual_n_modes, 3*n_joints)
    #    We must transpose to (actual_n_modes, n_joints) BEFORE concatenating along axis=1
    mode_shapes_for_snpo = np.concatenate([U1.T, U2.T, U3.T], axis=1)

    # 7. Save
    np.savetxt(out_csv_path, mode_shapes_for_snpo, delimiter=",")

    print(f"✅ Mode shapes successfully saved to {out_csv_path}")
    print(f"   Shape: {mode_shapes_for_snpo.shape}")
    print(f"   (n_modes={n_modes}, 3*n_sensors={3*n_joints}, "
          f"n_sensors={n_joints})")