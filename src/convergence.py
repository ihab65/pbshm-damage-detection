"""
convergence.py — Modal Convergence vs. Computational Cost Analysis

Standalone functions for:
  1. Sweeping mode counts in ETABS and collecting flexibility matrices + timings.
  2. Computing relative convergence error (Frobenius norm).
  3. Detecting the optimal "knee / elbow" point on the trade-off curve.
  4. Generating a publication-quality dual-axis Pareto figure.

These functions are intentionally self-contained and do NOT depend on
any other module in the codebase (etabs_api, optimization, etc.).
"""

import time
import numpy as np
from numpy.linalg import norm


# ---------------------------------------------------------------------------
# 1. Data Collection  (ETABS sweep)
# ---------------------------------------------------------------------------

def run_convergence_sweep(SapModel, group_name, E_i, mat_names,
                          modes_range=range(1, 13),
                          baseline_n=None,
                          modal_case="Modal"):
    """
    Sweep mode counts and collect flexibility matrices + wall-clock times.

    For every *n_modes* in ``modes_range`` this function:
      - Sets the ETABS modal case to use exactly *n_modes* Eigen modes.
      - Applies zero-damage (undamaged) material properties.
      - Runs the analysis.
      - Extracts mode shapes and frequencies, then computes the
        flexibility matrix.
      - Records the elapsed wall-clock time.

    Parameters
    ----------
    SapModel : object
        Live ETABS SapModel COM object.
    group_name : str
        Joint group used to extract mode shapes.
    E_i : float
        Undamaged Young's modulus.
    mat_names : list[str]
        Material names for each zone (length = n_elements).
    modes_range : range or list[int]
        Mode counts to test (default ``range(1, 13)``).
    baseline_n : int or None
        Optional massive mode count (e.g. 30) to establish a true 
        converged flexibility matrix. Computed alongside the sweep but 
        its time is not included in the returned compute_times.
    modal_case : str
        Name of the ETABS modal load case (default ``"Modal"``).

    Returns
    -------
    flexibility_matrices : dict[int, ndarray]
        ``{n_modes: F_n}`` flexibility matrices. Includes baseline_n if specified.
    compute_times : list[float]
        Wall-clock seconds for each mode count in ``modes_range``.
    """
    modes_to_run = list(modes_range)
    if baseline_n is not None and baseline_n not in modes_to_run:
        modes_to_run.append(baseline_n)

    try:
        from tqdm import tqdm
        iterator = tqdm(modes_to_run, desc="Convergence sweep")
    except ImportError:
        iterator = modes_to_run

    n_elements = len(mat_names)
    flexibility_matrices = {}
    compute_times_dict = {}

    for n_modes in iterator:
        t0 = time.time()

        # 1. Unlock and set mode count
        SapModel.SetModelIsLocked(False)
        ret = SapModel.LoadCases.ModalEigen.SetNumberModes(
            modal_case, n_modes, 1
        )
        if ret != 0:
            raise RuntimeError(
                f"ETABS refused to change the mode count to {n_modes}. "
                f"Check that '{modal_case}' is an Eigen modal case."
            )

        # 2. Zero-damage material properties
        for i in range(n_elements):
            SapModel.PropMaterial.SetMPIsotropic(
                mat_names[i], E_i, 0.3, 0.00001
            )

        # 3. Run analysis
        SapModel.Analyze.RunAnalysis()

        # 4. Configure output for modal case
        SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
        SapModel.Results.Setup.SetCaseSelectedForOutput(modal_case)
        SapModel.Results.Setup.SetOptionModeShape(1, n_modes, True)

        # 5. Extract mode shapes
        (NumberResults, Obj, Elm, LoadCase, StepType, StepNum,
         U1, U2, U3, R1, R2, R3, _ret) = SapModel.Results.ModeShape(
            group_name, 2, 0, "", "", "", "", [],
            [], [], [], [], [], []
        )

        # 6. Extract frequencies
        (_, _, _, _, _, _, Frequency, _, _) = SapModel.Results.ModalPeriod(
            0, '', '', [], [], [], [], []
        )

        # ── Diagnostic logging ──────────────────────────────────────
        n_freq = len(Frequency)
        n_results = NumberResults
        print(f"\n{'='*60}")
        print(f"  n_requested = {n_modes}")
        print(f"  ModeShape NumberResults = {n_results}")
        print(f"  ModalPeriod len(Frequency) = {n_freq}")
        print(f"  Frequencies (Hz): {[f'{f:.4f}' for f in Frequency]}")

        # Detect mismatch before reshape
        n_joints_from_results = n_results // n_modes if n_modes > 0 else 0
        print(f"  Inferred n_joints = {n_results} / {n_modes} "
              f"= {n_joints_from_results}")
        if n_results % n_modes != 0:
            print(f"  ⚠ WARNING: NumberResults ({n_results}) is NOT "
                  f"divisible by n_modes ({n_modes})!")
        if n_freq != n_modes:
            print(f"  ⚠ WARNING: len(Frequency)={n_freq} ≠ "
                  f"n_requested={n_modes}")
        # ────────────────────────────────────────────────────────────

        # ETABS groups by Object, then by Mode. 
        # Reshaping to (-1, n_modes) correctly assigns columns to modes.
        U1 = np.array(U1).reshape((-1, n_modes))
        U2 = np.array(U2).reshape((-1, n_modes))
        U3 = np.array(U3).reshape((-1, n_modes))
        
        # Because U1 is now (n_joints, n_modes), we no longer need .T
        mode_shapes = np.concatenate([U1, U2, U3], axis=0)

        # ── Diagnostic: Φ matrix shape ──────────────────────────────
        print(f"  Φ (mode_shapes) shape = {mode_shapes.shape}  "
              f"(expected: (3×{n_joints_from_results}, {n_modes}) "
              f"= ({3*n_joints_from_results}, {n_modes}))")
        print(f"{'='*60}")
        # ────────────────────────────────────────────────────────────

        # 8. Compute flexibility matrix:  F = Φ diag(1/ωᵢ²) Φᵀ
        #    NOTE: The correct modal flexibility uses 1/ω², NOT ω.
        #    Using ω causes higher modes to dominate (error increases
        #    with n), which is the opposite of physical convergence.
        omega = 2 * np.pi * np.array(Frequency[:n_modes])
        F_n = mode_shapes @ np.diag(1.0 / omega**2) @ mode_shapes.T
        flexibility_matrices[n_modes] = F_n

        compute_times_dict[n_modes] = time.time() - t0

    compute_times = [compute_times_dict[n] for n in modes_range]

    return flexibility_matrices, compute_times


# ---------------------------------------------------------------------------
# 2. Convergence-error computation
# ---------------------------------------------------------------------------

def compute_convergence_errors(flexibility_matrices, modes_range,
                               reference_n=None):
    """
    Relative Frobenius-norm error of each flexibility matrix w.r.t. a
    reference (highest mode-count) matrix.

    Parameters
    ----------
    flexibility_matrices : dict[int, ndarray]
        ``{n_modes: F_n}`` as returned by ``run_convergence_sweep``.
    modes_range : range or list[int]
        Mode counts to evaluate, in the same order used during the sweep.
    reference_n : int or None
        Which mode count to treat as "truth". Defaults to ``max(modes_range)``.

    Returns
    -------
    errors : list[float]
        Relative error (%) for each entry in ``modes_range``.
    """
    if reference_n is None:
        reference_n = max(modes_range)

    F_ref = flexibility_matrices[reference_n]
    norm_ref = norm(F_ref, 'fro')

    errors = []
    for n in modes_range:
        err = (norm(F_ref - flexibility_matrices[n], 'fro') / norm_ref) * 100
        errors.append(err)
    return errors


# ---------------------------------------------------------------------------
# 3. Knee / Elbow detection  (Kneedle-style algorithm)
# ---------------------------------------------------------------------------

def find_elbow(modes_range, convergence_errors, compute_times):
    """
    Detect the optimal elbow / knee point on the convergence vs. cost curve.

    **Method (Kneedle-style):**

    1. Normalize both metrics to [0, 1].
    2. Build a "benefit" score for each mode count:
           benefit(n) = Δ_error(n) / Δ_time(n)
       where Δ_error is the marginal *reduction* in error and Δ_time is the
       marginal *increase* in compute time when going from *n-1* → *n* modes.
    3. Additionally compute a **curvature** score on the normalized error
       curve:  κ(n) = |f''(n)|  (discrete second derivative).
    4. The elbow is the mode count that maximizes the distance from the
       straight line connecting the first and last points of the normalized
       error curve (classic "maximum distance" knee detection).

    Parameters
    ----------
    modes_range : list[int] or range
        Mode counts tested.
    convergence_errors : list[float]
        Relative error (%) for each mode count.
    compute_times : list[float]
        Compute time (seconds) for each mode count.

    Returns
    -------
    elbow_n : int
        Optimal mode count (best trade-off).
    diagnostics : dict
        Contains ``'distances'``, ``'norm_errors'``, ``'norm_times'``
        for further inspection / plotting.
    """
    modes = np.array(list(modes_range), dtype=float)
    errors = np.array(convergence_errors, dtype=float)
    times = np.array(compute_times, dtype=float)

    # --- Normalize to [0, 1] ---
    def _minmax(arr):
        lo, hi = arr.min(), arr.max()
        if hi - lo == 0:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    x_norm = _minmax(modes)
    y_norm = _minmax(errors)

    # --- Maximum perpendicular distance from the straight line ---
    # Line connecting first point (x_norm[0], y_norm[0]) to last point
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_len = norm(line_vec)

    distances = np.zeros(len(modes))
    for i in range(len(modes)):
        pt = np.array([x_norm[i], y_norm[i]])
        # Signed perpendicular distance
        distances[i] = np.abs(
            (p2[1] - p1[1]) * pt[0]
            - (p2[0] - p1[0]) * pt[1]
            + p2[0] * p1[1]
            - p2[1] * p1[0]
        ) / line_len

    elbow_idx = np.argmax(distances)
    elbow_n = int(modes[elbow_idx])

    diagnostics = {
        'distances': distances.tolist(),
        'norm_errors': y_norm.tolist(),
        'norm_times': _minmax(times).tolist(),
    }
    return elbow_n, diagnostics


# ---------------------------------------------------------------------------
# 4. Publication-quality plot
# ---------------------------------------------------------------------------

def plot_convergence_vs_cost(modes_range, convergence_errors, compute_times,
                             elbow_n=None,
                             figsize=(11, 6.5), dpi=300,
                             save_path=None):
    """
    Create a dual-axis Pareto trade-off figure:
      • Left y-axis  → Relative Flexibility Matrix Error (%)
      • Right y-axis → Computation Time (seconds)
      • Vertical annotation at the elbow / knee point

    Parameters
    ----------
    modes_range : range or list[int]
        Mode counts tested.
    convergence_errors : list[float]
        Relative error (%) for each mode count.
    compute_times : list[float]
        Wall-clock seconds for each mode count.
    elbow_n : int or None
        Optimal mode count to highlight. If None, auto-detected.
    figsize : tuple
        Figure dimensions (default ``(11, 6.5)``).
    dpi : int
        Resolution (default ``300`` for thesis printing).
    save_path : str or None
        If given, save the figure to this path (PNG / PDF / SVG …).

    Returns
    -------
    fig : matplotlib.figure.Figure
    (ax1, ax2) : tuple of Axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    modes = list(modes_range)

    # Auto-detect elbow if not provided
    if elbow_n is None:
        elbow_n, _ = find_elbow(modes_range, convergence_errors, compute_times)

    # ── Colour palette ──────────────────────────────────────────────────
    COLOR_ERROR  = '#0D47A1'   # deep blue
    COLOR_TIME   = '#C62828'   # deep red
    COLOR_ELBOW  = '#2E7D32'   # forest green
    COLOR_BG     = '#FAFAFA'
    COLOR_GRID   = '#E0E0E0'

    # ── Figure + axes ───────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi, facecolor=COLOR_BG)
    ax1.set_facecolor(COLOR_BG)

    # Left axis — convergence error
    ax1.set_xlabel('Number of Modes Included ($n$)',
                   fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Relative Flexibility Error  (%)',
                   color=COLOR_ERROR, fontsize=13, fontweight='bold',
                   labelpad=10)

    line1, = ax1.plot(
        modes, convergence_errors,
        color=COLOR_ERROR, marker='o', markersize=8,
        linewidth=2.5, zorder=5, label='Convergence Error',
    )
    # Subtle fill under the error curve
    ax1.fill_between(modes, convergence_errors,
                     alpha=0.08, color=COLOR_ERROR, zorder=2)
    ax1.tick_params(axis='y', labelcolor=COLOR_ERROR, labelsize=11)
    ax1.set_xticks(modes)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5, color=COLOR_GRID, zorder=0)

    # Right axis — compute time
    ax2 = ax1.twinx()
    ax2.set_facecolor('none')
    ax2.set_ylabel('Computation Time  (s)',
                   color=COLOR_TIME, fontsize=13, fontweight='bold',
                   labelpad=10)

    line2, = ax2.plot(
        modes, compute_times,
        color=COLOR_TIME, marker='s', markersize=8,
        linewidth=2.5, linestyle='--', zorder=5, label='Compute Time',
    )
    ax2.tick_params(axis='y', labelcolor=COLOR_TIME, labelsize=11)

    # ── Elbow annotation ────────────────────────────────────────────────
    elbow_idx = modes.index(elbow_n)
    elbow_err = convergence_errors[elbow_idx]
    elbow_time = compute_times[elbow_idx]

    # Vertical span highlighting the optimal zone
    ax1.axvspan(elbow_n - 0.35, elbow_n + 0.35,
                color=COLOR_ELBOW, alpha=0.12, zorder=1,
                label=f'Optimal point ($n={elbow_n}$)')

    # Star markers on both curves
    ax1.plot(elbow_n, elbow_err,
             marker='*', markersize=20, color=COLOR_ELBOW, zorder=10,
             markeredgecolor='black', markeredgewidth=0.8)
    ax2.plot(elbow_n, elbow_time,
             marker='*', markersize=20, color=COLOR_ELBOW, zorder=10,
             markeredgecolor='black', markeredgewidth=0.8)

    # Text box with metrics at the elbow
    textstr = (
        f"Optimal: $n = {elbow_n}$\n"
        f"Error: {elbow_err:.2f} %\n"
        f"Time:  {elbow_time:.1f} s"
    )
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor=COLOR_ELBOW,
                      alpha=0.15, edgecolor=COLOR_ELBOW, linewidth=1.5)

    # Position the annotation above the elbow point
    ax1.annotate(
        textstr,
        xy=(elbow_n, elbow_err),
        xytext=(elbow_n + 1.5, max(convergence_errors) * 0.60),
        fontsize=11, fontweight='bold', color='#1B5E20',
        bbox=bbox_props,
        arrowprops=dict(arrowstyle='->', color=COLOR_ELBOW, lw=2),
        zorder=15,
    )

    # ── Title + legend ──────────────────────────────────────────────────
    ax1.set_title(
        'Modal Flexibility Convergence  vs.  Computational Cost',
        fontsize=15, fontweight='bold', pad=18,
    )

    handles = [line1, line2, ax1.patches[0]]   # error, time, elbow span
    ax1.legend(
        handles=handles,
        loc='upper right', fontsize=11, framealpha=0.9,
        edgecolor='gray',
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Figure saved → {save_path}")

    return fig, (ax1, ax2)
