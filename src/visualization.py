"""
visualization.py — The Plotting Engine

Everything related to Matplotlib and the JSON-based structural drawing.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_scatter_by_z_levels(coord, selected_ids=None,
                             points_per_row=4, figsize=(16, 10)):
    """
    Plot scatter subplots of X vs Y for each unique Z level.
    Hides axis ticks and minimizes subplot spacing.

    Parameters
    ----------
    coord : ndarray
        Array with columns [id, x, y, z] (as strings).
    selected_ids : array-like or None
        IDs of sensors to highlight in red.
    points_per_row : int
        Number of subplots per row.
    figsize : tuple
        Figure size.
    """
    coord = coord.astype(str)
    z_levels = np.unique(coord[:, 3].astype(float))
    n_levels = len(z_levels)
    rows = (n_levels + points_per_row - 1) // points_per_row

    fig, axes = plt.subplots(rows, points_per_row, figsize=figsize)
    axes = axes.flatten()

    for i, z in enumerate(z_levels):
        ax = axes[i]
        layer = coord[coord[:, 3].astype(float) == z]

        # Plot all points
        ax.scatter(layer[:, 1].astype(float),
                   layer[:, 2].astype(float),
                   s=5, color='gray')

        # Highlight selected sensors
        if selected_ids is not None:
            mask = np.isin(layer[:, 0], selected_ids)
            highlighted = layer[mask]
            ax.scatter(highlighted[:, 1].astype(float),
                       highlighted[:, 2].astype(float),
                       s=20, color='red', edgecolors='black')

        # Title and clean axes
        ax.set_title(f"Level {i + 1}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Remove spacing
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.show()


def plot_frames_and_rectangles_on_axis(ax, frames_coords, rectangles,
                                       zone_values=None):
    """
    Draw structural frames and colored zone rectangles on a given axis.

    Updated to handle a dynamic number of zones (not just Zone 1 and Zone 2).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw on.
    frames_coords : list[dict]
        List of frame coordinate dicts with keys 'label', 'start', 'end'.
    rectangles : dict
        Dictionary of rectangle definitions with 'x' and 'z' lists.
    zone_values : list[float] or None
        Severity values for each zone, in the order they appear.
        The first value corresponds to the rectangle keyed 'Zone',
        and subsequent values correspond to frames labeled 'Zone'.
    """
    green_red_cmap = mcolors.LinearSegmentedColormap.from_list(
        'GreenRed', ['green', 'red']
    )
    norm = mcolors.Normalize(vmin=0, vmax=1)

    zone_frame_idx = 0  # index for accessing zone frame values

    # Determine zone frame values (all values after the first one)
    zone_frame_values = zone_values[1:] if zone_values and len(zone_values) > 1 else None

    # Plot regular frames
    for frame in frames_coords:
        if frame['label'] == 'Frame':
            x = [frame['start'][0], frame['end'][0]]
            z = [frame['start'][1], frame['end'][1]]
            ax.plot(x, z, color='blue', alpha=0.7, linewidth=3)

    # Plot Zone frames with their values
    for frame in frames_coords:
        if frame['label'] == 'Zone':
            x = [frame['start'][0], frame['end'][0]]
            z = [frame['start'][1], frame['end'][1]]
            ax.plot(x, z, color='red', alpha=0.7, linewidth=3)

            # Midpoint for text
            mid_x = (x[0] + x[1]) / 2
            mid_z = (z[0] + z[1]) / 2

            if zone_frame_values and zone_frame_idx < len(zone_frame_values):
                value = zone_frame_values[zone_frame_idx]
                ax.text(mid_x + 5, mid_z, f"{value * 100:.1f}%",
                        color='black', fontsize=12, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.7,
                                  boxstyle='round,pad=0.3'))
                zone_frame_idx += 1

    # Plot rectangles colored by value
    zone_rect_value = zone_values[0] if zone_values else None
    for key, rect in rectangles.items():
        if key == 'Zone' and zone_rect_value is not None:
            color = green_red_cmap(norm(zone_rect_value))
            ax.fill(rect['x'], rect['z'], color=color, alpha=0.6)

            cx = np.mean(rect['x'])
            cz = np.mean(rect['z'])
            ax.text(cx, cz, f"{zone_rect_value * 100:.1f}%",
                    color='black', fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7,
                              boxstyle='round,pad=0.3'))
        else:
            fill_color = 'grey' if key == 'Zone 1' else 'lightgrey'
            ax.fill(rect['x'], rect['z'], color=fill_color, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.grid(True)
    ax.axis('equal')


def plot_side_by_side(frames_coords, rectangles, pred_values, target_values):
    """
    Plot predicted vs target damage severity side by side.

    Updated to handle a dynamic number of zones.

    Parameters
    ----------
    frames_coords : list[dict]
        Frame coordinate definitions.
    rectangles : dict
        Rectangle definitions.
    pred_values : list[float]
        Predicted severity values for each zone.
    target_values : list[float]
        Target severity values for each zone.
    """
    fig, axs = plt.subplots(1, 2, figsize=(18, 9))

    # Predicted
    plot_frames_and_rectangles_on_axis(
        axs[0], frames_coords, rectangles, zone_values=pred_values
    )
    axs[0].set_title('Predicted Values')

    # Target
    plot_frames_and_rectangles_on_axis(
        axs[1], frames_coords, rectangles, zone_values=target_values
    )
    axs[1].set_title('Target Values')

    plt.show()
