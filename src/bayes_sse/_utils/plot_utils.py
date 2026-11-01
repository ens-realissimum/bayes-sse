import matplotlib.pyplot as plt
import numpy as np

fig_size = (20, 12)
hfont = {"fontsize": 12, "fontname": "Calibri"}
axis_names = ["x", "y", "z"]


def plot_2_3d_vectors_on_same_subplot_by_pairs(
    y1, y2, x, y_name, y_meas_unit, x_name, x_meas_unit, y1_label, y2_label
):
    fig, ax = plt.subplots(3, 1, figsize=fig_size)

    for k in range(3):
        ax[k].plot(x, y1[k, :], label=y1_label)
        ax[k].plot(x, y2[k, :], label=y2_label)

        ax[k].set_xlabel(x_name + " " + x_meas_unit, **hfont)
        ax[k].set_ylabel(
            "$" + y_name + "_" + axis_names[k] + ", " + y_meas_unit + "$", **hfont
        )

        ax[k].grid(True, which="both", axis="both")
        ax[k].legend(loc="upper right")
        ax[k].set_xlim([np.min(x), np.max(x)])

    plt.tight_layout()
    plt.subplots_adjust()
    fig.show()
