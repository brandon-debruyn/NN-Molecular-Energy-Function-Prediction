import matplotlib.pyplot as plt
import numpy as np

def plot_molecule_frame(df, frame, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

    frame_data = df[df["frame"] == frame]

    ATOM_STYLE = {
        1:  dict(color="lightgray", s=30),
        6:  dict(color="black", s=80),
        8:  dict(color="red", s=120),
    }

    for Z, group in frame_data.groupby("atomic_number"):
        style = ATOM_STYLE.get(Z, dict(color="blue", s=50))
        ax.scatter(
            group["x"], group["y"], group["z"],
            **style,
            label=f"Z={Z}"
        )

    ax.set_title(f"Malonaldehyde — Frame {frame}")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")

    xyz = frame_data[["x", "y", "z"]].to_numpy()
    max_range = (xyz.max(axis=0) - xyz.min(axis=0)).max() / 2
    mid = xyz.mean(axis=0)

    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.legend()
    return ax

def proton_transfer_coordinate_and_energy(df):
    frames = sorted(df["frame"].unique())
    qs = []
    Es = []

    for f in frames:
        g = df[df["frame"] == f].sort_values("atom_index")
        E = g["energy"].iloc[0]

        O = g[g["atomic_number"] == 8][["x","y","z"]].to_numpy()
        H = g[g["atomic_number"] == 1][["x","y","z"]].to_numpy()

        if O.shape[0] != 2 or H.shape[0] < 1:
            continue

        # distances from each H to each O: shape (nH, 2)
        dHO = np.linalg.norm(H[:, None, :] - O[None, :, :], axis=2)

        # bridging H = H with smallest distance to either O
        h_idx = np.argmin(dHO.min(axis=1))
        d1, d2 = dHO[h_idx, 0], dHO[h_idx, 1]

        q = d1 - d2
        qs.append(q)
        Es.append(E)

    return np.array(qs), np.array(Es)