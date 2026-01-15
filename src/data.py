# imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

import tensorflow as tf

import matplotlib.pyplot as plt

# constants
UNDESIRED_COLS = ["element", "atom_index"]

# methods
def read_xyz_trajectory(
    trajectory_file_path: str,
    trajectory_energy_file_path: str | None = None,
) -> pd.DataFrame:
    """
    Reads an XYZ *trajectory* (multiple concatenated XYZ frames) where each atom line is:
        Element  x  y  z  atomic_number

    If trajectory_energy_file_path is provided, it reads one energy per frame and
    attaches it to every atom-row in that frame via an 'energy' column.
    """
    rows: list[dict] = []

    with open(trajectory_file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    frame = 0

    # Parse frames
    while i < len(lines):
        try:
            n_atoms = int(lines[i])
        except ValueError as e:
            raise ValueError(
                f"Expected atom-count integer at line {i+1} of {trajectory_file_path}, got: {lines[i]!r}"
            ) from e
        i += 1

        # Read exactly n_atoms atom lines
        for atom_index in range(n_atoms):
            if i >= len(lines):
                raise ValueError(
                    f"Unexpected end of file while reading frame {frame} (expected {n_atoms} atoms)."
                )

            parts = lines[i].split()
            if len(parts) < 5:
                raise ValueError(
                    f"Atom line malformed at line {i+1}: {lines[i]!r} "
                    f"(expected: Element x y z Z)"
                )

            element, x, y, z, Z = parts[:5]
            rows.append(
                {
                    "frame": frame,
                    "atom_index": atom_index,
                    "element": element,
                    "atomic_number": int(Z),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                }
            )
            i += 1

        frame += 1

    df = pd.DataFrame(rows)

    if trajectory_energy_file_path is not None:
        energies: list[float] = []
        with open(trajectory_energy_file_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    energies.append(float(s))
                except ValueError:
                    continue

        n_frames = df["frame"].nunique()
        if len(energies) != n_frames:
            raise ValueError(
                f"Energy/frame mismatch: parsed {n_frames} frames from XYZ but found {len(energies)} energies "
                f"in {trajectory_energy_file_path} (need exactly one energy per frame)."
            )

        energy_map = dict(enumerate(energies))
        df["energy"] = df["frame"].map(energy_map)

    return df

def split_dataset(
    dataset: pd.DataFrame,
    val_size: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training and validation sets based on unique frames.

    Args:
        dataset (pd.DataFrame): The complete dataset containing multiple frames.
        val_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training and validation datasets.
    """
    unique_frames = dataset["frame"].unique()
    train_frames, val_frames = train_test_split(
        unique_frames, test_size=val_size, random_state=random_state
    )

    train_dataset = dataset[dataset["frame"].isin(train_frames)].reset_index(drop=True)
    val_dataset = dataset[dataset["frame"].isin(val_frames)].reset_index(drop=True)

    return train_dataset, val_dataset

def one_hot_encode_atomic_numbers(df, categories=None):
    """
    One-hot encode atomic numbers with consistent categories across splits.
    """
    if categories is None:
        categories = sorted(df["atomic_number"].unique())

    df = df.copy()
    df["atomic_number"] = pd.Categorical(df["atomic_number"], categories=categories)
    dummies = pd.get_dummies(df["atomic_number"], prefix="AN")
    df = df.drop(columns=["atomic_number"]).join(dummies)
    return df, categories

def standardize_coords(df, scaler=None):
    coords = df[["x", "y", "z"]]
    if scaler is None:
        scaler = StandardScaler()
        standardized = scaler.fit_transform(coords)
    else:
        standardized = scaler.transform(coords)
    df[["x", "y", "z"]] = standardized
    return df, scaler

def center_frame(df):
    coords = df[["x", "y", "z"]]
    centered = coords - coords.mean()
    df[["x", "y", "z"]] = centered
    return df

def normalize_energy(df, scaler=None):
    energy_df = df[["frame", "energy"]].drop_duplicates().set_index("frame")
    if scaler is None:
        scaler = StandardScaler()
        energy_df["energy_norm"] = scaler.fit_transform(energy_df[["energy"]])
    else:
        energy_df["energy_norm"] = scaler.transform(energy_df[["energy"]])

    return energy_df, scaler

def normalize_energy_relative(dataset: pd.DataFrame, energy_offset=None, scaler=None):
    """
    Compute relative energy per frame and (optionally) standardize it.

    Returns:
      energy_df indexed by frame with columns:
        - energy_rel
        - energy_norm
      energy_offset (float)
      scaler (StandardScaler)
    """
    energy_df = (
        dataset[["frame", "energy"]]
        .drop_duplicates()
        .set_index("frame")
        .sort_index()
    )

    if energy_offset is None:
        # TRAIN ONLY: choose reference from training set
        energy_offset = float(energy_df["energy"].mean())

    energy_df["energy_rel"] = energy_df["energy"] - energy_offset

    if scaler is None:
        scaler = StandardScaler()
        energy_df["energy_norm"] = scaler.fit_transform(energy_df[["energy_rel"]])
    else:
        energy_df["energy_norm"] = scaler.transform(energy_df[["energy_rel"]])

    return energy_df, energy_offset, scaler

def prepare_data(dataset, scalars=None, interatomic_distance=False, energy_mode="relative"):
    if scalars is None:
        scalars = {
            "coord": None,
            "energy": None,
            "dist": None,
            "atomic_categories": None,
            "dist_cols": None,
            "energy_offset": None,
        }
    else:
        if scalars.get("energy_offset") is None and energy_mode == "relative":
            raise ValueError("scalars['energy_offset'] is None. Did you forget to pass train-fitted scalars?")

    coord_scalar = scalars["coord"]
    dist_scalar = scalars["dist"]
    energy_scalar = scalars["energy"]
    dist_cols = scalars["dist_cols"]
    energy_offset = scalars["energy_offset"]

    # --- energy ---
    if energy_mode == "relative":
        if energy_scalar is None:
            energy_df, energy_offset, energy_scalar = normalize_energy_relative(dataset)
        else:
            energy_df, _, _ = normalize_energy_relative(dataset, energy_offset=energy_offset, scaler=energy_scalar)
    else:
        if energy_scalar is None:
            energy_df, energy_scalar = normalize_energy(dataset)
        else:
            energy_df, _ = normalize_energy(dataset, scaler=energy_scalar)

    dataset = dataset.merge(energy_df[["energy_norm"]], left_on="frame", right_index=True)

    if not interatomic_distance:
        # keep atom order consistent per frame
        if "atom_index" in dataset.columns:
            dataset = dataset.sort_values(["frame", "atom_index"]).reset_index(drop=True)

        # center and standardize coordinates
        dataset = dataset.groupby("frame", group_keys=False).apply(center_frame)
        
        # standardize coordinates
        if coord_scalar is None:
            dataset, coord_scalar = standardize_coords(dataset)
        else:
            dataset, _ = standardize_coords(dataset, scaler=coord_scalar)

        # one-hot encode atomic numbers
        dataset, atomic_categories = one_hot_encode_atomic_numbers(
            dataset,
            categories=scalars["atomic_categories"],
        )
        scalars["atomic_categories"] = atomic_categories

        drop_cols = ["energy_norm", "element", "atom_index"]
        X_dataset = dataset.drop(columns=[c for c in drop_cols if c in dataset.columns])
        y_dataset = dataset["energy_norm"]
        
        return X_dataset, y_dataset, scalars

    # 1) lock dist_cols once (train)
    if dist_cols is None:
        dist_cols = sorted([c for c in dataset.columns if c.startswith("dist_")])
        pt_cols = [c for c in ["pt_q", "pt_d1", "pt_d2"] if c in dataset.columns]
        dist_cols = dist_cols + pt_cols

        missing = [c for c in dist_cols if c not in dataset.columns]
        if missing:
            raise ValueError(f"Missing distance/proton columns: {missing}")

    # 2) frame-level X/y
    frame_X = dataset.groupby("frame")[dist_cols].first().sort_index()
    frame_y = dataset.groupby("frame")["energy_norm"].first().sort_index()

    # 3) scale X (fit only on train)
    if dist_scalar is None:
        dist_scalar = StandardScaler()
        X_scaled = dist_scalar.fit_transform(frame_X.values)
    else:
        X_scaled = dist_scalar.transform(frame_X.values)

    X_dataset = pd.DataFrame(X_scaled, index=frame_X.index, columns=dist_cols)
    y_dataset = frame_y

    # 4) store scalars
    scalars["coord"] = coord_scalar
    scalars["energy"] = energy_scalar
    scalars["dist"] = dist_scalar
    scalars["dist_cols"] = dist_cols
    scalars["energy_offset"] = energy_offset

    return X_dataset, y_dataset, scalars

def prepare_model_dataset(X, y, feature_cols=None):
    """
    Convert dataframe with frame-based molecule structure to tensor format for Keras.
    """
    
    if "frame" not in X.columns:
        raise ValueError("X must include a 'frame' column for grouping.")

    grouped = X.groupby("frame")
    frames = sorted(X["frame"].unique())
    max_atoms = grouped.size().max()

    if feature_cols is None:
        feature_cols = [
            c for c in X.columns
            if c not in {"frame", "atom_index", "element"}
        ]

    feature_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(X[c])
    ]

    if not feature_cols:
        raise ValueError("No numeric feature columns found for model input.")

    X_array = np.zeros((len(frames), max_atoms, len(feature_cols)), dtype=np.float32)
    for i, frame in enumerate(frames):
        frame_df = X[X["frame"] == frame]
        if "atom_index" in frame_df.columns:
            frame_df = frame_df.sort_values("atom_index")

        frame_data = frame_df[feature_cols].to_numpy(dtype=float)
        n_atoms = len(frame_data)
        X_array[i, :n_atoms, :] = frame_data
    
    X_tensor = tf.convert_to_tensor(X_array, dtype=tf.float32)
    if len(y) == len(X):
        y_frame = y.groupby(X["frame"]).first()
    else:
        y_frame = y
    y_tensor = tf.convert_to_tensor(y_frame.values, dtype=tf.float32)
    
    return X_tensor, y_tensor

def calculate_average_distances(dataset):
    frames = dataset['frame'].unique()
    n_atoms = 9

    average_atom_to_atom_distances = np.zeros((n_atoms, n_atoms))

    for frame in frames:
        frame_data = dataset[dataset['frame'] == frame]

        coords = (
            frame_data
            .sort_values('atom_index')[['x', 'y', 'z']]
            .values
        )  

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                average_atom_to_atom_distances[i, j] += dist
                average_atom_to_atom_distances[j, i] += dist

    average_atom_to_atom_distances /= len(frames)

    return average_atom_to_atom_distances

def standardize_X_train_apply(X_train, X_val, X_test, eps=1e-12):
    """
    Standardize per descriptor dimension across all atoms+frames in training set.
    """
    # flatten (F,N,D) -> (F*N, D)
    Xf = X_train.reshape(-1, X_train.shape[-1])
    mean = Xf.mean(axis=0)
    std = Xf.std(axis=0) + eps

    def apply(X):
        return (X - mean[None, None, :]) / std[None, None, :]

    return apply(X_train), apply(X_val), apply(X_test), mean, std

def build_bp_tensors(
    df,
    feature_cols,
    element_col="element",
    energy_col="energy",
    frame_col="frame",
    atom_index_col="atom_index",
    element_types=None
):
    frames = sorted(df[frame_col].unique())
    atoms_per_frame = df[df[frame_col] == frames[0]].shape[0]

    if element_types is None:
        element_types = sorted(df[element_col].unique().tolist())
    elem_to_id = {e: i for i, e in enumerate(element_types)}

    X_list, Z_list, y_list = [], [], []

    for f in frames:
        g = df[df[frame_col] == f].sort_values(atom_index_col)
        if g.shape[0] != atoms_per_frame:
            raise ValueError(f"Frame {f} has {g.shape[0]} atoms, expected {atoms_per_frame}")

        X = g.loc[:, feature_cols].to_numpy(np.float32)   # (N, D)
        Z = g[element_col].map(elem_to_id).to_numpy(np.int32)
        y = np.float32(g[energy_col].iloc[0])

        X_list.append(X)
        Z_list.append(Z)
        y_list.append(y)

    X_all = np.stack(X_list, axis=0)  # (F, N, D)
    Z_all = np.stack(Z_list, axis=0)  # (F, N)
    y_all = np.asarray(y_list)        # (F,)
    return X_all, Z_all, y_all, elem_to_id