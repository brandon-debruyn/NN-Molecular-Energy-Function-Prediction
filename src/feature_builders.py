import pandas as pd
import numpy as np
import math
from itertools import combinations_with_replacement

def add_pairwise_distance_features(dataset: pd.DataFrame):
    """
    Adds pairwise interatomic distance features to the DataFrame.
    """

    frames = sorted(dataset["frame"].unique())
    frame_counts = dataset.groupby("frame").size()
    if frame_counts.nunique() != 1:
        raise ValueError("Inconsistent atom counts across frames; cannot build fixed distance features.")
    n_atoms = frame_counts.iloc[0]

    # precompute upper-triangle indices 
    iu, ju = np.triu_indices(n_atoms, k=1)

    dist_columns = [f"dist_atom{i}_atom{j}" for i, j in zip(iu, ju)]

    frame_rows = []
    for frame in frames:
        g = dataset[dataset["frame"] == frame].sort_values("atom_index")

        coords = g[["x", "y", "z"]].to_numpy(dtype=float)
        diff = coords[:, None, :] - coords[None, :, :]     
        dist_matrix = np.linalg.norm(diff, axis=2)         

        dists = dist_matrix[iu, ju]

        row = {"frame": frame}
        row.update(dict(zip(dist_columns, dists)))
        frame_rows.append(row)

    dist_df = pd.DataFrame(frame_rows)
    dataset_out = dataset.merge(dist_df, on="frame", how="left")

    return dataset_out

def build_proton_transfer_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-frame proton transfer features (q, d1, d2).
    Returns a DataFrame indexed by frame.
    """
    frames = sorted(dataset["frame"].unique())
    rows = []

    for frame in frames:
        g = dataset[dataset["frame"] == frame].sort_values("atom_index")

        O = g[g["atomic_number"] == 8][["x", "y", "z"]].to_numpy(dtype=float)
        H = g[g["atomic_number"] == 1][["x", "y", "z"]].to_numpy(dtype=float)

        if O.shape[0] != 2 or H.shape[0] < 1:
            continue

        dHO = np.linalg.norm(H[:, None, :] - O[None, :, :], axis=2)
        h_idx = np.argmin(dHO.min(axis=1))
        d1, d2 = dHO[h_idx, 0], dHO[h_idx, 1]

        rows.append({"frame": frame, "pt_q": d1 - d2, "pt_d1": d1, "pt_d2": d2})

    return pd.DataFrame(rows).set_index("frame")

# atom-centered symmetry functions
def fc_cos(R, R_cut):
    """Cosine cutoff."""
    if R <= R_cut:
        return 0.5 * (math.cos(math.pi * (R / R_cut)) + 1.0)
    return 0.0

def compute_neighbors(positions, i, R_cut):
    """Return neighbor indices j != i within R_cut."""
    ri = positions[i]
    d = np.linalg.norm(positions - ri, axis=1)
    neigh = np.where((d <= R_cut) & (d > 0.0))[0]
    return neigh.tolist(), d  # also return full dists array for reuse if you want


def acsf_g2_radial_element_resolved(
    positions,
    elements,              # list/array of element strings, length N
    i,
    neigh_idx,
    R_cut,
    radial_params,         # list of dicts: [{"eta":..., "Rs":...}, ...]
    element_types=None     # optional ordered list of unique elements
):
    """
    Returns dict: keys like ("G2", Z, m) -> value
    where Z is neighbor element and m indexes radial_params.
    Definition:
      G2_i(Z,m) = sum_{j in neigh(i), elem(j)=Z} exp(-eta*(Rij - Rs)^2) * fc(Rij)
    """
    if element_types is None:
        element_types = sorted(set(elements))
    out = {}
    ri = positions[i]

    # pre-init
    for Z in element_types:
        for m, _ in enumerate(radial_params):
            out[("G2", Z, m)] = 0.0

    for j in neigh_idx:
        Zj = elements[j]
        R_ij = float(np.linalg.norm(positions[j] - ri))
        f = fc_cos(R_ij, R_cut)
        if f == 0.0:
            continue

        for m, p in enumerate(radial_params):
            eta = p["eta"]
            Rs  = p.get("Rs", 0.0)
            out[("G2", Zj, m)] += math.exp(-eta * (R_ij - Rs) ** 2) * f

    return out

def acsf_g4_angular_elementpair_resolved(
    positions,
    elements,              # list/array of element strings, length N
    i,
    neigh_idx,
    R_cut,
    angular_params,        
    element_types=None,    
    eps=1e-12
):
    """
    Returns dict: keys like ("G4", (Za,Zb), m) -> value  with Za<=Zb
    Behler G4-like definition (common in ACSF literature):
      G4_i(Za,Zb,m) =
        2^(1-zeta) * sum_{j<k in neigh(i), elem(j),elem(k) match Za,Zb}
          (1 + lambda*cos(theta_ijk))^zeta
          * exp(-eta*(Rij^2 + Rik^2 + Rjk^2))
          * fc(Rij) * fc(Rik)

    Notes:
    - Cutoff is applied ONLY to Rij and Rik (central-atom distances). This is the usual choice.
    - Includes Rjk in the exponential term (as in many G4 variants).
    """
    if element_types is None:
        element_types = sorted(set(elements))

    # all unordered element pairs (Za<=Zb)
    pairs = list(combinations_with_replacement(element_types, 2))

    out = {}
    for pair in pairs:
        for m, _ in enumerate(angular_params):
            out[("G4", pair, m)] = 0.0

    ri = positions[i]

    # loop over neighbor pairs j<k
    for a in range(len(neigh_idx)):
        j = neigh_idx[a]
        rj = positions[j]
        vij = rj - ri
        Rij = float(np.linalg.norm(vij))
        fc_ij = fc_cos(Rij, R_cut)
        if fc_ij == 0.0:
            continue

        for b in range(a + 1, len(neigh_idx)):
            k = neigh_idx[b]
            rk = positions[k]
            vik = rk - ri
            Rik = float(np.linalg.norm(vik))
            fc_ik = fc_cos(Rik, R_cut)
            if fc_ik == 0.0:
                continue

            # neighbor-neighbor distance
            Rjk = float(np.linalg.norm(rk - rj))

            # angle at i
            denom = (Rij * Rik) + eps
            cos_theta = float(np.dot(vij, vik) / denom)
            cos_theta = max(-1.0, min(1.0, cos_theta))

            # element pair key (unordered)
            Za, Zb = elements[j], elements[k]
            pair = tuple(sorted((Za, Zb)))

            for m, p in enumerate(angular_params):
                eta     = p["eta"]
                zeta    = p["zeta"]
                lambda_ = p["lambda"]

                ang = (1.0 + lambda_ * cos_theta) ** zeta
                rad = math.exp(-eta * (Rij*Rij + Rik*Rik + Rjk*Rjk))
                out[("G4", pair, m)] += (2.0 ** (1.0 - zeta)) * ang * rad * fc_ij * fc_ik

    return out

def add_acsf_columns(
    df,
    R_cut=5.5,
    radial_params=None,
    angular_params=None,
    element_col="element",
    pos_cols=("x", "y", "z"),
    frame_col="frame",
    atom_index_col="atom_index",
    prefix="ACSF"
):
    """
    Adds many descriptor columns to df:
      - Radial:  ACSF_G2_<Z>_<m>
      - Angular: ACSF_G4_<Za>-<Zb>_<m>
    Element-resolved and element-pair-resolved.
    """
    if radial_params is None:
        radial_params = [{"eta": e, "Rs": Rs} for e in (0.05, 0.5, 1.0, 2.0, 4.0, 8.0) for Rs in np.linspace(0.0, R_cut, 8)]
    if angular_params is None:
        angular_params = [{"eta": e, "zeta": z, "lambda": lam}
                          for e in (0.0005, 0.005)
                          for z in (1.0, 2.0, 4.0)
                          for lam in (-1.0, 1.0)]

    df = df.copy()
    element_types = sorted(df[element_col].unique().tolist())
    pairs = list(combinations_with_replacement(element_types, 2))

    # Create column names deterministically
    radial_colnames = []
    for Z in element_types:
        for m in range(len(radial_params)):
            radial_colnames.append(f"{prefix}_G2_{Z}_{m}")

    angular_colnames = []
    for (Za, Zb) in pairs:
        for m in range(len(angular_params)):
            angular_colnames.append(f"{prefix}_G4_{Za}-{Zb}_{m}")

    all_cols = radial_colnames + angular_colnames
    for c in all_cols:
        df[c] = np.nan

    # Fill per frame
    for frame_id, g in df.groupby(frame_col, sort=False):
        g_sorted = g.sort_values(atom_index_col)
        idx = g_sorted.index.to_numpy()

        positions = g_sorted.loc[:, list(pos_cols)].to_numpy(dtype=float)
        elems = g_sorted[element_col].astype(str).to_numpy()
        N = positions.shape[0]

        # compute features per atom
        feats = np.zeros((N, len(all_cols)), dtype=float)

        for i in range(N):
            neigh, _ = compute_neighbors(positions, i, R_cut)

            g2 = acsf_g2_radial_element_resolved(
                positions=positions,
                elements=elems,
                i=i,
                neigh_idx=neigh,
                R_cut=R_cut,
                radial_params=radial_params,
                element_types=element_types
            )
            g4 = acsf_g4_angular_elementpair_resolved(
                positions=positions,
                elements=elems,
                i=i,
                neigh_idx=neigh,
                R_cut=R_cut,
                angular_params=angular_params,
                element_types=element_types
            )

            # pack into vector in the same order as column names
            v = []
            for Z in element_types:
                for m in range(len(radial_params)):
                    v.append(g2[("G2", Z, m)])
            for (Za, Zb) in pairs:
                for m in range(len(angular_params)):
                    v.append(g4[("G4", (Za, Zb), m)])

            feats[i, :] = np.asarray(v, dtype=float)

        df.loc[idx, all_cols] = feats

    return df, all_cols, element_types
