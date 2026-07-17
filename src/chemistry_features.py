from __future__ import annotations

import numpy as np
import pandas as pd


ELEMENT_COLUMNS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
]
KEY_ELEMENTS = [
    "O", "Cu", "Ba", "Y", "Bi", "Sr", "Ca", "La", "Fe", "As", "Se", "Te",
    "Hg", "Tl", "Pb", "B", "Mg", "Ni", "Co", "P", "S",
]
RARE_EARTH_ELEMENTS = [
    "Sc", "Y", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
]
ATOMIC_NUMBER = {
    element: index + 1 for index, element in enumerate(ELEMENT_COLUMNS)
}
ATOMIC_MASS = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.811,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
    "S": 32.066, "Cl": 35.453, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.390,
    "Ga": 69.723, "Ge": 72.610, "As": 74.922, "Se": 78.960, "Br": 79.904,
    "Kr": 83.800, "Rb": 85.468, "Sr": 87.620, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.940, "Tc": 98.0, "Ru": 101.070, "Rh": 102.906,
    "Pd": 106.420, "Ag": 107.868, "Cd": 112.411, "In": 114.818, "Sn": 118.710,
    "Sb": 121.760, "Te": 127.600, "I": 126.904, "Xe": 131.290, "Cs": 132.905,
    "Ba": 137.327, "La": 138.906, "Ce": 140.116, "Pr": 140.908, "Nd": 144.240,
    "Pm": 145.0, "Sm": 150.360, "Eu": 151.964, "Gd": 157.250, "Tb": 158.925,
    "Dy": 162.500, "Ho": 164.930, "Er": 167.260, "Tm": 168.934, "Yb": 173.040,
    "Lu": 174.967, "Hf": 178.490, "Ta": 180.948, "W": 183.840, "Re": 186.207,
    "Os": 190.230, "Ir": 192.217, "Pt": 195.078, "Au": 196.967, "Hg": 200.590,
    "Tl": 204.383, "Pb": 207.200, "Bi": 208.980, "Po": 209.0, "At": 210.0,
    "Rn": 222.0,
}
ELECTRONEGATIVITY = {
    "H": 2.20, "He": 0.0, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55,
    "N": 3.04, "O": 3.44, "F": 3.98, "Ne": 0.0, "Na": 0.93, "Mg": 1.31,
    "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16, "Ar": 0.0,
    "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66,
    "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65,
    "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 3.00,
    "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.60, "Mo": 2.16,
    "Tc": 1.90, "Ru": 2.20, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69,
    "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.10, "I": 2.66, "Xe": 2.60,
    "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14,
    "Pm": 1.13, "Sm": 1.17, "Eu": 1.20, "Gd": 1.20, "Tb": 1.10, "Dy": 1.22,
    "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.10, "Lu": 1.27, "Hf": 1.30,
    "Ta": 1.50, "W": 2.36, "Re": 1.90, "Os": 2.20, "Ir": 2.20, "Pt": 2.28,
    "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33, "Bi": 2.02, "Po": 2.00,
    "At": 2.20, "Rn": 0.0,
}
PERIOD = {
    element: period
    for period, row in {
        1: ["H", "He"],
        2: ["Li", "Be", "B", "C", "N", "O", "F", "Ne"],
        3: ["Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"],
        4: ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"],
        5: ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"],
        6: ["Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"],
    }.items()
    for element in row
}
ELEMENT_GROUPS = {
    "alkali": ["Li", "Na", "K", "Rb", "Cs"],
    "alkaline_earth": ["Be", "Mg", "Ca", "Sr", "Ba"],
    "transition_metal": [
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    ],
    "lanthanide": ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"],
    "metalloid": ["B", "Si", "Ge", "As", "Sb", "Te"],
    "nonmetal": ["H", "C", "N", "O", "P", "S", "Se"],
    "halogen": ["F", "Cl", "Br", "I", "At"],
    "heavy_metal": ["Hg", "Tl", "Pb", "Bi"],
}


def _weighted_stats(fractions: pd.DataFrame, values: dict[str, float], prefix: str) -> pd.DataFrame:
    value_series = pd.Series(values, dtype=float).reindex(ELEMENT_COLUMNS).fillna(0.0)
    weighted_mean = fractions.mul(value_series, axis=1).sum(axis=1)
    value_array = value_series.to_numpy(dtype=float)
    centered = (value_array[None, :] - weighted_mean.to_numpy()[:, None]) ** 2
    weighted_std = np.sqrt((fractions.to_numpy(dtype=float) * centered).sum(axis=1))
    present_values = fractions.where(fractions > 0).mul(value_series, axis=1)
    max_value = present_values.max(axis=1).fillna(0.0)
    min_value = present_values.min(axis=1).fillna(0.0)
    return pd.DataFrame(
        {
            f"{prefix}_mean": weighted_mean,
            f"{prefix}_std": weighted_std,
            f"{prefix}_range": max_value - min_value,
            f"{prefix}_max": max_value,
            f"{prefix}_min": min_value,
        },
        index=fractions.index,
    )


def _stoich_distance(comp: pd.DataFrame, formula: dict[str, float]) -> pd.Series:
    formula_sum = sum(formula.values())
    target = {element: amount / formula_sum for element, amount in formula.items()}
    total = comp.sum(axis=1).replace(0, np.nan)
    fractions = comp.div(total, axis=0).fillna(0.0)
    distance = pd.Series(0.0, index=comp.index)
    for element, target_fraction in target.items():
        distance += (fractions[element] - target_fraction) ** 2
    extra_elements = [element for element in ELEMENT_COLUMNS if element not in formula]
    distance += fractions[extra_elements].sum(axis=1) ** 2
    return np.sqrt(distance)


def build_composition_features(unique_df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in ELEMENT_COLUMNS if col not in unique_df.columns]
    if missing:
        raise ValueError(f"unique_data is missing element columns: {missing[:10]}")

    comp = unique_df[ELEMENT_COLUMNS].astype(float).copy()
    total_atoms = comp.sum(axis=1).replace(0, np.nan)
    fractions = comp.div(total_atoms, axis=0).fillna(0.0)
    present = (comp > 0).astype(int)

    features = pd.DataFrame(index=unique_df.index)
    features["comp_total_atoms"] = total_atoms.fillna(0.0)
    features["comp_nonzero_elements"] = present.sum(axis=1)
    features["comp_max_fraction"] = fractions.max(axis=1)
    features["comp_min_nonzero_fraction"] = fractions.where(fractions > 0).min(axis=1).fillna(0.0)
    features["comp_fraction_entropy"] = -(
        fractions.where(fractions > 0, 1.0) * np.log(fractions.where(fractions > 0, 1.0))
    ).sum(axis=1)
    features["comp_fraction_herfindahl"] = (fractions ** 2).sum(axis=1)
    features["comp_fraction_evenness"] = features["comp_fraction_entropy"] / np.log(
        features["comp_nonzero_elements"].replace(0, np.nan)
    )
    features["comp_fraction_evenness"] = features["comp_fraction_evenness"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for element in KEY_ELEMENTS:
        features[f"amt_{element}"] = comp[element]
        features[f"frac_{element}"] = fractions[element]
        features[f"has_{element}"] = present[element]

    features["rare_earth_count"] = present[RARE_EARTH_ELEMENTS].sum(axis=1)
    features["rare_earth_amount"] = comp[RARE_EARTH_ELEMENTS].sum(axis=1)
    features["rare_earth_fraction"] = fractions[RARE_EARTH_ELEMENTS].sum(axis=1)

    for group_name, elements in ELEMENT_GROUPS.items():
        features[f"{group_name}_count"] = present[elements].sum(axis=1)
        features[f"{group_name}_amount"] = comp[elements].sum(axis=1)
        features[f"{group_name}_fraction"] = fractions[elements].sum(axis=1)

    for stats in [
        _weighted_stats(fractions, ATOMIC_NUMBER, "atomic_number"),
        _weighted_stats(fractions, ATOMIC_MASS, "atomic_mass"),
        _weighted_stats(fractions, ELECTRONEGATIVITY, "electronegativity"),
        _weighted_stats(fractions, PERIOD, "period"),
    ]:
        features = pd.concat([features, stats], axis=1)

    ratio_pairs = [
        ("Cu", "O"), ("O", "Cu"), ("Ba", "Cu"), ("Y", "Cu"), ("Bi", "Sr"),
        ("Sr", "Cu"), ("Ca", "Cu"), ("La", "Cu"), ("Hg", "Cu"), ("Tl", "Cu"),
        ("Pb", "Cu"), ("Fe", "As"), ("Fe", "Se"), ("Fe", "Te"), ("Mg", "B"),
        ("O", "Ba"), ("O", "Bi"), ("O", "Sr"), ("O", "Ca"),
    ]
    for numerator, denominator in ratio_pairs:
        features[f"ratio_{numerator}_to_{denominator}"] = (
            comp[numerator] / comp[denominator].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    features["is_cuprate"] = ((present["Cu"] == 1) & (present["O"] == 1)).astype(int)
    features["is_ybco_like"] = (
        (present["Y"] == 1) & (present["Ba"] == 1) & (present["Cu"] == 1) & (present["O"] == 1)
    ).astype(int)
    features["is_bi_based_cuprate"] = (
        (present["Bi"] == 1) & (present["Sr"] == 1) & (present["Cu"] == 1) & (present["O"] == 1)
    ).astype(int)
    features["is_hg_based_cuprate"] = (
        (present["Hg"] == 1) & (present["Cu"] == 1) & (present["O"] == 1)
    ).astype(int)
    features["is_tl_based_cuprate"] = (
        (present["Tl"] == 1) & (present["Cu"] == 1) & (present["O"] == 1)
    ).astype(int)
    features["is_iron_based"] = (
        (present["Fe"] == 1)
        & ((present["As"] == 1) | (present["Se"] == 1) | (present["Te"] == 1))
    ).astype(int)
    features["is_mgb2_like"] = ((present["Mg"] == 1) & (present["B"] == 1)).astype(int)
    features["cu_oxygen_product"] = comp["Cu"] * comp["O"]
    features["cu_oxygen_fraction_product"] = fractions["Cu"] * fractions["O"]
    features["cuprate_family_score"] = (
        features["is_cuprate"]
        + features["is_ybco_like"]
        + features["is_bi_based_cuprate"]
        + features["is_hg_based_cuprate"]
        + features["is_tl_based_cuprate"]
    )
    features["oxygen_transition_fraction_product"] = fractions["O"] * features["transition_metal_fraction"]
    features["oxygen_rare_earth_fraction_product"] = fractions["O"] * features["rare_earth_fraction"]
    features["heavy_cuprate_indicator"] = (
        (present["O"] == 1)
        & (present["Cu"] == 1)
        & ((present["Hg"] == 1) | (present["Tl"] == 1) | (present["Pb"] == 1) | (present["Bi"] == 1))
    ).astype(int)
    features["ybco_distance"] = _stoich_distance(comp, {"Y": 1, "Ba": 2, "Cu": 3, "O": 7})
    features["bi2212_distance"] = _stoich_distance(comp, {"Bi": 2, "Sr": 2, "Ca": 1, "Cu": 2, "O": 8})
    features["mgb2_distance"] = _stoich_distance(comp, {"Mg": 1, "B": 2})
    features["feas_distance"] = _stoich_distance(comp, {"Fe": 1, "As": 1})

    return features.copy()
