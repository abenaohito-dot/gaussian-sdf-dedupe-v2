import streamlit as st
import json
import time
from dataclasses import asdict
from engine import Thresholds, deduplicate
from gaussian_audit import annotate_gaussian
from batch_io import inventory, iter_inputs
import pandas as pd
import tempfile
import subprocess
import re
import io
import math
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Gaussian / SDF Dedupe App",
    layout="wide",
)

# =========================
# App metadata
# =========================
APP_VERSION = "2.0.3"

DEVELOPER_INFO = {
    "name": "Ken-ichi Nakashima",
    "affiliation_en": "Aichi-Gakuin University, School of Pharmacy, Laboratory of Natural Resources",
}

# Internal property names used in this app
APP_ENERGY_PROP = "AppEnergy"
APP_ENERGY_TYPE_PROP = "AppEnergyType"
APP_ENERGY_UNIT_PROP = "AppEnergyUnit"
APP_ENERGY_SOURCE_PROP = "AppEnergySource"
APP_SOURCE_FILE_PROP = "SourceFile"
APP_SOURCE_LABEL_PROP = "SourceLabel"
APP_INPUT_TYPE_PROP = "InputType"
APP_RECORD_INDEX_PROP = "RecordIndex"
APP_NORMAL_TERM_PROP = "NormalTermination"

# =========================
# Regex / constants
# =========================
SCF_RE = re.compile(r"SCF Done:\s+E\([RU]?\w+\)\s+=\s+(-?\d+\.\d+)")
ARCHIVE_HF_RE = re.compile(r"\|HF=(-?\d+\.\d+)")
FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?")

GIBBS_KEY = "Sum of electronic and thermal Free Energies"
NORMAL_TERM_KEY = "Normal termination of Gaussian"

ENERGY_HARTREE_TO_KCAL = 627.509474
ENERGY_KJ_TO_KCAL = 1.0 / 4.184

COMMON_SDF_ENERGY_PROPS = [
    "Energy",
    "energy",
    "ENERGY",
    "MMFF94 Energy",
    "MMFF94_energy",
    "CONFLEX Energy",
    "CONFLEX_Energy",
    "Total Energy",
    "Final Energy",
    "E",
    "SCF",
    "Free Energy",
    "Gibbs Free Energy",
]

# =========================
# UI text
# =========================
UI_TEXT = {'en': {'language_selector': 'Language', 'title': 'Gaussian / SDF Dedupe App', 'caption': f'Ver. {APP_VERSION}', 'description': 'Convert Gaussian logs, read conformer SDF files, and identify duplicates using atom mappings, RMSD, OH orientation, torsions, ring puckering and energy checks.', 'info': 'This app supports Gaussian .log files and conformer .sdf files (e.g., CONFLEX output). For SDF input, select the energy property and unit appropriately. If you mix Gaussian and SDF files in one run, energy ranking is only meaningful when the energy scales are comparable.', 'settings': 'Settings', 'developer_info': 'Developer information', 'developer_name': 'Name', 'developer_affiliation': 'Affiliation', 'log_energy_header': 'Gaussian .log settings', 'energy_to_extract': 'Energy to extract from Gaussian .log', 'scf': 'SCF', 'free_energy': 'Free Energy', 'sdf_energy_header': 'SDF settings', 'sdf_energy_mode': 'SDF energy property', 'auto_detect': 'Auto-detect common energy property', 'manual_property': 'Specify property name manually', 'sdf_property_name': 'SDF property name', 'sdf_property_hint': 'Examples: Energy, MMFF94 Energy, CONFLEX Energy', 'sdf_energy_unit': 'SDF energy unit', 'hartree': 'Hartree', 'kcal_mol': 'kcal/mol', 'kj_mol': 'kJ/mol', 'unknown_unit': 'Unknown (do not calculate relative energy)', 'symmetry_header': 'RMSD settings', 'symmetry_handling': 'Symmetry handling', 'symmetric_present': 'Symmetric structure present', 'no_symmetric_concern': 'No symmetric structure concern', 'rmsd_getbestrms': 'RMSD method: GetBestRMS (symmetry-aware)', 'rmsd_alignmol': 'RMSD method: AlignMol (direct alignment)', 'rmsd_cutoff': 'RMSD cutoff (Å)', 'rmsd_cutoff_help': 'Screening band only; additional checks are required.', 'heavy_atom_rmsd': 'Use heavy-atom RMSD (remove H atoms for RMSD calculation)', 'clear_results': 'Clear results', 'upload_files': 'Upload Gaussian .log, conformer .sdf, or ZIP archives', 'run_button': 'Run processing and filtering', 'need_file': 'Please upload at least one .log or .sdf file.', 'obabel_not_available': 'Open Babel is not available.', 'processing': 'Processing', 'conversion_finished': 'File processing finished.', 'no_valid_molecules': 'No valid structures could be loaded.', 'results': 'Results', 'uploaded_files_metric': 'Uploaded files', 'loaded_conformers_metric': 'Loaded conformers', 'kept_structures_metric': 'Structures kept', 'summary_table': 'Summary table', 'download_outputs': 'Download outputs', 'download_all': 'Download all_conformers.sdf', 'download_unique': 'Download unique_conformers.sdf', 'download_summary': 'Download summary.csv', 'show_details': 'Show processing details', 'no_details': 'No processing issues were recorded.', 'mixed_input_warning': 'Both Gaussian .log and SDF files are included. Representative selection is only meaningful when all uploaded energies are on a comparable scale.', 'status_kept': 'kept', 'status_removed_as_duplicate': 'removed as duplicate', 'status_energy_not_found': 'energy not found', 'status_conversion_failed': 'conversion failed', 'status_rdkit_read_failed': 'RDKit read failed', 'status_unsupported_file_type': 'unsupported file type', 'status_manual_property_not_found': 'specified SDF property not found'}}

COLUMN_LABELS = {'en': {'source_label': 'source_label', 'source_file': 'source_file', 'input_type': 'input_type', 'record_index': 'record_index', 'energy_type': 'energy_type', 'energy_property': 'energy_property', 'energy_unit': 'energy_unit', 'energy_value': 'energy_value', 'relative_energy_kcal_mol': 'relative_energy_kcal_mol', 'rank_by_energy': 'rank_by_energy', 'status': 'status', 'duplicate_of': 'duplicate_of', 'rmsd_to_representative': 'rmsd_to_representative', 'normal_termination': 'normal_termination'}}

# =========================
# Session state initialization
# =========================
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

if "result_payload" not in st.session_state:
    st.session_state.result_payload = None

if "ui_language" not in st.session_state:
    st.session_state.ui_language = "English"


# =========================
# Helper functions
# =========================
def get_texts():
    return "en", UI_TEXT["en"]


def check_obabel_available():
    try:
        result = subprocess.run(
            ["obabel", "-H"],
            capture_output=True,
            text=True,
            timeout=20
        )
        if result.returncode == 0:
            return True, "Open Babel detected."
        return False, f"Open Babel returned non-zero exit status.\nSTDERR:\n{result.stderr}"
    except FileNotFoundError:
        return False, "Open Babel ('obabel') was not found."
    except Exception as e:
        return False, f"Failed to run obabel: {e}"


def save_uploaded_file(uploaded_file, output_path: Path):
    if hasattr(uploaded_file, 'save_to'):
        uploaded_file.save_to(output_path)
    else:
        output_path.write_bytes(uploaded_file.getbuffer())


def parse_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None

    text = str(value).strip()
    match = FLOAT_RE.search(text)
    if not match:
        return None

    try:
        result = float(match.group(0).replace("D", "E").replace("d", "e"))
        return result if math.isfinite(result) else None
    except Exception:
        return None


def sanitize_filename_part(text):
    text = Path(text).stem
    text = re.sub(r"[^\w\-\.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_.-")
    return text or "output"


def longest_common_prefix(strings):
    if not strings:
        return ""

    prefix = strings[0]
    for s in strings[1:]:
        i = 0
        max_len = min(len(prefix), len(s))
        while i < max_len and prefix[i] == s[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return prefix


def clean_common_prefix(prefix):
    prefix = prefix.strip("_.- ")
    prefix = re.sub(r"[_\-.]+$", "", prefix)
    return prefix


def build_output_prefix(uploaded_files, min_prefix_len=3, fallback="output"):
    stems = [sanitize_filename_part(f.name) for f in uploaded_files]

    if not stems:
        return fallback

    if len(stems) == 1:
        return stems[0]

    prefix = longest_common_prefix(stems)
    prefix = clean_common_prefix(prefix)

    if len(prefix) >= min_prefix_len:
        return prefix

    return fallback


def extract_last_scf_energy(logfile: Path):
    last = None
    archive_last = None

    with open(logfile, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = SCF_RE.search(line)
            if m:
                last = float(m.group(1))

            for m2 in ARCHIVE_HF_RE.finditer(line):
                archive_last = float(m2.group(1))

    if last is not None:
        return last
    return archive_last


def extract_gibbs_energy(logfile: Path):
    found = None
    with open(logfile, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if GIBBS_KEY in line:
                try:
                    found = float(line.split()[-1])
                except Exception:
                    pass
    return found


def check_normal_termination(logfile: Path):
    found = False
    with open(logfile, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if NORMAL_TERM_KEY in line:
                found = True
    return found


def convert_log_to_sdf(log_path: Path, sdf_path: Path):
    try:
        result = subprocess.run(
            ["obabel", str(log_path), "-O", str(sdf_path)],
            capture_output=True,
            text=True,
            timeout=120
        )
        success = (result.returncode == 0) and sdf_path.exists() and sdf_path.stat().st_size > 0
        return success, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def read_first_mol_from_sdf(sdf_path: Path):
    try:
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mols = [m for m in suppl if m is not None]
        if not mols:
            return None
        return mols[0]
    except Exception:
        return None


def read_all_mols_from_sdf(sdf_path: Path):
    return list(Chem.SDMolSupplier(str(sdf_path), removeHs=False))


def find_prop_name_case_insensitive(prop_dict, target_name):
    target_lower = target_name.strip().lower()
    for name in prop_dict:
        if name.lower() == target_lower:
            return name
    return None


def collect_numeric_props(mol):
    numeric_props = {}
    for name in mol.GetPropNames():
        value = parse_float(mol.GetProp(name))
        if value is not None:
            numeric_props[name] = value
    return numeric_props


def extract_energy_from_sdf_mol(mol, mode="auto", manual_property_name=""):
    numeric_props = collect_numeric_props(mol)

    if not numeric_props:
        return None, None, "energy_not_found"

    if mode == "manual":
        if not manual_property_name.strip():
            return None, None, "manual_property_not_found"

        matched_name = find_prop_name_case_insensitive(numeric_props, manual_property_name)
        if matched_name is None:
            return None, None, "manual_property_not_found"
        return numeric_props[matched_name], matched_name, "ok"

    for candidate in COMMON_SDF_ENERGY_PROPS:
        matched_name = find_prop_name_case_insensitive(numeric_props, candidate)
        if matched_name is not None:
            return numeric_props[matched_name], matched_name, "ok"

    keyword_priority = [
        "free",
        "gibbs",
        "energy",
        "scf",
        "mmff",
        "conflex",
        "forcefield",
        "enthalpy",
    ]

    scored = []
    for name, value in numeric_props.items():
        lname = name.lower()
        score = sum(1 for kw in keyword_priority if kw in lname)
        if score > 0:
            scored.append((score, name, value))

    if scored:
        scored.sort(key=lambda x: (-x[0], x[1].lower()))
        _, best_name, best_value = scored[0]
        return best_value, best_name, "ok"


    return None, None, "energy_not_found"


def annotate_mol(
    mol,
    source_file,
    source_label,
    input_type,
    record_index,
    energy_value,
    energy_type,
    energy_unit,
    energy_source,
    normal_termination
):
    mol.SetProp(APP_SOURCE_FILE_PROP, str(source_file))
    mol.SetProp(APP_SOURCE_LABEL_PROP, str(source_label))
    mol.SetProp(APP_INPUT_TYPE_PROP, str(input_type))
    mol.SetProp(APP_RECORD_INDEX_PROP, str(record_index))
    mol.SetProp(APP_ENERGY_TYPE_PROP, str(energy_type))
    mol.SetProp(APP_ENERGY_UNIT_PROP, str(energy_unit))
    mol.SetProp(APP_ENERGY_SOURCE_PROP, str(energy_source))
    mol.SetProp(APP_NORMAL_TERM_PROP, str(normal_termination))

    if energy_value is not None:
        mol.SetProp(APP_ENERGY_PROP, str(energy_value))


def get_energy(mol):
    if mol.HasProp(APP_ENERGY_PROP):
        return parse_float(mol.GetProp(APP_ENERGY_PROP))
    return None


def get_prop_or_blank(mol, prop_name):
    return mol.GetProp(prop_name) if mol.HasProp(prop_name) else ""


def write_sdf_bytes(mols):
    sio = io.StringIO()
    writer = Chem.SDWriter(sio)
    for mol in mols:
        writer.write(mol)
    writer.close()
    return sio.getvalue().encode("utf-8")


def calculate_rmsd(mol_a, mol_b, method="GetBestRMS", remove_hs_for_rmsd=True):
    try:
        a = Chem.RemoveHs(mol_a) if remove_hs_for_rmsd else mol_a
        b = Chem.RemoveHs(mol_b) if remove_hs_for_rmsd else mol_b

        if method == "AlignMol":
            rmsd = AllChem.AlignMol(a, b)
        else:
            rmsd = AllChem.GetBestRMS(a, b)
        return rmsd
    except Exception:
        return None


def convert_relative_energy_to_kcal(delta_value, unit):
    unit = (unit or "").strip().lower()

    if unit == "hartree":
        return delta_value * ENERGY_HARTREE_TO_KCAL
    if unit == "kcal/mol":
        return delta_value
    if unit == "kj/mol":
        return delta_value * ENERGY_KJ_TO_KCAL

    return None


def deduplicate_molecules(mols, rmsd_cutoff=0.20, remove_hs_for_rmsd=True, rmsd_method="GetBestRMS"):
    cfg = Thresholds(heavy=rmsd_cutoff, **v2_settings)
    def update(completed, total, comparisons, label, elapsed):
        progress.progress(0.45 + 0.45 * completed / max(total, 1))
        status_box.info(f"Step 2/3: Comparing conformers — {completed}/{total} processed, {comparisons:,} pair comparisons, {elapsed:.0f}s elapsed. Current: {label or 'preparing geometries'}")
    kept, rows, pairs = deduplicate(mols, cfg, symmetry=rmsd_method == "GetBestRMS", progress_callback=update, evidence_mode="representatives")
    st.session_state.v2_pairwise = {"thresholds": asdict(cfg), "evidence_scope": "Selected representative comparison per input; every candidate comparison was evaluated", "comparisons_evaluated": sum(row['comparisons_evaluated'] for row in rows), "pairs": pairs}
    return kept, rows


def make_summary_csv_bytes(summary_rows):
    df = pd.DataFrame(summary_rows)
    csv_text = df.to_csv(index=False)
    return df, csv_text.encode("utf-8")


def localize_status(value, texts):
    mapping = {
        "kept": texts["status_kept"],
        "removed_as_duplicate": texts["status_removed_as_duplicate"],
        "energy_not_found": texts["status_energy_not_found"],
        "conversion_failed": texts["status_conversion_failed"],
        "rdkit_read_failed": texts["status_rdkit_read_failed"],
        "unsupported_file_type": texts["status_unsupported_file_type"],
        "manual_property_not_found": texts["status_manual_property_not_found"],
    }
    return mapping.get(value, value)


def make_display_df(raw_df, lang, texts):
    if raw_df is None or raw_df.empty:
        return raw_df

    df = raw_df.copy()

    if "status" in df.columns:
        df["status"] = df["status"].map(lambda x: localize_status(x, texts))

    rename_map = COLUMN_LABELS[lang]
    df = df.rename(columns=rename_map)
    return df


def render_results(payload, lang, texts):
    st.subheader(texts["results"])

    if payload.get("mixed_input_types", False):
        st.warning(texts["mixed_input_warning"])

    col1, col2, col3 = st.columns(3)
    col1.metric(texts["uploaded_files_metric"], payload["uploaded_count"])
    col2.metric(texts["loaded_conformers_metric"], payload["loaded_count"])
    col3.metric(texts["kept_structures_metric"], payload["kept_count"])

    st.subheader(texts["summary_table"])
    st.dataframe(make_display_df(payload["summary_df"], lang, texts), width="stretch")

    st.download_button("Download representative evidence and atom mappings (JSON)", data=json.dumps(payload.get("pairwise", {}), ensure_ascii=False, indent=2), file_name="pairwise.json", mime="application/json")
    st.subheader(texts["download_outputs"])
    d1, d2, d3 = st.columns(3)

    output_prefix = payload.get("output_prefix", "output")

    d1.download_button(
        label=texts["download_all"],
        data=payload["all_sdf_bytes"],
        file_name=f"{output_prefix}_all_conformers.sdf",
        mime="chemical/x-mdl-sdfile"
    )

    d2.download_button(
        label=texts["download_unique"],
        data=payload["unique_sdf_bytes"],
        file_name=f"{output_prefix}_unique_conformers.sdf",
        mime="chemical/x-mdl-sdfile"
    )

    d3.download_button(
        label=texts["download_summary"],
        data=payload["summary_csv_bytes"],
        file_name=f"{output_prefix}_summary.csv",
        mime="text/csv"
    )

    with st.expander(texts["show_details"]):
        details_df = payload.get("processing_logs_df")
        if details_df is not None and not details_df.empty:
            st.dataframe(make_display_df(details_df, lang, texts), width="stretch")
        else:
            st.write(texts["no_details"])


# =========================
# Interface language: English
# =========================
lang, texts = get_texts()

# =========================
# Main header
# =========================
st.title(texts["title"])
st.caption(texts["caption"])
st.write(texts["description"])
st.info(texts["info"])
st.info("Only confirmed duplicates are excluded. Distinct conformers and cases needing review are retained. Optimization convergence does not establish a minimum without vibrational analysis.")

# =========================
# Sidebar settings
# =========================
with st.sidebar:
    st.header(texts["settings"])

    with st.expander(texts["developer_info"], expanded=False):
        st.write(f"**{texts['developer_name']}**: {DEVELOPER_INFO['name']}")
        st.write(f"**{texts['developer_affiliation']}**: {DEVELOPER_INFO['affiliation_en']}")

    st.subheader(texts["log_energy_header"])
    energy_type_ui = st.radio(
        texts["energy_to_extract"],
        options=[texts["scf"], texts["free_energy"]],
        index=0,
    )
    log_energy_type_label = "SCF" if energy_type_ui == texts["scf"] else "Free Energy"

    st.subheader(texts["sdf_energy_header"])
    sdf_energy_mode_ui = st.radio(
        texts["sdf_energy_mode"],
        options=[texts["auto_detect"], texts["manual_property"]],
        index=0,
    )
    sdf_energy_mode = "auto" if sdf_energy_mode_ui == texts["auto_detect"] else "manual"

    sdf_manual_property_name = ""
    if sdf_energy_mode == "manual":
        sdf_manual_property_name = st.text_input(
            texts["sdf_property_name"],
            value="Energy",
            help=texts["sdf_property_hint"],
        )

    sdf_energy_unit_ui = st.selectbox(
        texts["sdf_energy_unit"],
        options=[texts["kcal_mol"], texts["hartree"], texts["kj_mol"], texts["unknown_unit"]],
        index=0,
    )
    sdf_energy_unit_map = {
        texts["hartree"]: "Hartree",
        texts["kcal_mol"]: "kcal/mol",
        texts["kj_mol"]: "kJ/mol",
        texts["unknown_unit"]: "Unknown",
    }
    sdf_energy_unit = sdf_energy_unit_map[sdf_energy_unit_ui]
    sdf_context = st.text_input("Shared SDF calculation context (blank: separate files)", value="", help="Enter a shared ID only after confirming that the SDF files use the same calculation conditions and comparable energies.")

    st.subheader(texts["symmetry_header"])
    symmetry_mode = st.radio(
        texts["symmetry_handling"],
        options=[
            texts["symmetric_present"],
            texts["no_symmetric_concern"],
        ],
        index=0,
    )

    if symmetry_mode == texts["symmetric_present"]:
        rmsd_method = "GetBestRMS"
        st.caption(texts["rmsd_getbestrms"])
    else:
        rmsd_method = "AlignMol"
        st.caption(texts["rmsd_alignmol"])

    rmsd_cutoff = st.number_input(
        texts["rmsd_cutoff"],
        min_value=0.01,
        max_value=10.00,
        value=0.20,
        step=0.01,
        format="%.2f",
        help="Screening band only; a low heavy-atom RMSD alone does not confirm a duplicate."
    )

    remove_hs_for_rmsd = True
    st.caption("Heavy-atom alignment with consistent atom mappings for all-atom, OH, torsion and ring comparisons.")
    v2_settings = {}
    with st.expander("Ver.2 confirmation bands", expanded=False):
        st.warning("Provisional thresholds. Cases needing review are retained. Validate these settings against known conformers for your compound.")
        for key, label, default, step in [
            ("all_atom", "All-atom RMSD (Å)", 0.08, 0.01),
            ("oh_deg", "OH orientation (°)", 12.0, 1.0),
            ("torsion_deg", "Torsion difference (°)", 8.0, 1.0),
            ("ring_deg", "Cyclobutane ring torsion difference (°)", 3.0, 0.5),
            ("ring_height", "Signed ring out-of-plane displacement difference (Å)", 0.015, 0.005),
            ("energy_kcal", "Energy difference (kcal/mol)", 0.05, 0.01),
            ("separate_factor", "Separation factor for distinct conformers", 2.0, 0.1),
        ]:
            v2_settings[key] = st.number_input(label, value=default, min_value=1.1 if key == "separate_factor" else step, step=step, format="%.3f")

    if st.button(texts["clear_results"]):
        st.session_state.results_ready = False
        st.session_state.result_payload = None
        st.session_state.pop('v2_pairwise', None)
        st.rerun()

# =========================
# Main UI
# =========================
st.info("For large batches, upload ZIP archive(s) containing .log or .sdf files. ZIP members are processed one at a time to reduce memory use. Keep this tab open until processing completes.")
if 'upload_revision' not in st.session_state: st.session_state.upload_revision = 0

def clear_uploads():
    st.session_state.upload_revision += 1
    st.session_state.results_ready = False
    st.session_state.result_payload = None
    st.session_state.pop('v2_pairwise', None)

uploaded_files = st.file_uploader(
    texts["upload_files"],
    type=["log", "sdf", "zip"],
    accept_multiple_files=True,
    key=f"batch_input_{st.session_state.upload_revision}",
    max_upload_size=200
)

st.button("Clear uploaded files", on_click=clear_uploads)
uploaded_bytes = sum(getattr(f, 'size', len(f.getbuffer()) if hasattr(f, 'getbuffer') else 0) for f in (uploaded_files or []))
large_raw_batch = uploaded_bytes > 150 * 1024 * 1024 and any(Path(f.name).suffix.lower() != '.zip' for f in (uploaded_files or []))
if large_raw_batch:
    st.warning("This uncompressed upload is too large for reliable processing on this server. Clear uploaded files, then upload ZIP archive(s) instead. Each archive can be up to 200 MB.")
run_button = st.button(texts["run_button"], type="primary", disabled=large_raw_batch)

# =========================
# Run
# =========================
if run_button:
    try:
        st.session_state.results_ready = False
        st.session_state.result_payload = None
        st.session_state.pop('v2_pairwise', None)
        run_started = time.monotonic()
        if not uploaded_files:
            st.error(texts["need_file"])
            st.stop()
    
        output_prefix = build_output_prefix(uploaded_files)
    
        total_input_files, needs_obabel = inventory(uploaded_files)
        if needs_obabel:
            obabel_ok, obabel_msg = check_obabel_available()
            if not obabel_ok:
                st.error(texts["obabel_not_available"])
                st.code(obabel_msg)
                st.stop()
            st.success(obabel_msg)
    
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            loaded_mols = []
            processing_logs = []
    
            progress = st.progress(0)
            status_box = st.empty()
    
            total_files = total_input_files
            input_types_seen = set()
    
            for i, uploaded_file in enumerate(iter_inputs(uploaded_files), start=1):
                status_box.info(f"Step 1/3: Reading and converting file {i}/{total_files}: {uploaded_file.name} ({time.monotonic()-run_started:.0f}s elapsed)")
    
                suffix = Path(uploaded_file.name).suffix.lower()
                file_path = tmpdir / f"input_{i:05d}{suffix}"
                try:
                    save_uploaded_file(uploaded_file, file_path)
        
                    if suffix == ".log":
                        input_types_seen.add("log")
        
                        normal_term = check_normal_termination(file_path)
                        energy_value = extract_last_scf_energy(file_path) if log_energy_type_label == "SCF" else extract_gibbs_energy(file_path)
        
                        sdf_path = tmpdir / f"converted_{i:05d}.sdf"
                        conv_success, conv_stdout, conv_stderr = convert_log_to_sdf(file_path, sdf_path)
        
                        if not conv_success:
                            processing_logs.append({
                                "source_label": uploaded_file.name,
                                "source_file": uploaded_file.name,
                                "input_type": "log",
                                "record_index": 1,
                                "energy_type": log_energy_type_label,
                                "energy_property": "",
                                "energy_unit": "Hartree",
                                "energy_value": energy_value,
                                "relative_energy_kcal_mol": None,
                                "rank_by_energy": None,
                                "status": "conversion_failed",
                                "duplicate_of": "",
                                "rmsd_to_representative": None,
                                "normal_termination": normal_term,
                            })
                            progress.progress(0.45 * i / total_files)
                            continue
        
                        mol = read_first_mol_from_sdf(sdf_path)
                        if mol is None:
                            processing_logs.append({
                                "source_label": uploaded_file.name,
                                "source_file": uploaded_file.name,
                                "input_type": "log",
                                "record_index": 1,
                                "energy_type": log_energy_type_label,
                                "energy_property": "",
                                "energy_unit": "Hartree",
                                "energy_value": energy_value,
                                "relative_energy_kcal_mol": None,
                                "rank_by_energy": None,
                                "status": "rdkit_read_failed",
                                "duplicate_of": "",
                                "rmsd_to_representative": None,
                                "normal_termination": normal_term,
                            })
                            progress.progress(0.45 * i / total_files)
                            continue
        
                        annotate_mol(
                            mol=mol,
                            source_file=uploaded_file.name,
                            source_label=uploaded_file.name,
                            input_type="log",
                            record_index=1,
                            energy_value=energy_value,
                            energy_type=log_energy_type_label,
                            energy_unit="Hartree",
                            energy_source=log_energy_type_label,
                            normal_termination=normal_term,
                        )
                        annotate_gaussian(mol, file_path)
                        loaded_mols.append(mol)
        
                    elif suffix == ".sdf":
                        input_types_seen.add("sdf")
        
                        mols_in_sdf = read_all_mols_from_sdf(file_path)
                        if not mols_in_sdf:
                            processing_logs.append({
                                "source_label": uploaded_file.name,
                                "source_file": uploaded_file.name,
                                "input_type": "sdf",
                                "record_index": "",
                                "energy_type": "SDF",
                                "energy_property": sdf_manual_property_name if sdf_energy_mode == "manual" else "",
                                "energy_unit": sdf_energy_unit,
                                "energy_value": None,
                                "relative_energy_kcal_mol": None,
                                "rank_by_energy": None,
                                "status": "rdkit_read_failed",
                                "duplicate_of": "",
                                "rmsd_to_representative": None,
                                "normal_termination": "",
                            })
                            progress.progress(0.45 * i / total_files)
                            continue
        
                        for idx, mol in enumerate(mols_in_sdf, start=1):
                            if mol is None:
                                processing_logs.append({"source_label": f"{uploaded_file.name} [record {idx}]", "source_file": uploaded_file.name, "record_index": idx, "status": "rdkit_read_failed", "reason": "Unreadable SDF record; check source file"})
                                continue
                            energy_value, prop_name, extract_status = extract_energy_from_sdf_mol(
                                mol,
                                mode=sdf_energy_mode,
                                manual_property_name=sdf_manual_property_name,
                            )
        
                            source_label = f"{uploaded_file.name} [record {idx}]"
        
                            annotate_mol(
                                mol=mol,
                                source_file=uploaded_file.name,
                                source_label=source_label,
                                input_type="sdf",
                                record_index=idx,
                                energy_value=energy_value,
                                energy_type="SDF",
                                energy_unit=sdf_energy_unit,
                                energy_source=prop_name or (sdf_manual_property_name if sdf_energy_mode == "manual" else ""),
                                normal_termination="",
                            )
                            mol.SetProp("CalculationContext", sdf_context.strip() or f"sdf-file:{uploaded_file.name}")
                            loaded_mols.append(mol)
        
                            if extract_status == "manual_property_not_found":
                                processing_logs.append({
                                    "source_label": source_label,
                                    "source_file": uploaded_file.name,
                                    "input_type": "sdf",
                                    "record_index": idx,
                                    "energy_type": "SDF",
                                    "energy_property": sdf_manual_property_name,
                                    "energy_unit": sdf_energy_unit,
                                    "energy_value": None,
                                    "relative_energy_kcal_mol": None,
                                    "rank_by_energy": None,
                                    "status": "manual_property_not_found",
                                    "duplicate_of": "",
                                    "rmsd_to_representative": None,
                                    "normal_termination": "",
                                })
        
                    else:
                        processing_logs.append({
                            "source_label": uploaded_file.name,
                            "source_file": uploaded_file.name,
                            "input_type": "",
                            "record_index": "",
                            "energy_type": "",
                            "energy_property": "",
                            "energy_unit": "",
                            "energy_value": None,
                            "relative_energy_kcal_mol": None,
                            "rank_by_energy": None,
                            "status": "unsupported_file_type",
                            "duplicate_of": "",
                            "rmsd_to_representative": None,
                            "normal_termination": "",
                        })
        
                    progress.progress(0.45 * i / total_files)
        
                except Exception as file_error:
                    processing_logs.append({"source_label": uploaded_file.name, "source_file": uploaded_file.name, "status": "processing_failed", "reason": str(file_error)})
                finally:
                    file_path.unlink(missing_ok=True)
                    (tmpdir / f"converted_{i:05d}.sdf").unlink(missing_ok=True)

            status_box.info(f"Step 2/3: Preparing comparison of {len(loaded_mols)} conformers. All input files have been read.")
    
            if not loaded_mols:
                st.error(texts["no_valid_molecules"])
                if processing_logs:
                    st.dataframe(make_display_df(pd.DataFrame(processing_logs), lang, texts), width="stretch")
                st.stop()
    
            all_sdf_bytes = None
    
            kept_mols, summary_rows = deduplicate_molecules(
                mols=loaded_mols,
                rmsd_cutoff=rmsd_cutoff,
                remove_hs_for_rmsd=remove_hs_for_rmsd,
                rmsd_method=rmsd_method,
            )
    
            progress.progress(0.92)
            status_box.info("Step 3/3: Generating result table and downloads...")
            all_sdf_bytes = write_sdf_bytes(loaded_mols)
            unique_sdf_bytes = write_sdf_bytes(kept_mols)
    
            summary_rows.extend(processing_logs)
    
            summary_df, summary_csv_bytes = make_summary_csv_bytes(summary_rows)
            if not summary_df.empty:
                summary_df = summary_df.sort_values(
                    by=["status", "rank_by_energy", "source_label"],
                    na_position="last"
                ).reset_index(drop=True)
    
            processing_logs_df = pd.DataFrame(processing_logs) if processing_logs else pd.DataFrame()
    
            st.session_state.result_payload = {
                "pairwise": st.session_state.v2_pairwise,
                "uploaded_count": total_input_files,
                "loaded_count": len(loaded_mols),
                "kept_count": len(kept_mols),
                "summary_df": summary_df,
                "all_sdf_bytes": all_sdf_bytes,
                "unique_sdf_bytes": unique_sdf_bytes,
                "summary_csv_bytes": summary_csv_bytes,
                "processing_logs_df": processing_logs_df,
                "mixed_input_types": len(input_types_seen) > 1,
                "output_prefix": output_prefix,
            }
            st.session_state.results_ready = True
            progress.progress(1.0)
            status_box.success(f"Completed: {len(loaded_mols)} conformers processed, {len(kept_mols)} kept, {len(loaded_mols)-len(kept_mols)} confirmed duplicates excluded ({time.monotonic()-run_started:.0f}s). Results and downloads are below.")
    except Exception as error:
        st.session_state.results_ready = False
        st.session_state.result_payload = None
        st.error("Processing stopped before results were ready. No structures were deleted from your input files. See the error details below.")
        with st.expander("Processing error details", expanded=True):
            st.exception(error)

# render from session state
if st.session_state.results_ready and st.session_state.result_payload is not None:
    render_results(st.session_state.result_payload, lang, texts)

