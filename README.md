# Gaussian / SDF Dedupe App Ver.2.0.2

An independent Streamlit app for conservative conformer deduplication in natural products chemistry. Derived from Ken-ichi Nakashima's Gaussian / SDF Dedupe App v1.1 at https://github.com/nakashima-art/gaussian-sdf-dedupe-app. The original repository is unchanged.

Supports Gaussian .log conversion with Open Babel, multi-record SDF input (including CONFLEX), SCF/free-energy extraction, energy ranking within comparable calculation contexts, and SDF/CSV downloads. The interface and output labels are in English.

## Decisions

- **Confirmed duplicate**: A valid complete atom mapping passes every confirmation band, with comparable energies and verified Gaussian optimization/geometry where applicable. Only this class is excluded.
- **Distinct conformer**: Every enumerated valid mapping exceeds a geometric separation band. Retained.
- **Needs review**: Ambiguous geometry, missing information, incomparable energies, unverified optimization, mapping limits, or other uncertainty. Retained. The first representative also has this status because no pair comparison is available.

The comparison uses consistent stereochemistry-preserving atom mappings for heavy-atom alignment, all-atom RMSD, OH directions, proper and OH torsions, and cyclobutane ring torsions/signed out-of-plane displacement. No reflection is allowed. Hydrogen permutations are limited to the corresponding bonded atom, respecting isotopes. Missing energies never cause a structure to be omitted.

Thresholds are provisional and editable. Validate them against known conformers for your compound, especially subtle ring-puckering differences. An optimization converging does not establish a minimum without vibrational analysis. Multi-job Gaussian logs are retained for review unless the optimization/geometry correspondence can be verified; single-job logs are preferred.

## Deploy on Streamlit Community Cloud

Repository branch: `main`; entry point: `app.py`; Python: **3.13**. `requirements.txt` supplies Python dependencies and `packages.txt` installs Open Babel. No secrets are required.

## Run locally

Install Python 3.13 and Open Babel (`obabel` on PATH), then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Uploads are processed by the hosting server in temporary files. Download the outputs from the app. This deployment repository contains application source and configuration only; research validation inputs and their derived data are kept outside the deployment package.

## Attribution

Original developer: Ken-ichi Nakashima, Aichi-Gakuin University, School of Pharmacy, Laboratory of Natural Resources.

The v2.0 comparison engine and Gaussian provenance checks were developed as an independent local extension at the user's request. See `PROVENANCE.md` for the original source revision. No new license is asserted for the upstream-derived code.

## Large batches (v2.0.2)

For hundreds of Gaussian logs, upload ZIP archive(s), up to 200 MB per archive. Members are copied to a generated temporary path one at a time and deleted after conversion/audit. Expanded ZIP inputs are not accumulated in RAM. Large raw uploads are blocked before processing; use ZIP instead.

The UI shows reading, comparison and output stages with counts and elapsed time. The hosted app stores only the selected representative evidence for each input, while evaluating all candidate comparisons needed for the same decisions. The JSON declares its evidence scope and total comparison count. The local CLI can still collect full pair evidence.

Validation on 203 structures with 20,503 evaluated comparisons reduced stored JSON from 29.155 MB to 0.272 MB; peak process RSS in this local benchmark fell from 264.6 MB to 79.2 MB (excluding uploaded research logs and Streamlit). These measurements do not guarantee that every dataset will fit within the hosting service's resource limits.
