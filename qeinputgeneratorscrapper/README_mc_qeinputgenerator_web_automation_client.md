# Materials Cloud QE Input Generator Automation

This README documents how to use the requests-based script that submits a structure file to the Materials Cloud Quantum ESPRESSO input generator, extracts the generated `scf.in`, captures any warning messages from the result page, and optionally downloads the ZIP file containing the input plus pseudopotentials.

Main script:

```bash
mc_qeinputgenerator_web_automation_client.py
```

Option-testing script:

```bash
test_html_options.py
```

---

## 1. What this workflow does

This workflow automates these manual website steps:

1. Open the Materials Cloud QE input generator
2. Upload a structure file
3. Select the file format
4. Select protocol, XC functional, magnetism/smearing, refinement, pseudopotential, k-point, and threshold settings
5. Generate the Quantum ESPRESSO `pw.x` input
6. Save `scf.in`
7. Save the full `result_page.html`
8. Capture warning messages, if present
9. Optionally download the ZIP file containing `scf.in` and pseudopotentials

Result:

```text
structure file → Materials Cloud QE input generator → scf.in + warnings.txt + pseudopotentials ZIP
```

---

## 2. Install requirements

```bash
pip install requests beautifulsoup4
```

---

## 3. Basic usage

For a CIF file:

```bash
python mc_qeinputgenerator_web_automation_client.py your_file.cif \
  --out qe_out \
  --download-zip
```

Expected output:

```text
qe_out/
├── scf.in
├── result_page.html
├── warnings.txt                 # only created if warnings are found
└── qe_input_and_pseudos.zip      # only if --download-zip is used
```

Extract the ZIP:

```bash
cd qe_out
unzip qe_input_and_pseudos.zip
```

Then run Quantum ESPRESSO:

```bash
pw.x < scf.in > scf.out
```

---

## 4. Capturing warnings from `result_page.html`

The script reads the generated HTML page and looks for warning-like containers such as:

```text
.alert-warning
.warning
[class*='warning']
[class*='alert']
[id*='warning']
[role='alert']
```

It also searches for text containing `warning` or `WARNING:`.

If warnings are found, the script prints them and saves them to:

```text
OUTPUT_FOLDER/warnings.txt
```

Example output:

```text
Captured 1 warning(s): qe_out/warnings.txt
WARNING 1: Warning: the input structure was refined using spglib ...
```

To make the script fail when warnings are detected:

```bash
python mc_qeinputgenerator_web_automation_client.py your_file.cif \
  --out qe_out \
  --download-zip \
  --fail-on-warning
```

This is useful in batch workflows where you want to manually inspect any structure that produces a warning.

---

## 5. Select the file format

The website field is:

```text
Select the file format:
```

In the script, pass it using:

```bash
--file-format VALUE
```

The alias also works:

```bash
--fileformat VALUE
```

The discovered backend values are:

| Value | Website label |
|---|---|
| `qeinp-qetools` | Quantum ESPRESSO input [parser: qetools] |
| `cif-ase` | CIF File (.cif) [parser: ase] |
| `cif-pymatgen` | CIF File (.cif) [parser: pymatgen] |
| `xyz-ase` | XYZ File (.xyz) [parser: ase] |
| `xsf-ase` | XCrySDen (.xsf) [parser: ase] |
| `vasp-ase` | VASP POSCAR [parser: ase] |
| `castep-ase` | CASTEP (.cell) [parser: ase] |
| `pdb-ase` | Protein Data Bank format (.pdb) [parser: ase] |

To re-check the current website values:

```bash
python mc_qeinputgenerator_web_automation_client.py --list-form-options
```
Currently only the `cif-ase` is tested.

---

## 6. File-format examples

### Quantum ESPRESSO input

```bash
python mc_qeinputgenerator_web_automation_client.py input.in \
  --file-format qeinp-qetools \
  --out qe_from_input \
  --download-zip
```

### CIF using ASE parser

```bash
python mc_qeinputgenerator_web_automation_client.py your_file.cif \
  --file-format cif-ase \
  --out qe_cif_ase \
  --download-zip
```

### CIF using pymatgen parser

```bash
python mc_qeinputgenerator_web_automation_client.py your_file.cif \
  --file-format cif-pymatgen \
  --out qe_cif_pymatgen \
  --download-zip
```

### XYZ

```bash
python mc_qeinputgenerator_web_automation_client.py structure.xyz \
  --file-format xyz-ase \
  --out qe_xyz \
  --download-zip
```

Important for XYZ: the web form has cell-vector fields. The script currently submits the same default/blank values as the form. For periodic systems in XYZ format, check the generated `CELL_PARAMETERS` carefully. This is not yet supported.

### XCrySDen XSF

```bash
python mc_qeinputgenerator_web_automation_client.py structure.xsf \
  --file-format xsf-ase \
  --out qe_xsf \
  --download-zip
```

### VASP POSCAR

```bash
python mc_qeinputgenerator_web_automation_client.py POSCAR \
  --file-format vasp-ase \
  --out qe_poscar \
  --download-zip
```

### CASTEP `.cell`

```bash
python mc_qeinputgenerator_web_automation_client.py structure.cell \
  --file-format castep-ase \
  --out qe_castep \
  --download-zip
```

### PDB

```bash
python mc_qeinputgenerator_web_automation_client.py structure.pdb \
  --file-format pdb-ase \
  --out qe_pdb \
  --download-zip
```

---

## 7. Automatic file-format guessing

By default:

```bash
--file-format auto
```

The script guesses:

| Filename/type | Auto value |
|---|---|
| `.cif` | `cif-ase` |
| `.xyz` | `xyz-ase` |
| `.xsf` | `xsf-ase` |
| `.cell` | `castep-ase` |
| `.pdb` | `pdb-ase` |
| `POSCAR`, `CONTCAR`, `.vasp`, `.poscar` | `vasp-ase` |
| `.in`, `.pwi`, `.pw`, `.qe` | `qeinp-qetools` |

For maximum reliability, explicitly pass `--file-format`.

---

## 8. Other valid option values

These backend values were discovered from the real HTML form and tested successfully.

### `--protocol`

| Value | Website label |
|---|---|
| `fast` | fast |
| `balanced` | balanced |
| `stringent` | stringent |

---

### `--xc`

| Value | Website label |
|---|---|
| `PBEsol` | PBEsol |
| `PBE` | PBE |

---

### `--magnetization`

| Value | Website label |
|---|---|
| `non-magnetic-fractional` | non-magnetic metal, fractional occupations |
| `non-magnetic-fixed` | non-magnetic insulator, fixed occupations |
| `magnetic` | magnetic, fractional occupations |

---

### `--refine-cell`

| Value | Website label |
|---|---|
| `no` | No |
| `yes` | Yes, choose the symmetry tolerance below |

Example:

```bash
--refine-cell yes --tolerance -5
```

---

### `--pseudo-override`

| Value | Website label |
|---|---|
| `""` | Protocol default |
| `efficiency` | SSSP 1.3: Efficiency |
| `precision` | SSSP 1.3: Precision |

For protocol default:

```bash
--pseudo-override ""
```

---

### `--kmesh-override`

| Value | Website label |
|---|---|
| `""` | Protocol default |
| `coarse` | coarse, 0.30 1/Å, 0.0275 Ry |
| `medium` | medium, 0.15 1/Å, 0.0200 Ry |
| `fine` | fine, 0.10 1/Å, 0.0125 Ry |

For protocol default:

```bash
--kmesh-override ""
```

---

### `--threshold-override`

| Value | Website label |
|---|---|
| `""` | Protocol default |
| `loose` | loose |
| `intermediate` | intermediate |
| `tight` | tight |

For protocol default:

```bash
--threshold-override ""
```

---

## 9. Full examples

### Fast PBE non-magnetic metal

```bash
python mc_qeinputgenerator_web_automation_client.py your_file.cif \
  --file-format cif-ase \
  --out qe_fast_pbe_metal \
  --download-zip \
  --protocol fast \
  --xc PBE \
  --magnetization non-magnetic-fractional \
  --refine-cell yes \
  --pseudo-override efficiency \
  --kmesh-override coarse \
  --threshold-override loose
```

### Stringent PBEsol magnetic

```bash
python mc_qeinputgenerator_web_automation_client.py your_file.cif \
  --file-format cif-ase \
  --out qe_stringent_pbesol_magnetic \
  --download-zip \
  --protocol stringent \
  --xc PBEsol \
  --magnetization magnetic \
  --refine-cell yes \
  --pseudo-override precision \
  --kmesh-override fine \
  --threshold-override tight
```

### Use protocol defaults for all override fields

```bash
python mc_qeinputgenerator_web_automation_client.py your_file.cif \
  --file-format cif-ase \
  --out qe_protocol_defaults \
  --download-zip \
  --protocol balanced \
  --xc PBEsol \
  --magnetization magnetic \
  --refine-cell yes \
  --pseudo-override "" \
  --kmesh-override "" \
  --threshold-override ""
```

---

## 10. Testing option values

Use:

```bash
python test_html_options.py your_file.cif \
  --generator-script mc_qeinputgenerator_web_automation_client.py \
  --out mc_real_option_tests \
  --baseline-file-format cif-ase
```

This script:

1. Downloads the current Materials Cloud generator page
2. Reads the real `<option value="...">` values
3. Tests each non-fileformat option one by one using the generator script
4. Writes a CSV report

Check failures:

```bash
grep False mc_real_option_tests/real_option_test_results.csv
```

If all values pass, this command prints nothing.

---

## 11. Discover values only

To print current website values without submitting any structure:

```bash
python test_html_options.py --discover-only
```

or:

```bash
python mc_qeinputgenerator_web_automation_client.py --list-form-options
```

---

## 12. Testing file-format values

File-format testing is different from testing other dropdowns.

A CIF file can be parsed by CIF parsers, but it should not be expected to pass with an XSF, POSCAR, CASTEP, or PDB parser. Therefore `fileformat` tests are skipped by default.

To include file-format tests using the same uploaded file:

```bash
python test_html_options.py your_file.cif \
  --generator-script mc_qeinputgenerator_web_automation_client.py \
  --out mc_fileformat_tests \
  --baseline-file-format cif-ase \
  --include-file-format-tests
```

With a CIF input, expect only CIF-related parser values to pass. Failures for XSF/POSCAR/CASTEP/PDB are not necessarily script problems; they usually mean the selected parser does not match the file content.

Recommended method:

```text
Use a real sample file for each parser you want to test.
```

Examples:

```bash
python mc_qeinputgenerator_web_automation_client.py sample.cif   --file-format cif-ase
python mc_qeinputgenerator_web_automation_client.py sample.cif   --file-format cif-pymatgen
python mc_qeinputgenerator_web_automation_client.py sample.xsf   --file-format xsf-ase
python mc_qeinputgenerator_web_automation_client.py POSCAR       --file-format vasp-ase
python mc_qeinputgenerator_web_automation_client.py sample.cell  --file-format castep-ase
python mc_qeinputgenerator_web_automation_client.py sample.pdb   --file-format pdb-ase
python mc_qeinputgenerator_web_automation_client.py input.in     --file-format qeinp-qetools
```

---

## 13. Batch conversion

For many CIF files:

```bash
for f in *.cif; do
    name="${f%.cif}"

    python mc_qeinputgenerator_web_automation_client.py "$f" \
      --file-format cif-ase \
      --out "qe_${name}" \
      --download-zip \
      --protocol balanced \
      --xc PBEsol \
      --magnetization magnetic \
      --refine-cell yes \
      --pseudo-override precision \
      --kmesh-override fine \
      --threshold-override tight

    sleep 1
done
```

Batch conversion with warning stop:

```bash
for f in *.cif; do
    name="${f%.cif}"

    python mc_qeinputgenerator_web_automation_client.py "$f" \
      --file-format cif-ase \
      --out "qe_${name}" \
      --download-zip \
      --fail-on-warning

    sleep 1
done
```

Use `sleep 1` or `sleep 2` so you do not overload the public server.

---

## 14. Troubleshooting

### `Could not find QE input in the response HTML`

Usually one of these happened:

1. Wrong backend option value
2. Wrong file-format parser for the uploaded file
3. The uploaded file could not be parsed by the selected parser

Fix:

```bash
python mc_qeinputgenerator_web_automation_client.py --list-form-options
```

Then use exact backend values.

Also check:

```text
OUTPUT_FOLDER/result_page.html
```

---


### QE cannot find pseudopotentials

Check `pseudo_dir` inside `scf.in`.

Example:

```text
pseudo_dir = './pseudo'
```

Make sure the `.UPF` files from the ZIP are in that directory.

---

## 15. Scientific note

The generated input is a starting point. For production calculations, still test:

1. k-point convergence
2. cutoff convergence
3. force convergence
4. stress convergence
5. magnetic versus non-magnetic setup, when relevant
