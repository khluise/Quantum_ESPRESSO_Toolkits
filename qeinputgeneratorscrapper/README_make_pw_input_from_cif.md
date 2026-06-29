# make_pw_input_from_cif.py

Simple local Quantum ESPRESSO `pw.x` input generator from an ASE-supported structure file.

The script reads a structure using ASE, optionally refines the primitive cell using `spglib.find_primitive`, generates a k-point mesh from a user-controlled reciprocal-space spacing, and writes a QE SCF input file.
---

# Features

- Read CIF, XCrySDen XSF, CASTEP `.cell`, and VASP POSCAR/CONTCAR structures with ASE
- Optional primitive-cell refinement using spglib
- Automatic k-point mesh generation
- Automatic chemical formula prefix
- Dummy pseudopotential filenames generated automatically
- Generates:
  - `&CONTROL`
  - `&SYSTEM`
  - `&ELECTRONS`
  - `ATOMIC_SPECIES`
  - `ATOMIC_POSITIONS crystal`
  - `K_POINTS automatic`
  - `CELL_PARAMETERS angstrom`

---

# Installation

Install dependencies:

```bash
pip install ase numpy spglib
```

---

# Basic Usage

Generate `scf.in` from a structure file:

```bash
python make_pw_input_from_cif.py structure.cif
```

This writes:

```text
scf.in
```

---

# Custom Output Filename

```bash
python make_pw_input_from_cif.py structure.cif my_scf.in
```

---

# Supported Structure Inputs

| Input | ASE format |
|---|---|
| `.cif` | `cif` |
| `.xsf` | `xsf` |
| `.cell` | `castep-cell` |
| `POSCAR`, `CONTCAR`, `.vasp`, `.poscar` | `vasp` |

Examples:

```bash
python make_pw_input_from_cif.py structure.xsf scf.in
python make_pw_input_from_cif.py structure.cell scf.in
python make_pw_input_from_cif.py POSCAR scf.in
```

---

# User Options

## Protocol

Default:

```text
balanced
```

Example:

```bash
python make_pw_input_from_cif.py structure.cif \
  --protocol stringent
```

Available values:

| Protocol | k-point spacing | degauss | ecutwfc | ecutrho | etot per atom | conv per atom | forc conv thr |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fast` | 0.30 | 0.0275 | 35 | 280 | 1.0e-4 | 4.0e-10 | 1.0e-3 |
| `balanced` | 0.15 | 0.0200 | 35 | 280 | 1.0e-5 | 2.0e-10 | 1.0e-4 |
| `stringent` | 0.10 | 0.0125 | 60 | 480 | 5.0e-6 | 1.0e-10 | 5.0e-5 |

---

## Magnetism / occupations

Default:

```text
non-magnetic-fractional
```

For metals:

```bash
python make_pw_input_from_cif.py structure.cif \
  --magnetism non-magnetic-fractional
```

This writes:

```text
occupations = 'smearing'
smearing = 'cold'
```

For insulators:

```bash
python make_pw_input_from_cif.py structure.cif \
  --magnetism non-magnetic-fixed
```

This writes:

```text
occupations = 'fixed'
```

If `non-magnetic-fixed` is selected for a structure with an odd total electron count, the script prints a warning and generates the non-magnetic fractional-occupation input instead.

---

## K-point spacing

Default comes from `--protocol`. For the default `balanced` protocol:

```text
0.15 1/angstrom
```

Example:

```bash
python make_pw_input_from_cif.py structure.cif \
  --kpoint-spacing 0.15
```

Smaller spacing:

```text
0.10
```

→ denser mesh

Larger spacing:

```text
0.20
```

→ coarser mesh

---

## Structure refinement

Default:

```text
yes
```

Uses:

```python
spglib.find_primitive(...)
```

Disable refinement:

```bash
python make_pw_input_from_cif.py structure.cif \
  --refine no
```

---

## spglib symmetry tolerance

Default:

```text
1e-5
```

Example:

```bash
python make_pw_input_from_cif.py structure.cif \
  --symprec 1e-6
```

Larger values:

```text
1e-4
```

→ more aggressive symmetry detection

Smaller values:

```text
1e-7
```

→ stricter symmetry matching

---

# Full Example

```bash
python make_pw_input_from_cif.py ZrAsRu.cif \
  output.in \
  --protocol stringent \
  --magnetism non-magnetic-fractional \
  --refine yes \
  --symprec 1e-5
```

---

# Generated QE Parameters

The script uses the selected protocol for numerical thresholds and the selected magnetism mode for occupations. With the default `balanced` protocol and `non-magnetic-fractional` magnetism:

```text
calculation = 'scf'
occupations = 'smearing'
smearing = 'cold'
degauss = 0.0200 Ry

ecutwfc = 35 Ry
ecutrho = 280 Ry

etot_conv_thr = nat * 1e-5
conv_thr      = nat * 2e-10
forc_conv_thr = 1e-4
```

---

# Automatic Prefix

The QE prefix is automatically generated from the chemical formula.

Example:

```text
prefix = 'Zr3Ru3As3'
```

or:

```text
prefix = 'Mg2B2'
```

---

# Dummy Pseudopotential Names

The script does NOT check or download pseudopotentials.

It automatically writes placeholder names:

```text
Mg.pseudo_file.UPF
B.pseudo_file.UPF
Zr.pseudo_file.UPF
```

You should replace them with real `.UPF` files before production calculations.

---

# K-point Generation Method

The k-point mesh is generated from reciprocal lattice vector lengths:

```text
N_i = ceil(|b_i| / spacing)
```

where:

```text
|b_i|
```

includes the standard `2π` factor.

Example output:

```text
K_POINTS automatic
11 11 17 0 0 0
```

---


# Notes

- The script is intended for quick QE input generation.
- The default cutoffs are generic placeholders.
- Real production calculations require:
  - converged cutoffs
  - real pseudopotentials
  - convergence testing

---
