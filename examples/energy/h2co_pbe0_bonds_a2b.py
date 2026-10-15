#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
from pyscf import gto, scf
from pyscf.opentrustregion import PipekMezeyOTR, mf_to_otr
from pandas import set_option

import decodense

# init molecule (formaldehyde)
mol = gto.M(
    atom="""
    C     0.00072760  -0.00011425   0.00000109
    O     1.17836953  -0.18502406   0.00176610
    H    -0.73339368  -0.82354548  -0.00115967
    H    -0.44570339   1.00868375  -0.00060753
    """,
    verbose=0,
    output=None,
    basis="pcseg1",
)

# mf calc
mf = mf_to_otr(scf.RKS(mol, xc="pbe0"))
mf.conv_tol = 1.0e-10
mf.kernel()

# verify SCF solution is a true minimum
stable, direction = mf.stability_check()

# occupied orbitals
occ_mo = np.where(mf.mo_occ == 2.0)[0]

# pipek-mezey procedure with OTR
loc = PipekMezeyOTR(mol, mf.mo_coeff[:, occ_mo])
loc.pop_method = "iao"
loc.conv_tol = 1e-10
mo_coeff = loc.kernel()

# verify optimum is a true minimum
stable, direction = loc.stability_check()

# decomposition
# by default, all atom pairs will be considered for the decomposition
# verbose = 1 prints the intermediates
print("\nDecomposition without atomic reference energies nor bond mask:")
decomp = decodense.DecompCls(
    pop_method="iao", part="bonds", part_method="a2b", verbose=1
)
res = decodense.main(mol, decomp, mf, mo_coeff)

set_option("display.max_rows", None, "display.max_columns", None)
print(res)

del(res)

# atomic reference energies
atom_ref_dict = {
    "H": -0.500429058214292,
    "C": -37.63763262290564,
    "O": -74.97343342944708,
}

# decomposition
# by default, all atom pairs will be considered for the decomposition
# verbose = 1 prints the intermediates
print("\nDecomposition with atomic reference energies without bond mask:")
decomp = decodense.DecompCls(
    pop_method="iao", part="bonds", part_method="a2b", verbose=1, atom_ref=atom_ref_dict
)
res = decodense.main(mol, decomp, mf, mo_coeff)

set_option("display.max_rows", None, "display.max_columns", None)
print(res)

del(res)

# bond_mask
bond_mask = [
    (0,1), # C0-O1
    (0,2), # C0-H2
    (0,3), # C0-H3
]

# other valid bond mask:
# bond_mask = [
#     True, # C0-O1
#     True, # C0-H2
#     True, # C0-H3
#     False, # O1-H2
#     False, # O1-H3
#     False, # H2-H3
# ]

# decomposition
# by default, all atom pairs will be considered for the decomposition
# verbose = 1 prints the intermediates
print("\nDecomposition with bond mask without atomic reference energies:")
decomp = decodense.DecompCls(
    pop_method="iao", part="bonds", part_method="a2b", verbose=1, bond_mask=bond_mask
)
res = decodense.main(mol, decomp, mf, mo_coeff)

set_option("display.max_rows", None, "display.max_columns", None)
print(res)

del(res)

# decomposition
# verbose = 1 prints the intermediates
print("\nDecomposition with atomic reference energies and bond mask:")
decomp = decodense.DecompCls(
    pop_method="iao", part="bonds", part_method="a2b", verbose=1, atom_ref=atom_ref_dict, bond_mask=bond_mask
)
res = decodense.main(mol, decomp, mf, mo_coeff)

set_option("display.max_rows", None, "display.max_columns", None)
print(res)
