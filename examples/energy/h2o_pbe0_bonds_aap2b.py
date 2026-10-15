#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
from pyscf import gto, scf
from pyscf.opentrustregion import PipekMezeyOTR, mf_to_otr
from pandas import set_option

import decodense

# init molecule (water)
mol = gto.M(
    atom="""
        O  0.00000000  0.00000000  0.00000000
        H -0.75390364  0.00000000 -0.58783729
        H  0.75390364  0.00000000 -0.58783729
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
    pop_method="iao", part="bonds", part_method="aap2b", verbose=1
)
res = decodense.main(mol, decomp, mf, mo_coeff)

set_option("display.max_rows", None, "display.max_columns", None)
print(res)

del(res)

# atomic reference energies
atom_ref_dict = {
    "H": -0.500429058214292,
    "O": -74.97343342944708,
}

# decomposition
# by default, all atom pairs will be considered for the decomposition
# verbose = 1 prints the intermediates
print("\nDecomposition with atomic reference energies without bond mask:")
decomp = decodense.DecompCls(
    pop_method="iao", part="bonds", part_method="aap2b", verbose=1, atom_ref=atom_ref_dict
)
res = decodense.main(mol, decomp, mf, mo_coeff)

set_option("display.max_rows", None, "display.max_columns", None)
print(res)

del(res)

# bond mask
bond_mask = [
    (0,1), # O0-H1
    (0,2), # O0-H2
]

# other valid bond mask
# bond_mask = [
#     True, # O0-H1
#     True, # O0-H2
#     False, # H1-H2
# ]

# decomposition
# verbose = 1 prints the intermediates
print("\nDecomposition with bond mask without atomic reference energies:")
decomp = decodense.DecompCls(
    pop_method="iao", part="bonds", part_method="aap2b", verbose=1, bond_mask=bond_mask
)
res = decodense.main(mol, decomp, mf, mo_coeff)

set_option("display.max_rows", None, "display.max_columns", None)
print(res)

del(res)

# decomposition
# verbose = 1 prints the intermediates
print("\nDecomposition with atomic reference energies and bond mask:")
decomp = decodense.DecompCls(
    pop_method="iao", part="bonds", part_method="aap2b", verbose=1, atom_ref=atom_ref_dict, bond_mask=bond_mask
)
res = decodense.main(mol, decomp, mf, mo_coeff)

set_option("display.max_rows", None, "display.max_columns", None)
print(res)
