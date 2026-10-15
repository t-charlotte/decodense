#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
from pyscf import gto, scf
from pyscf.opentrustregion import mf_to_otr

import decodense

# init molecule
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

# mo coefficients
mo_coeff = 2 * (mf.mo_coeff[:, occ_mo],)

# decomposition
# default pop_method = "mulliken"
decomp = decodense.DecompCls(part="atoms", part_method="ao")
res = decodense.main(mol, decomp, mf, mo_coeff)

print("Decomposition without atomic reference energies:")
print(res)

del(res)

# atomic reference energies
atom_ref_dict = {
    "H": -0.500429058214292,
    "O": -74.97343342944708,
}
# decomposition
# default pop_method = "mulliken"
decomp = decodense.DecompCls(part="atoms", part_method="ao",atom_ref=atom_ref_dict)
res = decodense.main(mol, decomp, mf, mo_coeff)

print("\nDecomposition with atomic reference energies:")
print(res)
