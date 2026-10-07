#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
from pyscf import gto, scf
from pyscf.opentrustregion import PipekMezeyOTR, mf_to_otr

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
# default part_method = "mo"
decomp = decodense.DecompCls(pop_method="iao", part="atoms")
res = decodense.main(mol, decomp, mf, mo_coeff)

print(res)
