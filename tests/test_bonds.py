#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import pytest
from pyscf import gto, scf

from decodense.bonds import bond_mbo


# helper: rhf water, occupied mo coefficients and occupations in the format of
# decodense.main
def _h2o_rhf_input():
    mol = gto.M(verbose=0, output=None, basis="sto-3g", atom="geom/h2o.xyz")
    mf = scf.RHF(mol).run()
    occ = np.where(mf.mo_occ > 0)[0]
    mo_coeff = (mf.mo_coeff[:, occ],) * 2
    mo_occ = (np.ones(occ.size),) * 2
    return mol, mf, mo_coeff, mo_occ


# bond_mbo
# the atom-to-atom-pair weights must have shape (natm, npairs), be zero for atom pairs
# that are not bonds and have rows that sum to one, with and without a bond mask
@pytest.mark.parametrize("pop_method", ["mulliken", "iao"])
@pytest.mark.parametrize("bond_mask", [None, np.array([True, True, False])])
def test_bond_mbo(pop_method, bond_mask):
    mol, mf, mo_coeff, mo_occ = _h2o_rhf_input()
    w = bond_mbo(mol, mf, mo_coeff, mo_occ, "minao", pop_method, False, 0, bond_mask)
    assert w.shape == (3, 3)
    if bond_mask is not None:
        assert np.allclose(w[:, ~bond_mask], 0.0)
    assert np.allclose(w.sum(axis=1), 1.0)
