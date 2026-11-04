#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import pytest
from pyscf import gto, scf, dft

from decodense.decomp import CompKeys
from decodense.properties import prop_tot, aap_prop_tot, a2ap_redistribute
from decodense.tools import make_rdm1


# helper: converged hf/dft calculation of h2o (restricted) or h2o+ (unrestricted)
def _mf_h2o(restrict, xc):
    charge, spin = (0, 0) if restrict else (1, 1)
    mol = gto.M(
        verbose=0,
        output=None,
        basis="sto-3g",
        atom="geom/h2o.xyz",
        charge=charge,
        spin=spin,
    )
    if xc == "hf":
        mf = scf.RHF(mol) if restrict else scf.UHF(mol)
    else:
        mf = dft.RKS(mol, xc=xc) if restrict else dft.UKS(mol, xc=xc)
    return mf.run(conv_tol=1e-10)


# helper: occupied mo coefficients and occupations in the format of decodense.main,
# and random atom weights (identical for both spins if restricted)
def _decomp_input(mf, restrict):
    if restrict:
        mo_coeff_mf, mo_occ_mf = (mf.mo_coeff, mf.mo_coeff), (mf.mo_occ / 2,) * 2
    else:
        mo_coeff_mf, mo_occ_mf = mf.mo_coeff, mf.mo_occ
    occ = [np.where(o > 0)[0] for o in mo_occ_mf]
    mo_coeff = tuple(mo_coeff_mf[s][:, occ[s]] for s in range(2))
    mo_occ = tuple(np.ones(o.size) for o in occ)
    rng = np.random.default_rng(8)
    weights = [rng.random((o.size, mf.mol.natm)) for o in occ]
    if restrict:
        weights[1] = weights[0]
    return mo_coeff, mo_occ, weights


# helper: kinetic, nuclear attraction and coulomb energies of the total density
def _e_ref_components(mf):
    mol = mf.mol
    dm = mf.make_rdm1()
    dm_tot = dm if dm.ndim == 2 else dm[0] + dm[1]
    e_kin = np.einsum("ij,ij", mol.intor("int1e_kin"), dm_tot)
    e_nuc_att = np.einsum("ij,ij", mol.intor("int1e_nuc"), dm_tot)
    e_coul = 0.5 * np.einsum("ij,ij", scf.hf.get_jk(mol, dm_tot)[0], dm_tot)
    return e_kin, e_nuc_att, e_coul


# helper: sum of an atom-wise array or of orbital-wise arrays (one per spin)
def _total(v):
    return sum(np.sum(x) for x in v) if isinstance(v, list) else np.sum(v)


# prop_tot, energies
# atom-wise ("mo"/"ao") totals must sum to pyscf's total energy, orbital-wise totals
# to the electronic energy; the kinetic, nuclear attraction (local + global for atoms)
# and coulomb parts must sum to their values for the total density
@pytest.mark.parametrize("part_method", ["mo", "ao", None])
@pytest.mark.parametrize("xc", ["hf", "pbe0", "camb3lyp"])
@pytest.mark.parametrize("restrict", [True, False])
def test_prop_tot_energy(restrict, xc, part_method):
    mf = _mf_h2o(restrict, xc)
    mol = mf.mol
    mo_coeff, mo_occ, weights = _decomp_input(mf, restrict)
    res = prop_tot(
        mol, mf, mo_coeff, mo_occ, None, "minao", "mulliken", "energy",
        part_method, False, np.zeros(3), weights if part_method == "mo" else None,
    )
    e_ref = mf.e_tot if part_method is not None else mf.e_tot - mol.energy_nuc()
    assert np.isclose(_total(res[CompKeys.tot]), e_ref, atol=1e-8)
    e_kin, e_nuc_att, e_coul = _e_ref_components(mf)
    nuc_att_keys = (CompKeys.nuc_att, CompKeys.nuc_att_loc, CompKeys.nuc_att_glob)
    assert np.isclose(_total(res[CompKeys.kin]), e_kin, atol=1e-8)
    assert np.isclose(
        sum(_total(res[k]) for k in nuc_att_keys if k in res), e_nuc_att, atol=1e-8
    )
    assert np.isclose(_total(res[CompKeys.coul]), e_coul, atol=1e-8)


# prop_tot, dipole moments
# atom-wise totals must sum to pyscf's dipole moment (gauge origin at zero, au)
@pytest.mark.parametrize("part_method", ["mo", "ao"])
@pytest.mark.parametrize("restrict", [True, False])
def test_prop_tot_dipole(restrict, part_method):
    mf = _mf_h2o(restrict, "hf")
    mol = mf.mol
    mo_coeff, mo_occ, weights = _decomp_input(mf, restrict)
    res = prop_tot(
        mol, mf, mo_coeff, mo_occ, None, "minao", "mulliken", "dipole",
        part_method, False, np.zeros(3), weights if part_method == "mo" else None,
    )
    assert res[CompKeys.tot].shape == (mol.natm, 3)
    dip_ref = mf.dip_moment(unit="au", verbose=0)
    assert np.allclose(res[CompKeys.tot].sum(axis=0), dip_ref, atol=1e-8)


# prop_tot, atomic reference energies
# atom-wise results must contain the reference energies and the total energy relative
# to them
def test_prop_tot_atom_ref():
    mf = _mf_h2o(True, "hf")
    mo_coeff, mo_occ, weights = _decomp_input(mf, True)
    atom_ref = np.array([-74.0, -0.5, -0.5])
    res = prop_tot(
        mf.mol, mf, mo_coeff, mo_occ, None, "minao", "mulliken", "energy",
        "mo", False, np.zeros(3), weights, atom_ref,
    )
    assert np.array_equal(res[CompKeys.atom_ref], atom_ref)
    assert np.allclose(res[CompKeys.tot_rel], res[CompKeys.tot] - atom_ref)


# aap_prop_tot
# all properties must have atom-and-atom-pair shape, the total must match pyscf's total
# energy, and the kinetic, nuclear attraction, coulomb and nuclear repulsion parts must
# sum to their values for the total density (kinetic: no atom-pair contributions)
@pytest.mark.parametrize("xc", ["hf", "pbe0", "camb3lyp"])
@pytest.mark.parametrize("restrict", [True, False])
def test_aap_prop_tot_sum(restrict, xc):
    mf = _mf_h2o(restrict, xc)
    mol = mf.mol
    natm = mol.natm
    mo_coeff, mo_occ, weights = _decomp_input(mf, restrict)
    res = aap_prop_tot(mol, mf, mo_coeff, mo_occ, None, "minao", "mulliken", weights)
    for v in res.values():
        assert v.shape == (natm + natm * (natm - 1) // 2,)
    assert np.isclose(np.sum(res[CompKeys.tot]), mf.e_tot, atol=1e-8)
    e_kin, e_nuc_att, e_coul = _e_ref_components(mf)
    assert np.isclose(np.sum(res[CompKeys.kin]), e_kin, atol=1e-8)
    assert np.allclose(res[CompKeys.kin][natm:], 0.0)
    assert np.isclose(np.sum(res[CompKeys.nuc_att]), e_nuc_att, atol=1e-8)
    assert np.isclose(np.sum(res[CompKeys.coul]), e_coul, atol=1e-8)
    assert np.isclose(np.sum(res[CompKeys.struct]), mol.energy_nuc(), atol=1e-10)


# a user-supplied 1-RDM identical to the mo 1-RDM (full dft exchange operator in
# _e_dft_x) must give the same results as rdm1=None (shortcut via the exchange matrix)
@pytest.mark.parametrize("xc", ["pbe0", "camb3lyp"])
@pytest.mark.parametrize("restrict", [True, False])
def test_aap_prop_tot_rdm1(restrict, xc):
    mf = _mf_h2o(restrict, xc)
    mol = mf.mol
    mo_coeff, mo_occ, weights = _decomp_input(mf, restrict)
    res = aap_prop_tot(mol, mf, mo_coeff, mo_occ, None, "minao", "mulliken", weights)
    rdm1 = np.array([make_rdm1(mo_coeff[s], mo_occ[s]) for s in range(2)])
    res_rdm1 = aap_prop_tot(
        mol, mf, mo_coeff, mo_occ, rdm1, "minao", "mulliken", weights
    )
    assert res_rdm1.keys() == res.keys()
    for k in res:
        assert np.allclose(res_rdm1[k], res[k], atol=1e-9)


# a2ap_redistribute
# hand-derived example: 3 atoms with pairs (0,1), (0,2), (1,2), of which (0,1) and
# (0,2) are bonds; atom 0 is split equally over its two bonds, atoms 1 and 2 belong to
# one bond each (rows normalized)
A2AP_WEIGHTS = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
BOND_MASK = np.array([True, True, False])


# atom-wise (a2b): atom energies must be redistributed over the bonds (lossless), keys
# outside PROP_KEYS discarded, and the energy relative to the isolated atoms added
def test_a2ap_redistribute_a2b():
    atom_ref = np.array([0.1, 0.2, 0.3])
    res = {CompKeys.tot: np.array([1.0, 2.0, 3.0]), CompKeys.atom_ref: atom_ref}
    bond_res = a2ap_redistribute(res, A2AP_WEIGHTS, BOND_MASK, atom_ref, aap=False)
    assert np.allclose(bond_res[CompKeys.tot], [2.5, 3.5])
    assert np.isclose(bond_res[CompKeys.tot].sum(), res[CompKeys.tot].sum())
    assert np.allclose(bond_res[CompKeys.tot_rel], [2.25, 3.15])
    assert np.array_equal(bond_res[CompKeys.bonds], [[0, 1], [0, 2]])
    assert CompKeys.atom_ref not in bond_res


# atom-and-atom-pair-wise (aap2b): the non-bonded pair (1,2) must be split equally over
# atoms 1 and 2, and the bonded pairs added to their bonds (lossless)
def test_a2ap_redistribute_aap2b():
    res = {CompKeys.tot: np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])}
    bond_res = a2ap_redistribute(res, A2AP_WEIGHTS, BOND_MASK, aap=True)
    assert np.allclose(bond_res[CompKeys.tot], [27.5, 38.5])
    assert np.isclose(bond_res[CompKeys.tot].sum(), res[CompKeys.tot].sum())
    assert np.array_equal(bond_res[CompKeys.bonds], [[0, 1], [0, 2]])


# without a bond mask, every atom pair must be treated as a bond
def test_a2ap_redistribute_no_bond_mask():
    res = {CompKeys.tot: np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])}
    bond_res = a2ap_redistribute(res, A2AP_WEIGHTS, aap=True)
    assert np.allclose(bond_res[CompKeys.tot], [12.5, 23.5, 30.0])
    assert np.array_equal(bond_res[CompKeys.bonds], [[0, 1], [0, 2], [1, 2]])


# arrays of the wrong shape must raise ValueError
@pytest.mark.parametrize("aap,size", [(False, 6), (True, 3)])
def test_a2ap_redistribute_wrong_shape(aap, size):
    res = {CompKeys.tot: np.zeros(size)}
    with pytest.raises(ValueError):
        a2ap_redistribute(res, A2AP_WEIGHTS, BOND_MASK, aap=aap)
