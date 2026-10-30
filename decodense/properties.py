#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
properties module
"""

import copy
import numpy as np
from pyscf import gto, scf, dft, df, lo, lib, solvent
from pyscf.pbc import dft as pbc_dft
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf
from typing import Union, Optional

from .pbctools import ewald_e_nuc
from .tools import dim, make_rdm1, orbsym, contract
from .decomp import CompKeys
from .prop_tools import _h_core, _vk_dft, _xc_energies, _orb_atom_occ

# block size in _mm_pot()
BLKSIZE = 200


def prop_tot(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: Union[
        scf.hf.SCF,
        dft.rks.KohnShamDFT,
        pbc_scf.hf.RHF,
        pbc_scf.uhf.UHF,
        pbc_dft.rks.RKS,
        pbc_dft.uks.UKS,
    ],
    mo_coeff: tuple[np.ndarray, np.ndarray],
    mo_occ: tuple[np.ndarray, np.ndarray],
    rdm1: Optional[np.ndarray],
    minao: str,
    pop_method: str,
    prop_type: str,
    part_method: Optional[str],
    ndo: bool,
    gauge_origin: np.ndarray,
    weights: Optional[list[np.ndarray]],
    atom_ref: Optional[np.ndarray] = None,
) -> dict[str, Union[np.ndarray, list[np.ndarray]]]:
    """
    this function returns atom-decomposed and orbital-decomposed mean-field properties
    """
    # declare nested kernel functions in global scope
    global prop_atom_mo
    global prop_atom_ao
    global prop_orb

    # restricted reference
    if mo_occ[0].size == mo_occ[1].size:
        restrict = np.allclose(mo_coeff[0], mo_coeff[1]) and np.allclose(
            mo_occ[0], mo_occ[1]
        )
    else:
        restrict = False

    # dft logical
    dft_calc = isinstance(mf, dft.rks.KohnShamDFT)

    # ndo assertion
    if dft_calc and ndo:
        raise NotImplementedError(
            "NDOs for KS-DFT do not yield a lossless decomposition"
        )

    # ao dipole integrals with specified gauge origin
    if prop_type == "dipole":
        with mol.with_common_origin(gauge_origin):
            ao_dip = mol.intor_symmetric("int1e_r", comp=3)
    else:
        ao_dip = None

    # compute total 1-RDMs (AO basis), identical for both spins if restricted
    rdm1_tot_a = make_rdm1(mo_coeff[0], mo_occ[0])
    rdm1_tot = np.array(
        [rdm1_tot_a, rdm1_tot_a if restrict else make_rdm1(mo_coeff[1], mo_occ[1])]
    )
    if rdm1 is None:
        rdm1 = rdm1_tot
    if rdm1.ndim == 2:
        rdm1 = np.array([rdm1, rdm1]) * 0.5
    del rdm1_tot_a

    # mol object projected into minao basis
    if pop_method == "iao":
        pmol = lo.iao.reference_mol(mol, minao=minao)
    else:
        pmol = mol

    # molecular dimensions
    alpha, beta = dim(mo_occ)
    natm = pmol.natm

    # nuclear repulsion property
    if prop_type == "energy":
        if isinstance(mol, pbc_gto.Cell):
            prop_nuc_rep = ewald_e_nuc(mol)
        else:
            prop_nuc_rep = _e_nuc(pmol)
    elif prop_type == "dipole":
        prop_nuc_rep = _dip_nuc(pmol, gauge_origin)

    # spin-summed total 1-RDM
    rdm1_tot_sum = np.sum(rdm1_tot, axis=0)

    # core hamiltonian, solvent contributions and fock potential
    # (only needed for energies)
    kin = nuc = e_nuc_att = vj = vk = None
    pot_solv = nuc_solv = vdW_solv = None
    if prop_type == "energy":
        # spin-summed 1-RDM and the 1-RDM(s) passed to the j/k builds
        rdm1_sum = np.sum(rdm1, axis=0)
        rdm1_jk = rdm1_sum if restrict else rdm1
        # core hamiltonian (incl. tr(V_K D) of the individual atomic potentials)
        kin, nuc, e_nuc_att = _h_core(mol, mf, rdm1_tot_sum)
        # get possible solvent contributions
        pot_solv, nuc_solv, vdW_solv = _solvent(mol, mf, rdm1_sum)
        # (unscaled) exact exchange is needed for hf and hybrid dft (same threshold as in _vk_dft())
        if dft_calc:
            with_k = abs(mf._numint.rsh_and_hybrid_coeff(mf.xc)[2]) > 1e-10
        else:
            with_k = True
        # fock potential (coulomb and exact exchange from a single call)
        if hasattr(mf, "vj"):
            vj = mf.vj
            if hasattr(mf, "vk"):
                vk = copy.copy(mf.vk)
            elif with_k:
                _, vk = mf.get_jk(mol=mol, dm=rdm1_jk, with_j=False, with_k=True)
        else:
            vj, vk = mf.get_jk(mol=mol, dm=rdm1_jk, with_k=with_k)

    # xc energy contributions: atom-wise ("mo"/"ao") or orbital-wise
    e_xc: Optional[Union[np.ndarray, list[np.ndarray]]] = None
    e_xc_nlc: Optional[Union[np.ndarray, list[np.ndarray]]] = None

    if dft_calc:
        if prop_type == "energy":
            # update exchange operator wrt range-separated parameter and exact exchange components
            vk = _vk_dft(mol, mf, mf.xc, rdm1_jk, vk)
            # xc energy (incl. nlc (vv10)) from individual atoms or orbitals
            e_xc, e_xc_nlc = _xc_energies(
                mol, mf, mo_coeff, mo_occ, rdm1, weights, part_method, natm
            )

    if prop_type == "energy":
        # spin-summed coulomb potential
        if not restrict:
            vj_sum = np.sum(vj, axis=0)
        # global nuclear attraction: 1/2 tr(V_K D) for all atoms K
        e_nuc_att_glob = 0.5 * e_nuc_att

    # ao-to-atom indices
    if part_method == "ao":
        ao_atm = np.array([lbl[0] for lbl in mol.ao_labels(fmt=None)])

    def energy_tail(res: dict[str, np.ndarray]) -> None:
        """
        this function adds the atom-wise energy terms that are computed outside the
        kernels (solvent nuclear and vdw terms, xc and nlc energies), and the sum of
        all electronic contributions
        """
        if nuc_solv is not None:
            res[CompKeys.solvent] += nuc_solv
        if vdW_solv is not None:
            res[CompKeys.solvent_vdw] = np.asarray(vdW_solv, dtype=np.float64)
        # additional xc energy contribution
        if e_xc is not None:
            res[CompKeys.xc] = e_xc
        if e_xc_nlc is not None:
            res[CompKeys.xc_nlc] = e_xc_nlc
        # sum up electronic contributions
        res[CompKeys.el] = sum(res.values())

    def prop_atom_mo() -> dict[str, np.ndarray]:
        """
        this function returns MO-based atom-wise energy/dipole contributions of all
        atoms, using tr(O d_K) = sum_m F[m,K] c_m^T O c_m (d_K is never formed)
        """
        # orbital-to-atom occupation factors and occupied mo coefficients
        F = _orb_atom_occ(mo_occ, weights, (alpha, beta), natm)
        c = [mo_coeff[i][:, spin_mo] for i, spin_mo in enumerate((alpha, beta))]

        def tr_atom(op: np.ndarray, spin: Optional[int] = None) -> np.ndarray:
            # tr(op d_K) for all atoms K, for one spin or summed over spins
            if spin is not None:
                return _mo_diag(op, c[spin]) @ F[spin]
            if restrict:
                return _mo_diag(op, c[0]) @ (F[0] + F[1])
            return _mo_diag(op, c[0]) @ F[0] + _mo_diag(op, c[1]) @ F[1]

        # init results
        res: dict[str, np.ndarray] = {}
        if prop_type == "energy":
            if restrict:
                res[CompKeys.coul] = 0.5 * tr_atom(vj)
                res[CompKeys.exch] = -0.25 * tr_atom(vk)
            else:
                res[CompKeys.coul] = 0.5 * tr_atom(vj_sum)
                res[CompKeys.exch] = -0.5 * (tr_atom(vk[0], 0) + tr_atom(vk[1], 1))
            res[CompKeys.kin] = tr_atom(kin)
            res[CompKeys.nuc_att_loc] = 0.5 * tr_atom(nuc)
            res[CompKeys.nuc_att_glob] = e_nuc_att_glob
            if pot_solv is not None:
                res[CompKeys.solvent] = tr_atom(pot_solv)
            energy_tail(res)
        elif prop_type == "dipole":
            res[CompKeys.el] = -tr_atom(ao_dip).T
        return res

    def prop_atom_ao() -> dict[str, np.ndarray]:
        """
        this function returns AO-based atom-wise (EDA - energy density analysis)
        energy/dipole contributions of all atoms, using
        sum_{mu in K} sum_nu O_munu D_munu
        """

        def tr_atom(op: np.ndarray, dm: np.ndarray) -> np.ndarray:
            # per-ao contributions sum_nu op_munu dm_munu, summed per atom
            o = np.einsum("...ij,ij->...i", op, dm)
            if o.ndim == 1:
                return np.bincount(ao_atm, weights=o, minlength=natm)
            return np.stack([np.bincount(ao_atm, weights=x, minlength=natm) for x in o])

        # init results
        res: dict[str, np.ndarray] = {}
        if prop_type == "energy":
            if restrict:
                res[CompKeys.coul] = 0.5 * tr_atom(vj, rdm1_tot_sum)
                res[CompKeys.exch] = -0.25 * tr_atom(vk, rdm1_tot_sum)
            else:
                res[CompKeys.coul] = 0.5 * tr_atom(vj_sum, rdm1_tot_sum)
                res[CompKeys.exch] = -0.5 * (
                    tr_atom(vk[0], rdm1_tot[0]) + tr_atom(vk[1], rdm1_tot[1])
                )
            res[CompKeys.kin] = tr_atom(kin, rdm1_tot_sum)
            res[CompKeys.nuc_att_loc] = 0.5 * tr_atom(nuc, rdm1_tot_sum)
            res[CompKeys.nuc_att_glob] = e_nuc_att_glob
            if pot_solv is not None:
                res[CompKeys.solvent] = tr_atom(pot_solv, rdm1_tot_sum)
            energy_tail(res)
        elif prop_type == "dipole":
            res[CompKeys.el] = -tr_atom(ao_dip, rdm1_tot_sum).T
        return res

    def prop_orb() -> dict[str, list[np.ndarray]]:
        """
        this function returns orbital-wise energy/dipole contributions of all occupied
        spin-orbitals, using tr(O d_m) = n_m c_m^T O c_m (d_m is never formed)
        """
        # occupied mo coefficients and occupations (ordered by position within the
        # spin channel, which is relevant for non-aufbau occupation)
        c = [mo_coeff[i][:, spin_mo] for i, spin_mo in enumerate((alpha, beta))]
        occ = [mo_occ[i][spin_mo] for i, spin_mo in enumerate((alpha, beta))]

        def tr_orb(op: np.ndarray, per_spin: bool = False) -> list[np.ndarray]:
            # tr(op D_m) for all occupied spin-orbitals m, per spin
            # (per_spin: op[i] is the operator of spin i)
            if restrict and not per_spin:
                o = occ[0] * _mo_diag(op, c[0])
                return [o, o.copy()]
            return [
                occ[i] * _mo_diag(op[i] if per_spin else op, c[i]) for i in range(2)
            ]

        # init results
        res: dict[str, list[np.ndarray]] = {}
        if prop_type == "energy":
            if restrict:
                res[CompKeys.coul] = [0.5 * x for x in tr_orb(vj)]
                res[CompKeys.exch] = [-0.25 * x for x in tr_orb(vk)]
            else:
                res[CompKeys.coul] = [0.5 * x for x in tr_orb(vj_sum)]
                res[CompKeys.exch] = [-0.5 * x for x in tr_orb(vk, per_spin=True)]
            res[CompKeys.kin] = tr_orb(kin)
            res[CompKeys.nuc_att] = tr_orb(nuc)
            if pot_solv is not None:
                res[CompKeys.solvent] = tr_orb(pot_solv)
            # additional xc energy contribution (e_xc is indexed by mo index)
            if e_xc is not None:
                res[CompKeys.xc] = [
                    e_xc[i][spin_mo] for i, spin_mo in enumerate((alpha, beta))
                ]
            if e_xc_nlc is not None:
                res[CompKeys.xc_nlc] = [
                    e_xc_nlc[i][spin_mo] for i, spin_mo in enumerate((alpha, beta))
                ]
            # sum up electronic contributions
            res[CompKeys.el] = [sum(v[i] for v in res.values()) for i in range(2)]
        elif prop_type == "dipole":
            res[CompKeys.el] = [-x.T for x in tr_orb(ao_dip)]
        return res

    # perform decomposition
    prop: dict[str, Union[np.ndarray, list[np.ndarray]]]
    if part_method in ["mo", "ao"]:
        # execute kernel (all atoms at once)
        prop = prop_atom_mo() if part_method == "mo" else prop_atom_ao()
        if ndo:
            prop[CompKeys.struct] = np.zeros_like(prop_nuc_rep)
        else:
            prop[CompKeys.struct] = prop_nuc_rep
        prop[CompKeys.tot] = prop[CompKeys.el] + prop[CompKeys.struct]

        if (prop_type == "energy") and (atom_ref is not None):
            prop[CompKeys.atom_ref] = atom_ref
            prop[CompKeys.tot_rel] = prop[CompKeys.tot] - atom_ref

        return {**prop}
    else:  # orbs
        # execute kernel (all occupied spin-orbitals at once)
        prop = prop_orb()
        if ndo:
            prop[CompKeys.struct] = np.zeros_like(prop_nuc_rep)
        else:
            prop[CompKeys.struct] = prop_nuc_rep
        prop[CompKeys.tot] = prop[CompKeys.el]
        return {
            **prop,
            CompKeys.mo_occ: [mo_occ[0][alpha], mo_occ[1][beta]],
            CompKeys.orbsym: orbsym(mol, (mo_coeff[0][:, alpha], mo_coeff[1][:, beta])),
        }


def _e_nuc(mol: gto.Mole) -> np.ndarray:
    """
    this function returns the nuclear repulsion energy
    """
    # coordinates and charges of nuclei
    charges = mol.atom_charges()
    # internuclear distances (with self-repulsion removed)
    dist = gto.inter_distance(mol)
    dist[np.diag_indices_from(dist)] = 1e200
    return contract("i,ij,j->i", charges, 1.0 / dist, charges) * 0.5


def _dip_nuc(mol: gto.Mole, gauge_origin: np.ndarray) -> np.ndarray:
    """
    this function returns the nuclear contribution to the molecular dipole moment
    """
    # coordinates and formal/actual charges of nuclei
    coords = mol.atom_coords()
    form_charges = mol.atom_charges()
    return contract("i,ix->ix", form_charges, coords - gauge_origin)


def _solvent(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT, pbc_scf.RHF],
    rdm1: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    # initialize
    pot_solv, nuc_solv, vdW_solv = None, None, None

    # point charges
    if hasattr(mf, "mm_mol"):
        mm_mol = getattr(mf, "mm_mol", None)
        pot_solv, nuc_solv = _point_charges(mol, mm_mol)
    # pcm
    elif hasattr(mf, "with_solvent"):
        if not isinstance(mf.with_solvent, solvent.pcm.PCM):
            raise NotImplementedError(
                "decodense only supports PCM-type solvent models (C-PCM, IEF-PCM, "
                f"SS(V)PE, COSMO, SMD), not {type(mf.with_solvent).__name__}"
            )
        pot_solv, nuc_solv = _pcm(mol, rdm1, mf.with_solvent)
    # OpenMM polarizable embedding
    elif hasattr(mf, "h1e_mmpol"):
        # static contribution to one-electron Hamiltonian
        pot_solv = getattr(mf, "h1e_mmpol").copy()

        # static nuclear contribution
        nuc_solv = np.array(
            [
                mf.V_mm_at_nucl[i] * mol.atom_charges()[i]
                for i in range(len(mol.atom_charges()))
            ]
        )

        # QM-MM vdW potential
        vdW_solv = mf.ommp_qm_helper.vdw_energy_by_atom(mf.ommp_obj)

        # polarization contributions
        if hasattr(mf, "v_mmpol_d"):
            # IPD contribution to the Fock Matrix
            pot_solv += 0.5 * getattr(mf, "v_mmpol_d")

            # polarization contribution from the potential of the IPDs at the nuclei
            nuc_solv += 0.5 * np.array(
                [
                    mf.V_pol_at_nucl[i] * mol.atom_charges()[i]
                    for i in range(len(mol.atom_charges()))
                ]
            )

    return pot_solv, nuc_solv, vdW_solv


def _point_charges(mol: gto.Mole, mm_mol: gto.Mole) -> tuple[np.ndarray, np.ndarray]:
    """
    this function returns the full mm potential and the nuclei interaction with the
    point charges (adapted from: qmmm/itrf.py:get_hcore() in PySCF)
    """
    # settings
    coords = mm_mol.atom_coords()
    charges = mm_mol.atom_charges()
    # integrals
    intor = "int3c2e_cart" if mol.cart else "int3c2e_sph"
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, intor)
    # compute interaction potential
    nao = mol.nao_nr()
    mm_pot = np.zeros(nao * (nao + 1) // 2, dtype=np.float64)
    for i0, i1 in lib.prange(0, charges.size, BLKSIZE):
        fakemol = gto.fakemol_for_charges(coords[i0:i1])
        j3c = df.incore.aux_e2(mol, fakemol, intor=intor, aosym="s2ij", cintopt=cintopt)
        mm_pot += np.einsum("xk,k->x", j3c, -charges[i0:i1])
    mm_pot = lib.unpack_tril(mm_pot)
    # nuclei interaction with point charges
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    nuc_solv = np.zeros(mol.natm)
    mm_atom_charges = mm_mol.atom_charges()
    mm_atom_coords = mm_mol.atom_coords()
    for j in range(mol.natm):
        q2, r2 = atom_charges[j], atom_coords[j]
        r = lib.norm(r2 - mm_atom_coords, axis=1)
        nuc_solv[j] = q2 * np.sum(mm_atom_charges / r)
    return mm_pot, nuc_solv


def _pcm(
    mol: gto.Mole, rdm1: np.ndarray, solvent_model: solvent.PCM
) -> tuple[np.ndarray, np.ndarray]:
    """
    this function returns the pcm potential matrix and the nuclei interaction with the
    solvent (adapted from: solvent/pcm.py:_get_vind() in PySCF)
    """
    surface = solvent_model.surface
    nao = mol.nao_nr()
    rdm1 = rdm1.reshape(-1, nao, nao)
    if rdm1.shape[0] == 2:
        rdm1 = (rdm1[0] + rdm1[1]).reshape(-1, nao, nao)
    # get the electronic part of the potential
    vmat_e = 0.5 * solvent_model._get_vind(np.sum(rdm1, axis=0))[1]
    # calculate the cavity surface charges
    K = solvent_model._intermediates["K"]
    R = solvent_model._intermediates["R"]
    v_grids_e = solvent_model._get_v(rdm1)
    v_grids_n = solvent_model.v_grids_n
    v_grids = v_grids_n - v_grids_e
    b = np.dot(R, v_grids.T)
    q = np.linalg.solve(K, b).T
    vK_1 = np.linalg.solve(K.T, v_grids.T)
    qt = np.dot(R.T, vK_1).T
    q_sym = (q + qt) / 2.0
    # get the nuclear part of the potential
    nuc_solv_pcm = np.zeros(mol.natm)
    for j in range(mol.natm):
        q2, r2 = mol.atom_charges()[j], mol.atom_coords()[j]
        r = lib.norm(r2 - surface["grid_coords"], axis=1)
        nuc_solv_pcm[j] = 0.5 * q2 * np.sum(q_sym / r)
    return vmat_e, nuc_solv_pcm


def _trace(
    op: np.ndarray, rdm1: np.ndarray, scaling: float = 1.0
) -> Union[float, np.ndarray]:
    """
    this function returns the trace between an operator and an rdm1
    """
    if op.ndim == 2:
        return contract("ij,ij", op, rdm1) * scaling
    else:
        return contract("xij,ij->x", op, rdm1) * scaling


def _mo_diag(op: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    this function returns c_m^T op c_m for all columns c_m of c
    (op: (nao, nao) -> shape (nmo,), op: (ncomp, nao, nao) -> shape (ncomp, nmo))
    """
    return np.sum((op @ c) * c, axis=-2)
