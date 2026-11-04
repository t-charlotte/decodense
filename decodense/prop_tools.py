#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
prop_tools module
"""

import copy
import numpy as np
from pyscf import gto, scf, dft, df, lib, solvent
from pyscf.dft import numint
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf
from pyscf.pbc.dft import numint as pbc_numint
from typing import Union, Optional, Iterable, Iterator

from .pbctools import get_nuc_pbc
from .tools import dim, contract, logger, make_rdm1

# atom-wise (array) or orbital-wise (list of arrays per spin) xc energies
XCPart = Union[np.ndarray, list[np.ndarray]]

# block size in _point_charges()
BLKSIZE = 200

# max. number of atom-RDM1s per batched get_jk call in _e_jk_atoms()
# (lower it to save memory)
JK_BLKSIZE = 16


def _h_core(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT, pbc_scf.RHF],
    rdm1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    this function returns the kinetic and total nuclear attraction operators,
    and the nuclear attraction energies of the individual atomic potentials
    """
    if not isinstance(mol, pbc_gto.Cell) and mol.has_ecp():
        raise NotImplementedError("Decodense does not support ECP")
    if isinstance(mol, pbc_gto.Cell) and isinstance(
        mf, (pbc_scf.hf.RHF, pbc_scf.uhf.UHF)
    ):
        # kinetic integrals
        kin = mol.pbc_intor("int1e_kin")
        mydf = mf.with_df
        # individual atomic potentials
        sub_nuc = get_nuc_pbc(mol, mydf)
        # total nuclear potential
        nuc = np.sum(sub_nuc, axis=0)
        e_nuc = sub_nuc.reshape(sub_nuc.shape[0], -1) @ rdm1.ravel()
    else:
        # kinetic integrals
        kin = mol.intor_symmetric("int1e_kin")
        # total nuclear potential and nuclear attraction energies
        nuc = np.zeros_like(kin)
        e_nuc = np.zeros(mol.natm, dtype=np.float64)
        for k, sub_nuc in enumerate(_v_nuc_atoms(mol)):
            nuc += sub_nuc
            e_nuc[k] = np.einsum("ij,ij", sub_nuc, rdm1)
    return kin, nuc, e_nuc


def _e_nuc_pairs(mol: gto.Mole) -> np.ndarray:
    """
    this function returns the nuclear repulsion energies of all atom pairs
    """
    # coordinates and charges of nuclei
    charges = mol.atom_charges()
    # internuclear distances (with self-repulsion removed)
    dist = gto.inter_distance(mol)
    dist[np.diag_indices_from(dist)] = 1e200
    return contract("i,ij,j->ij", charges, 1.0 / dist, charges)


def _dip_nuc(mol: gto.Mole, gauge_origin: np.ndarray) -> np.ndarray:
    """
    this function returns the nuclear contribution to the molecular dipole moment
    """
    # coordinates and formal/actual charges of nuclei
    coords = mol.atom_coords()
    form_charges = mol.atom_charges()
    return contract("i,ix->ix", form_charges, coords - gauge_origin)


def _v_nuc_atoms(mol: gto.Mole) -> Iterator[np.ndarray]:
    """
    this function yields the individual atomic potentials of a molecule,
    one atom at a time
    """
    # coordinates and charges of nuclei
    coords = mol.atom_coords()
    charges = mol.atom_charges()
    for k in range(mol.natm):
        with mol.with_rinv_origin(coords[k]):
            sub_nuc = -1.0 * charges[k] * mol.intor_symmetric("int1e_rinv")
        # yielded outside the context manager, so the rinv origin is restored
        yield sub_nuc


def _make_rho_interm1(
    ao_value: np.ndarray, rdm1: np.ndarray, xc_type: str
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    this function returns the rho intermediates (c0, c1) needed in _make_rho()
    (adpated from: dft/numint.py:eval_rho() in PySCF)
    """
    # compute rho intermediate based on xctype
    xctype = xc_type.upper()
    if xctype == "LDA" or xctype == "HF":
        c0 = contract("ik,kj->ij", ao_value, rdm1)
        c1 = None
    elif xctype in ("GGA", "NLC"):
        c0 = contract("ik,kj->ij", ao_value[0], rdm1)
        c1 = None
    else:  # meta-GGA
        ngrids, nao = ao_value[0].shape
        c0 = contract("ik,kj->ij", ao_value[0], rdm1)
        c1 = np.empty((3, ngrids, nao), dtype=np.float64)
        for i in range(1, 4):
            c1[i - 1] = contract("ik,kj->ij", ao_value[i], rdm1)
    return c0, c1


def _make_rho_interm2(
    c0: np.ndarray, c1: Optional[np.ndarray], ao_value: np.ndarray, xc_type: str
) -> np.ndarray:
    """
    this function returns rho from intermediates (c0, c1)
    (adpated from: dft/numint.py:eval_rho() in PySCF)
    """
    # determine dimensions based on xctype
    xctype = xc_type.upper()
    if xctype == "LDA" or xctype == "HF":
        ngrids = ao_value.shape[0]
    else:
        ngrids = ao_value[0].shape[0]
    # compute rho intermediate based on xctype
    if xctype == "LDA" or xctype == "HF":
        rho = contract("pi,pi->p", ao_value, c0)
    elif xctype in ("GGA", "NLC"):
        rho = np.empty((4, ngrids), dtype=np.float64)
        rho[0] = contract("pi,pi->p", c0, ao_value[0])
        for i in range(1, 4):
            rho[i] = contract("pi,pi->p", c0, ao_value[i]) * 2.0
    else:  # meta-GGA
        assert c1 is not None
        rho = np.empty((6, ngrids), dtype=np.float64)
        rho[0] = contract("pi,pi->p", ao_value[0], c0)
        rho[5] = 0.0
        for i in range(1, 4):
            rho[i] = contract("pi,pi->p", c0, ao_value[i]) * 2.0
            rho[5] += contract("pi,pi->p", c1[i - 1], ao_value[i])
        XX, YY, ZZ = 4, 7, 9
        ao_value_2 = ao_value[XX] + ao_value[YY] + ao_value[ZZ]
        rho[4] = contract("pi,pi->p", c0, ao_value_2)
        rho[4] += rho[5]
        rho[4] *= 2.0
        rho[5] *= 0.5
    return rho


def _make_rho(
    ao_value: np.ndarray, rdm1: np.ndarray, xc_type: str
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    this function returns important dft intermediates, e.g., energy density, grid
    weights, etc.
    """
    # rho corresponding to given 1-RDM
    if rdm1.ndim == 2:
        c0, c1 = _make_rho_interm1(ao_value, rdm1, xc_type)
        rho = _make_rho_interm2(c0, c1, ao_value, xc_type)
    else:
        if np.allclose(rdm1[0], rdm1[1]):
            c0, c1 = _make_rho_interm1(ao_value, rdm1[0] * 2.0, xc_type)
            rho = _make_rho_interm2(c0, c1, ao_value, xc_type)
        else:
            c0_a, c1_a = _make_rho_interm1(ao_value, rdm1[0], xc_type)
            c0_b, c1_b = _make_rho_interm1(ao_value, rdm1[1], xc_type)
            rho = np.stack(
                (
                    _make_rho_interm2(c0_a, c1_a, ao_value, xc_type),
                    _make_rho_interm2(c0_b, c1_b, ao_value, xc_type),
                )
            )
            c0 = c0_a + c0_b
            if c1_a is not None and c1_b is not None:
                c1 = c1_a + c1_b
            else:
                c1 = None
    return c0, c1, rho


def _vk_dft(
    mol: gto.Mole,
    mf: dft.rks.KohnShamDFT,
    xc_func: str,
    rdm1: np.ndarray,
    vk: Optional[np.ndarray],
) -> np.ndarray:
    """
    this function returns the appropriate dft exchange operator
    (vk: unscaled exchange operator if already available, otherwise None)
    """
    # range-separated and exact exchange parameters
    ks_omega, ks_alpha, ks_hyb = mf._numint.rsh_and_hybrid_coeff(xc_func)
    # if hybrid func: compute vk
    if abs(ks_hyb) > 1e-10:
        if vk is None:
            vk = mf.get_k(mol, rdm1)
        # scale amount of exact exchange
        vk *= ks_hyb
    else:
        vk = np.zeros_like(rdm1)  # rdm1 has the same layout as vk
    # range separated coulomb operator
    if abs(ks_omega) > 1e-10:
        vk_lr = mf.get_k(mol, rdm1, omega=ks_omega)
        vk_lr *= ks_alpha - ks_hyb
        vk += vk_lr  # vk_lr has the same (restricted or unrestricted) layout as vk
    return vk


def _ao_val(mol: gto.Mole, grids_coords: np.ndarray, ao_deriv: int) -> np.ndarray:
    """
    this function returns ao function values on the given grid
    """
    if not isinstance(mol, pbc_gto.Cell):
        return numint.eval_ao(mol, grids_coords, deriv=ao_deriv)
    else:
        return pbc_numint.eval_ao(mol, grids_coords, deriv=ao_deriv)


def _orb_atom_occ(
    mo_occ: tuple[np.ndarray, np.ndarray],
    weights: list[np.ndarray],
    spin_mos: tuple[np.ndarray, np.ndarray],
    natm: int,
) -> list[np.ndarray]:
    """
    this function returns the orbital-to-atom occupation factors
    """
    F = []
    for i, spin_mo in enumerate(spin_mos):
        if len(spin_mo) == 0:
            F.append(np.zeros((0, natm), dtype=np.float64))
            continue
        w = np.asarray(weights[i], dtype=np.float64)
        F.append(mo_occ[i][spin_mo][:, None] * w / np.sum(w, axis=1, keepdims=True))
    return F


def _xc_blocks(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: dft.rks.KohnShamDFT,
    rdm1: np.ndarray,
    xc_type: str,
    ao_deriv: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    this function yields the following quantities block-wise on the dft grid:
    - ao values (without derivatives)
    - weighted xc energy densities
    - w(r) * eps_xc(r)
    - the rho intermediate c0 = ao . D
    """
    if xc_type == "HF":
        return
    if mf.grids.coords is None or mf.grids.weights is None:
        logger.warning(
            f'\nCalculating grids for mf.xc="{mf.xc}" with mf.grids.level={mf.grids.level}\n'
            "Should you wish to use other values, please set mf.xc and mf.grids.level in your input.\n"
        )
        mf.grids.build(with_non0tab=True)
    if isinstance(mol, pbc_gto.Cell):
        # pbc block_loop has a different interface -> single block
        blocks: Iterable = [(_ao_val(mol, mf.grids.coords, ao_deriv), mf.grids.weights)]
    else:
        max_memory = max(2000, mf.max_memory - lib.current_memory()[0])
        blocks = (
            (ao, weight)
            for ao, _, weight, _ in mf._numint.block_loop(
                mol, mf.grids, mol.nao_nr(), ao_deriv, max_memory=max_memory
            )
        )
    xc_spin = 0 if np.allclose(rdm1[0], rdm1[1]) else 1
    for ao, weight in blocks:
        c0, c1, rho = _make_rho(ao, rdm1, xc_type)
        # c1 is only needed for rho
        del c1
        eps_xc = dft.libxc.eval_xc(mf.xc, rho, spin=xc_spin, deriv=0)[0]
        yield (ao if ao.ndim == 2 else ao[0]), weight * eps_xc, c0
        # free c0 before the next block is evaluated
        del c0


def _nlc_blocks(
    mol: gto.Mole, mf: dft.rks.KohnShamDFT, rdm1: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    this function returns the following quantities on the nlc grid:
    - ao values (without derivatives)
    - weighted nlc (vv10) energy densities
    - w(r) * eps_nlc(r)
    - the rho intermediate c0 = ao . C
    (adapted from: rks.get_veff and numint.nr_nlc_vxc in PySCF)
    """
    if mf.nlcgrids.coords is None or mf.nlcgrids.weights is None:
        logger.warning(
            f'\nCalculating nlcgrids for mf.xc="{mf.xc}" with mf.nlcgrids.level={mf.nlcgrids.level}\n'
            "Should you wish to use other values, please set mf.xc and mf.nlcgrids.level in your input.\n"
        )
        mf.nlcgrids.build(with_non0tab=True)
    nlc_xc = mf.xc if mf._numint.libxc.is_nlc(mf.xc) else mf.nlc
    coords, weights = mf.nlcgrids.coords, mf.nlcgrids.weights
    ao_value_nlc = _ao_val(mol, coords, 1)
    c0_vv10, _, rho_vv10 = _make_rho(ao_value_nlc, np.sum(rdm1, axis=0), "GGA")
    eps_xc_nlc = sum(
        fac * numint._vv10nlc(rho_vv10, coords, rho_vv10, weights, coords, nlc_pars)[0]
        for nlc_pars, fac in mf._numint.nlc_coeff(nlc_xc)
    )
    return [(ao_value_nlc[0], weights * eps_xc_nlc, c0_vv10)]


def _xc_partition(
    blocks: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    mol: Union[gto.Mole, pbc_gto.Cell],
    mo_coeff: tuple[np.ndarray, np.ndarray],
    mo_occ: tuple[np.ndarray, np.ndarray],
    weights: Optional[list[np.ndarray]],
    part_method: Optional[str],
    natm: int,
) -> XCPart:
    """
    this function takes block-wise quantities and returns atom- or orbital-wise xc energies
    """
    spin_mos = dim(mo_occ)
    if part_method == "ao":
        # ao contributions: e_mu = sum_r w(r) eps(r) chi_mu(r) sum_nu D_munu chi_nu(r)
        # (sum_nu D_munu chi_nu(r) = c0, reused from the evaluation of rho)
        e_ao = np.zeros(mol.nao_nr(), dtype=np.float64)
        for ao0, weps, c0 in blocks:
            e_ao += contract("p,pi,pi->i", weps, ao0, c0)
            # free c0 before the next block is evaluated
            del c0
        ao_atm = np.array([lbl[0] for lbl in mol.ao_labels(fmt=None)])
        return np.bincount(ao_atm, weights=e_ao, minlength=natm)
    # orbital contributions: e_m = sum_r w(r) eps(r) |phi_m(r)|^2
    e_orb = [np.zeros(spin_mo.size, dtype=np.float64) for spin_mo in spin_mos]
    for ao0, weps, c0 in blocks:
        for i, spin_mo in enumerate(spin_mos):
            e_orb[i] += weps @ (ao0 @ mo_coeff[i][:, spin_mo]) ** 2
        # free c0 before the next block is evaluated
        del c0
    if part_method == "mo":
        # atom contributions via orbital-to-atom occupation factors
        assert weights is not None
        F = _orb_atom_occ(mo_occ, weights, spin_mos, natm)
        return sum(e_orb[i] @ F[i] for i in range(2))
    # occupation-weighted orbital contributions (indexed by mo index)
    e_mo = [np.zeros(mo_occ[i].size, dtype=np.float64) for i in range(2)]
    for i, spin_mo in enumerate(spin_mos):
        e_mo[i][spin_mo] = mo_occ[i][spin_mo] * e_orb[i]
    return e_mo


def _xc_energies(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: dft.rks.KohnShamDFT,
    mo_coeff: tuple[np.ndarray, np.ndarray],
    mo_occ: tuple[np.ndarray, np.ndarray],
    rdm1: np.ndarray,
    weights: Optional[list[np.ndarray]],
    part_method: Optional[str],
    natm: int,
) -> tuple[XCPart, Optional[XCPart]]:
    """
    this function returns the "pure" xc and nlc (vv10) energies (None if no nlc) from
    individual atoms ("mo"/"ao") or orbitals (otherwise)
    """
    # type of xc functional and level of ao derivatives needed
    xc_type = dft.libxc.xc_type(mf.xc)
    if xc_type in ("LDA", "HF"):
        ao_deriv = 0
    elif xc_type in ("GGA", "NLC"):
        ao_deriv = 1
    elif xc_type == "MGGA":
        ao_deriv = 2
    else:
        raise NotImplementedError(f"xc functional type {xc_type} not supported")
    e_xc = _xc_partition(
        _xc_blocks(mol, mf, rdm1, xc_type, ao_deriv),
        mol,
        mo_coeff,
        mo_occ,
        weights,
        part_method,
        natm,
    )
    e_xc_nlc = None
    if not isinstance(mol, pbc_gto.Cell) and mf.do_nlc():
        e_xc_nlc = _xc_partition(
            _nlc_blocks(mol, mf, rdm1),
            mol,
            mo_coeff,
            mo_occ,
            weights,
            part_method,
            natm,
        )
    return e_xc, e_xc_nlc


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
    point charges
    (adapted from: qmmm/itrf.py:get_hcore() in PySCF)
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
    solvent
    (adapted from: solvent/pcm.py:_get_vind() in PySCF)
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


def _mo_diag(op: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    this function returns c_m^T op c_m for all columns c_m of c
    (op: (nao, nao) -> shape (nmo,), op: (ncomp, nao, nao) -> shape (ncomp, nmo))
    """
    return np.sum((op @ c) * c, axis=-2)


def _tr_atom(
    op: np.ndarray,
    c: list[np.ndarray],
    F: list[np.ndarray],
    restrict: bool,
    spin: Optional[int] = None,
) -> np.ndarray:
    """
    this function returns tr(op d_K) for all atoms K, for one spin or summed over
    spins, using tr(op d_K) = sum_m F[m,K] c_m^T op c_m (d_K is never formed)
    (op: (nao, nao) -> shape (natm,), op: (ncomp, nao, nao) -> shape (ncomp, natm))
    """
    if spin is not None:
        return _mo_diag(op, c[spin]) @ F[spin]
    if restrict:
        return _mo_diag(op, c[0]) @ (F[0] + F[1])
    return _mo_diag(op, c[0]) @ F[0] + _mo_diag(op, c[1]) @ F[1]


def _e_jk_atoms(
    mol: gto.Mole,
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT],
    c: list[np.ndarray],
    F: list[np.ndarray],
    restrict: bool,
    blksize: int = JK_BLKSIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    this function returns the atom-atom Coulomb and exchange energy matrices
    """
    natm, nao = F[0].shape[1], c[0].shape[0]
    nspin = 1 if restrict else 2
    coul = np.zeros((natm, natm), dtype=np.float64)
    exch = np.zeros((natm, natm), dtype=np.float64)
    for k0, k1 in lib.prange(0, natm, blksize):
        n = k1 - k0
        # atom-specific 1-RDMs of this batch (alpha only if restricted)
        dm = np.array(
            [make_rdm1(c[s], F[s][:, k]) for s in range(nspin) for k in range(k0, k1)]
        )
        vj, vk = mf.get_jk(mol=mol, dm=dm, with_j=True, with_k=True)
        del dm
        if restrict:  # alpha and beta 1-RDMs are identical -> one J/K build per atom
            # 1/2 tr(2 vj, d_L)
            coul[k0:k1] = _tr_atom(vj, c, F, restrict)
            # -1/2 * 2 spins * tr(vk, d_L,alpha)
            exch[k0:k1] = -_tr_atom(vk, c, F, restrict, spin=0)
        else:
            vj, vk = vj.reshape(2, n, nao, nao), vk.reshape(2, n, nao, nao)
            coul[k0:k1] = 0.5 * _tr_atom(vj[0] + vj[1], c, F, restrict)
            exch[k0:k1] = -0.5 * sum(
                _tr_atom(vk[s], c, F, restrict, spin=s) for s in range(2)
            )
    return coul, exch


def _e_dft_x(
    mol: gto.Mole,
    mf: dft.rks.KohnShamDFT,
    rdm1: np.ndarray,
    c: list[np.ndarray],
    F: list[np.ndarray],
    exch: np.ndarray,
    restrict: bool,
    rdm1_from_mos: bool,
) -> np.ndarray:
    """
    this function returns the atom contributions to the exact exchange included in
    the DFT XC energy
    two possible routes:
    1) rdm1_from_mos=True: the full-range part is taken from the atom-atom exchange matrix that was calculated in _e_jk_atoms;
    2) rdm1_from_mos=False: the full-range part is built here from the provided rdm1, or taken from mf.vk if present.
    the long-range part of range-separated functionals is always built here.
    """
    rdm1_jk = np.sum(rdm1, axis=0) if restrict else rdm1

    def tr_vk(vk: np.ndarray) -> np.ndarray:
        """
        this function returns -1/4 tr(vk d_K) (restricted) or
        -1/2 sum_s tr(vk_s d_K,s) (unrestricted) for all atoms K
        """
        if restrict:
            return -0.25 * _tr_atom(vk, c, F, restrict)
        return -0.5 * sum(_tr_atom(vk[s], c, F, restrict, spin=s) for s in range(2))

    # user-supplied 1-RDM: the d_L do not sum to D -> full K build
    if not rdm1_from_mos:
        return tr_vk(
            _vk_dft(
                mol,
                mf,
                mf.xc,
                rdm1_jk,
                copy.copy(mf.vk) if hasattr(mf, "vk") else None,
            )
        )
    ks_omega, _, ks_hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc)
    e_dft_x = ks_hyb * np.sum(exch, axis=1)
    if abs(ks_omega) > 1e-10:
        # long-range part only (a zero full-range operator is passed to _vk_dft)
        e_dft_x += tr_vk(_vk_dft(mol, mf, mf.xc, rdm1_jk, np.zeros_like(rdm1_jk)))
    return e_dft_x
