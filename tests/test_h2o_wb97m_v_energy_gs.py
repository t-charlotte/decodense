#!/usr/bin/env python
# -*- coding: utf-8 -*

from pathlib import Path
import unittest
import numpy as np
from pyscf import gto, dft

import decodense

# decimal tolerance
TOL = 9

# settings
# atom- or orbital-wise schemes
POP_METHOD = ("mulliken", "lowdin", "meta_lowdin", "becke", "iao")
PART = (
    ("orbitals", None),  # orbital-wise scheme
    ("atoms", "ao"),  # Nakai's AO-based atom-wise scheme (EDA)
    ("atoms", "mo"),  # Eriksen's MO-based atom-wise scheme
)
# bond-wise schemes
POP_METHOD_BOND = ("mulliken", "iao")
PART_BOND = (
    ("bonds", "a2b"),  # atoms-to-bonds
    ("bonds", "aap2b"),  # atoms-and-atom-pairs-to-bonds
)

# geometry directory
GEOM_DIR = Path(__file__).parent / "geom"

# init molecule
mol = gto.M(
    verbose=0,
    output="/dev/null",
    basis="pcseg1",
    symmetry=True,
    atom=str(GEOM_DIR / "h2o.xyz"),
)

# mf calc
mf = dft.RKS(mol)
mf.xc = "wb97m_v"
mf.nlc = "vv10"
mf.nlcgrids.atom_grid = (50, 194)
mf.nlcgrids.prune = dft.gen_grid.sg1_prune
mf.kernel()

# occupied orbitals
occ_mo = np.where(mf.mo_occ == 2.0)[0]

# mo coefficients
mo_coeff = 2 * (mf.mo_coeff[:, occ_mo],)


def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


class KnownValues(unittest.TestCase):
    def test(self):
        mf_e_tot = mf.e_tot

        def sub_test(pop_list, part_list):
            for pop_method in pop_list:
                for part_pair in part_list:
                    with self.subTest(pop_method=pop_method, part_pair=part_pair):
                        part, part_method = part_pair
                        decomp = decodense.DecompCls(
                            pop_method=pop_method, part=part, part_method=part_method
                        )
                        res = decodense.main(mol, decomp, mf, mo_coeff)
                        if part == "orbitals":
                            e_tot = np.sum(res.tot[0]) + np.sum(res.tot[1])
                            ref = mf_e_tot - mol.energy_nuc()
                        else:
                            e_tot = np.sum(res.tot)
                            ref = mf_e_tot
                        self.assertAlmostEqual(ref, e_tot, TOL)


        # atom- or orbital-wise schemes
        sub_test(POP_METHOD, PART)
        # bond-wise schemes
        sub_test(POP_METHOD_BOND, PART_BOND)



if __name__ == "__main__":
    print("test: test_h2o_wb97m_v_energy_gs")
    unittest.main()
