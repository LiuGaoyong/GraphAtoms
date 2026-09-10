import numpy as np
from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from graphatoms.utils.asetools import call_dimer, call_dimer_displace


def test_dimer_method() -> None:
    atoms = fcc100("Pt", size=(2, 2, 1), vacuum=10.0)
    add_adsorbate(atoms, "Pt", 1.611, "hollow")
    mask = [atom.tag > 0 for atom in atoms]
    atoms.set_constraint(FixAtoms(mask=mask))
    atoms.calc = EMT()
    atoms.get_potential_energy()

    disp = call_dimer_displace(atoms, calc=atoms.calc)
    print(disp)
    lst, _ = call_dimer(
        atoms.copy(),
        calc=atoms.calc,
        displacement=disp,
        # logfile="-",
    )
    for i, this_atoms in enumerate(lst[:3]):
        print("====")
        print(i)
        print(this_atoms.positions - atoms.positions)
        print(this_atoms.positions - atoms.positions - disp)
        print("---")

    np.testing.assert_array_almost_equal(
        lst[1].positions, atoms.positions + disp
    )
