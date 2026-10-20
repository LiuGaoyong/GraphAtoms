import numpy as np
from ase.build import add_adsorbate, fcc100, fcc111
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from graphatoms.utils.asetools import (
    call_dimer,
    call_dimer_displace,
    call_neb,
)


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


def test_neb_method() -> None:
    atoms = fcc111("Cu", size=(2, 2, 2), vacuum=10.0)
    add_adsorbate(atoms, "Cu", 1.5, "fcc")
    atoms.set_constraint(FixAtoms(mask=[a.tag >= 2 for a in atoms]))
    atoms.calc = EMT()
    atoms.get_potential_energy()

    final = atoms.copy()
    final.positions[-1] += np.array([0.8, 0.8, 0.1])
    nimages = 3
    lst, converged = call_neb(
        atoms.copy(),
        calc=EMT(),
        final_atoms=final,
        nimages=nimages,
        max_steps=100,
        fmax=0.2,
    )
    # one band (i.e. `nimages` atoms objects) is recorded per step
    assert converged
    assert len(lst) % nimages == 0
    band = lst[-nimages:]
    assert len(band) == nimages
    # the endpoints are relaxed (thus close to the guess), and the TS image
    # is the highest one along the band.
    assert np.linalg.norm(band[0].positions - atoms.positions) < 0.5
    assert np.linalg.norm(band[-1].positions[-1] - band[0].positions[-1]) > 0.5
    energies = [i.get_potential_energy() for i in band]
    assert energies[1] > max(energies[0], energies[2])
