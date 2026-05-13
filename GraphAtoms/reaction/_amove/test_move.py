from GraphAtoms.runner.mmc._moveFB import CanonicalState, FBDisp

if __name__ == "__main__":
    from ase.calculators.emt import EMT
    from ase.cluster import Octahedron
    from ase.visualize import view

    atoms = Octahedron("Cu", 10)
    atoms.positions -= atoms.get_center_of_mass()
    # atoms.numbers[atoms.positions[:, 2] > 0] = 47

    state, record = FBDisp.run(
        state=CanonicalState.from_ase(atoms),
        calculator=EMT(),
        stepsize_max=0.1,
    )
    print(record)
    view(
        [
            record.state_old.atoms,  # type: ignore
            record.state_new.atoms,  # type: ignore
        ]
    )
