import warnings
from pathlib import Path

from ase.calculators.emt import EMT
from ase.cluster import Octahedron
from ase.optimize import LBFGS

THIS_DIR = Path(__file__).parent
MODEL_DIR = THIS_DIR.parent / "models"
model = MODEL_DIR / "PdAgCHO-S.nequip.pth"


atoms = Octahedron("Pd", 8)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    atoms.calc = EMT()
LBFGS(atoms).run()
atoms.write(THIS_DIR / "structure.xyz")
