import warnings
from pathlib import Path

from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.optimize import LBFGS

THIS_DIR = Path(__file__).parent
MODEL_DIR = THIS_DIR.parent / "models"
model = MODEL_DIR / "PdAgCHO-S.nequip.pth"


atoms = fcc111("Pd", [10, 10, 5], vacuum=30, orthogonal=True, periodic=True)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    atoms.calc = EMT()
LBFGS(atoms).run()
atoms.write(THIS_DIR / "structure.xyz")
