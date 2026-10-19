"""Opt-in adapter to the archived driver; model and site packages stay unchanged."""
import importlib.util
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fast_graph import install

root = Path(os.environ.get("MLIP_PROJECT_ROOT", "/sdf/group/mli/MLIP_Li_S_Fl_Elec"))
spec = importlib.util.spec_from_file_location("original_gnnp_driver", root / "advancesoft-lammps/src/ML-GNNP/gnnp_driver.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def gnnp_initialize(*args, **kwargs):
    result = original.gnnp_initialize(*args, **kwargs)
    if original.myCalculator.potential.calc_hessian:
        raise RuntimeError("Inference-only optimization is incompatible with Hessians")
    install(inference_grad=True)
    return result


def gnnp_get_energy_forces_stress(*args, **kwargs):
    return original.gnnp_get_energy_forces_stress(*args, **kwargs)
