"""Local device-aligned ASE adapter; leaves installed MatGL untouched."""
import numpy as np
import torch
from ase.calculators.calculator import Calculator,all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from ase.units import GPa
from matgl.ext.ase import Atoms2Graph
class DevicePESCalculator(Calculator):
 implemented_properties=['energy','free_energy','forces','stress']
 def __init__(self,potential,**kwargs):
  super().__init__(**kwargs);self.potential=potential
  self.converter=Atoms2Graph(potential.model.element_types,potential.model.cutoff)
 def calculate(self,atoms=None,properties=None,system_changes=all_changes):
  super().calculate(atoms,properties,system_changes)
  device=next(self.potential.parameters()).device
  graph,lat,state=self.converter.get_graph(self.atoms)
  state=torch.tensor(np.asarray(state).tolist(),dtype=lat.dtype,device=device)
  with torch.enable_grad():e,f,s,*_=self.potential(g=graph.to(device),lat=lat.to(device),state_attr=state)
  # ASE's RATTLE projection iterates in-place at a double-precision tolerance.
  # Preserve every predicted value while giving that projection a float64 buffer.
  energy=float(e.detach().cpu());forces=f.detach().cpu().numpy().astype(np.float64)
  if not np.isfinite(energy) or not np.isfinite(forces).all():raise RuntimeError('Nonfinite calculator output')
  self.results=dict(energy=energy,free_energy=energy,forces=forces)
  if self.potential.calc_stresses:self.results['stress']=full_3x3_to_voigt_6_stress(s.detach().cpu().numpy().reshape(3,3))*GPa
