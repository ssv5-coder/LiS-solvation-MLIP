"""Private MatGL 1.2.3 three-body edge-ID repair, required for this pilot.

No learned weights, elemental references, cutoffs, or installed files are changed.
Exports must carry this module and load it before inference.
"""
import hashlib
import json
from pathlib import Path

ADAPTER_VERSION='threebody_original_edge_ids_v1'

def install(dtype=None):
 import numpy as np
 import torch
 import matgl
 import matgl.models._m3gnet as model_module
 from matgl.layers._three_body import ThreeBodyInteractions
 from matgl.utils.maths import scatter_sum
 if dtype is not None:
  matgl.float_th=dtype
  matgl.float_np=np.float64 if dtype==torch.float64 else np.float32
 if getattr(ThreeBodyInteractions,'_pilot_adapter',None)==ADAPTER_VERSION:return
 original_create=model_module.create_line_graph
 original_ensure=model_module.ensure_line_graph_compatibility
 def attach(graph,line,cutoff):
  ids=torch.nonzero(graph.edata['bond_dist']<=cutoff,as_tuple=False).reshape(-1)
  if line.num_nodes()>len(ids):raise RuntimeError('Inconsistent three-body graph')
  line.ndata['pilot_original_edge_id']=ids[:line.num_nodes()]
  return line
 def create(graph,threebody_cutoff,*args,**kwargs):
  return attach(graph,original_create(graph,threebody_cutoff,*args,**kwargs),threebody_cutoff)
 def ensure(graph,line,threebody_cutoff,*args,**kwargs):
  return attach(graph,original_ensure(graph,line,threebody_cutoff,*args,**kwargs),threebody_cutoff)
 def forward(self,graph,line_graph,three_basis,three_cutoff,node_feat,edge_feat):
  mapping=line_graph.ndata['pilot_original_edge_id']
  left,right=line_graph.edges();left=mapping[left].long();right=mapping[right].long()
  end_atoms=graph.edges()[1][right].long()
  basis=three_basis*self.update_network_atom(node_feat)[end_atoms]
  basis=basis*(three_cutoff[left]*three_cutoff[right])[:,None]
  new_bonds=scatter_sum(basis,segment_ids=left,num_segments=graph.num_edges(),dim=0)
  return edge_feat+self.update_network_bond(new_bonds)
 model_module.create_line_graph=create
 model_module.ensure_line_graph_compatibility=ensure
 ThreeBodyInteractions.forward=forward
 ThreeBodyInteractions._pilot_adapter=ADAPTER_VERSION

def load_potential(path,device='cuda',dtype=None):
 import torch
 from matgl.apps.pes import Potential
 dtype=torch.float32 if dtype is None else dtype
 install(dtype)
 return Potential.load(path).to(device=device,dtype=dtype)

def save_potential(potential,path):
 import copy
 path=Path(path);copy.deepcopy(potential).cpu().save(path)
 source=Path(__file__).read_bytes();(path/'runtime_adapter.py').write_bytes(source)
 (path/'inference_contract.json').write_text(json.dumps(dict(adapter=ADAPTER_VERSION,adapter_sha256=hashlib.sha256(source).hexdigest(),loader='from runtime_adapter import load_potential; potential = load_potential(checkpoint_path)',requires_adapter=True,production_approved=False,preserved=['Potential wrapper','elemental references','element mapping','graph cutoffs']),indent=2)+'\n')
