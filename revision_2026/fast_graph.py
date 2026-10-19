"""Optional inference-only optimizations; never edit the installed MatGL package."""
import numpy as np


def ordered_triplets(degrees):
    """Same ordered distinct-edge pairs as the original per-atom meshgrid loop."""
    degrees = np.asarray(degrees, dtype=np.int64)
    per_atom = degrees * (degrees - 1)
    starts = np.cumsum(degrees) - degrees
    local = np.arange(per_atom.sum()) - np.repeat(np.cumsum(per_atom)-per_atom, per_atom)
    widths = np.repeat(degrees-1, per_atom)
    first = local // widths
    second = local % widths
    second += second >= first
    offsets = np.repeat(starts, per_atom)
    return first+offsets, second+offsets


def install(inference_grad=False):
    import dgl
    import torch
    import matgl
    import matgl.graph.compute as compute

    def tensor(array, dtype, device):
        # This legacy Torch build cannot use the installed NumPy C API. The
        # buffer protocol avoids its slow list-conversion compatibility path.
        numpy_type = {torch.int32: np.int32, torch.int64: np.int64,
                      torch.float32: np.float32, torch.float64: np.float64}[dtype]
        values = np.array(array, dtype=numpy_type, order="C", copy=True)
        if values.size == 0:
            return torch.empty(values.shape, dtype=dtype, device=device)
        return torch.frombuffer(memoryview(values), dtype=dtype).reshape(values.shape).to(device=device)

    def vectorized_3body(graph):
        src = graph.edges()[0].detach().cpu().numpy()
        degrees = np.bincount(src, minlength=graph.num_nodes())
        first, second = ordered_triplets(degrees)
        count = int(max(first.max(), second.max())+1) if len(first) else 0
        left = tensor(first, matgl.int_th, graph.device)
        right = tensor(second, matgl.int_th, graph.device)
        line = dgl.graph((left, right), num_nodes=count, device=graph.device)
        for key in ["bond_dist", "bond_vec", "pbc_offset"]:
            line.ndata[key] = graph.edata[key][:count]
        line.ndata["n_triple_ij"] = tensor(np.repeat(degrees-1, degrees)[:count], matgl.int_th, graph.device)
        return line

    compute._compute_3body = vectorized_3body
    if inference_grad:
        from matgl.graph.converters import GraphConverter

        def convert(self, structure, src_id, dst_id, images, lattice_matrix,
                    element_types, frac_coords, is_atoms=False):
            device = torch.empty(0).device
            graph = dgl.graph((tensor(src_id, matgl.int_th, device),
                               tensor(dst_id, matgl.int_th, device)),
                              num_nodes=len(structure), device=device)
            graph.edata["pbc_offset"] = tensor(images, matgl.float_th, device)
            symbols = (structure.get_chemical_symbols() if is_atoms else
                       [site.specie.symbol for site in structure])
            mapping = {element: index for index, element in enumerate(element_types)}
            graph.ndata["node_type"] = tensor([mapping[s] for s in symbols], matgl.int_th, device)
            graph.ndata["frac_coords"] = tensor(frac_coords, matgl.float_th, device)
            lattice = tensor(lattice_matrix, matgl.float_th, device)
            return graph, lattice, np.zeros(2, dtype=matgl.float_np)

        GraphConverter.get_graph_from_processed_structure = convert
        import matgl.apps.pes as pes
        original = pes.grad

        def first_derivative_only(*args, **kwargs):
            kwargs["create_graph"] = False
            kwargs["retain_graph"] = False
            return original(*args, **kwargs)

        pes.grad = first_derivative_only
