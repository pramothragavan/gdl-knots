from collections import defaultdict, deque
import torch
from utils import build_designA_graph, to_strand_split_graph, add_reverse_edges
from preprocess import conway_unfold
import torch.nn.functional as F
import numpy as np

def flip_crossing_pd(pd_list, crossing_idx):
    """Flip crossing i by rotating its PD tuple by 1 position.

    [a,b,c,d] -> [b,c,d,a] swaps which strand is over vs under.
    This is a crossing change (L+ <-> L-).
    """
    new_pd = [list(q) for q in pd_list]
    a, b, c, d = new_pd[crossing_idx]
    new_pd[crossing_idx] = [b, c, d, a]
    return new_pd

def get_crossing_signs(pd_list):
    """Extract crossing signs from a PD list using the existing pipeline."""
    g = build_designA_graph(pd_list)
    return g.x[:, 0].numpy().copy()

def build_graph_for_inference(pd_list):
    """Build a single-graph batch ready for model inference."""
    g = build_designA_graph(pd_list)
    g = to_strand_split_graph(g, add_coupling=True)
    g = add_reverse_edges(g, num_relations=5)
    g.batch = torch.zeros(g.num_nodes, dtype=torch.long)
    return g

def decode_values(value_raw, value_mode, codec=None, decode_mode='expected'):
    if value_mode == 'regression':
        return value_raw
    class_values = torch.tensor(codec['class_values'], dtype=value_raw.dtype, device=value_raw.device)
    probs = F.softmax(value_raw, dim=-1)
    if decode_mode == 'argmax':
        cls = probs.argmax(dim=-1)
        return class_values[cls]
    return (probs * class_values.view(1, 1, -1)).sum(dim=-1)

@torch.no_grad()
def model_predict(model, g, device, value_mode='classification', codec=None):
    """Run hurdle model on a single-graph batch, return soft-gated prediction."""
    model.eval()
    g = g.to(device)
    logits, value_raw = model(g)
    p = torch.sigmoid(logits)
    v = decode_values(value_raw, value_mode=value_mode,
                      codec=codec, decode_mode='expected')
    return (p * v).squeeze(0).cpu()  # [L_folded]

@torch.no_grad()
def skein_explain(model, pd_list, device, kind_data,
                  value_mode='classification', codec=None,
                  y_true_folded=None):
    """Compute per-crossing attribution by flipping each crossing."""
    model.eval()
    n = len(pd_list)
    fold_info = kind_data['fold_info']
    min_exp = kind_data['min_exp']
    L_full = kind_data['L_full']

    # Original prediction
    g_orig = build_graph_for_inference(pd_list)
    pred_orig = model_predict(model, g_orig, device, value_mode, codec)
    signs = get_crossing_signs(pd_list)

    delta_preds = torch.zeros(n, pred_orig.shape[0])
    attributions = torch.zeros(n)
    flip_ok = torch.ones(n, dtype=torch.bool)

    for i in range(n):
        flipped_pd = flip_crossing_pd(pd_list, i)
        try:
            g_flip = build_graph_for_inference(flipped_pd)
            pred_flip = model_predict(model, g_flip, device, value_mode, codec)
            delta = pred_orig - pred_flip
            delta_preds[i] = delta
            attributions[i] = delta.abs().sum()
        except Exception as e:
            print(f"  Crossing {i} flip failed: {e}")
            attributions[i] = float('nan')
            flip_ok[i] = False

    # Unfold deltas for skein consistency analysis
    if fold_info.get('type') == 'conway_even':
        even_indices = fold_info['even_indices']
        delta_preds_full = conway_unfold(delta_preds, L_full, even_indices)
    else:
        delta_preds_full = delta_preds

    result = {
        'attributions': attributions,
        'delta_preds': delta_preds,
        'delta_preds_full': delta_preds_full,
        'pred_original': pred_orig,
        'signs': signs,
        'n_crossings': n,
        'flip_ok': flip_ok,
    }

    if y_true_folded is not None:
        result['y_true_folded'] = y_true_folded
        result['pred_error_l1'] = (pred_orig - y_true_folded).abs().sum().item()

    return result

class EmbeddingIntervenor:
    """Extract and intervene on per-node embeddings at each GNN layer."""
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.layer_embeddings = {}
        self.ablation_config = None  # (layer_name, node_indices_to_zero)

        encoder = model.encoder if hasattr(model, 'encoder') else model
        if hasattr(encoder, 'convs'):
            for i, conv in enumerate(encoder.convs):
                name = f'conv_{i}'
                hook = conv.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]

            # Store clean embeddings
            self.layer_embeddings[name] = output.detach().clone()

            # Apply ablation if configured for this layer
            if self.ablation_config is not None:
                abl_layer, abl_nodes = self.ablation_config
                if name == abl_layer and abl_nodes is not None:
                    # Zero out specified nodes' embeddings
                    output = output.clone()
                    output[abl_nodes] = 0.0
                    return output
        return hook_fn

    def clean_forward(self, g, device):
        """Run model without intervention, store all embeddings."""
        self.ablation_config = None
        self.layer_embeddings = {}
        self.model.eval()
        g = g.to(device)
        with torch.no_grad():
            logits, value_raw = self.model(g)
        return logits, value_raw

    def ablated_forward(self, g, device, layer_name, node_indices):
        """Run model with specific nodes zeroed at a specific layer."""
        self.ablation_config = (layer_name, node_indices)
        self.model.eval()
        g = g.to(device)
        with torch.no_grad():
            logits, value_raw = self.model(g)
        self.ablation_config = None
        return logits, value_raw

    def cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def pred_from_outputs(logits, value_raw, codec):
    """Convert model outputs to soft-gated prediction."""
    p = torch.sigmoid(logits)
    v = decode_values(value_raw, value_mode='classification',
                      codec=codec, decode_mode='expected')
    return (p * v).squeeze(0).cpu()


# Method 1: Node ablation at final layer
def node_ablation_attribution(model, pd_list, device, kd, codec,
                               layer='final'):
    """Per-crossing attribution by zeroing each crossing's embedding."""
    g = build_graph_for_inference(pd_list)
    n_crossings = len(pd_list)
    # In strand-split graph: nodes 0..n-1 are under, nodes n..2n-1 are over
    n_strand_nodes = g.num_nodes
    assert n_strand_nodes == 2 * n_crossings, \
        f"Expected {2*n_crossings} nodes, got {n_strand_nodes}"

    intervenor = EmbeddingIntervenor(model)

    # Find layer name
    if layer == 'final':
        n_layers = len(intervenor.layer_embeddings) if intervenor.layer_embeddings else 0
        # Do a clean forward to discover layer names
        intervenor.clean_forward(g, device)
        layer_names = sorted(intervenor.layer_embeddings.keys())
        layer_name = layer_names[-1] if layer_names else 'conv_5'
    else:
        layer_name = layer
        intervenor.clean_forward(g, device)

    # Clean prediction
    logits_clean, vraw_clean = intervenor.clean_forward(g, device)
    pred_clean = pred_from_outputs(logits_clean, vraw_clean, codec)

    # Ablate each crossing
    attributions = torch.zeros(n_crossings)

    for ci in range(n_crossings):
        # Ablate both strand-split nodes for this crossing
        nodes_to_ablate = torch.tensor([ci, ci + n_crossings],
                                        dtype=torch.long, device=device)

        logits_abl, vraw_abl = intervenor.ablated_forward(
            g, device, layer_name, nodes_to_ablate)
        pred_abl = pred_from_outputs(logits_abl, vraw_abl, codec)

        attributions[ci] = (pred_clean - pred_abl).abs().sum()

    intervenor.cleanup()
    return attributions, pred_clean


# Method 3: Greedy minimal subgraph

def find_minimal_subgraph(model, pd_list, device, kd, codec,
                           tolerance=0.1):
    """Find the smallest set of crossings sufficient for prediction.

    Greedy backward elimination. Repeatedly remove the crossing whose
    ablation changes the prediction least, until the prediction degrades
    beyond tolerance.
    """
    g = build_graph_for_inference(pd_list)
    n_crossings = len(pd_list)

    intervenor = EmbeddingIntervenor(model)
    intervenor.clean_forward(g, device)
    layer_names = sorted(intervenor.layer_embeddings.keys())
    final_layer = layer_names[-1]

    logits_clean, vraw_clean = intervenor.clean_forward(g, device)
    pred_clean = pred_from_outputs(logits_clean, vraw_clean, codec)

    # Track which crossings are still active
    active = set(range(n_crossings))
    removal_order = []
    current_ablated = set()

    while len(active) > 1:
        # Find the least-important remaining crossing
        best_ci = None
        best_change = float('inf')

        for ci in active:
            # Test ablating ci plus all previously removed crossings
            test_ablated = current_ablated | {ci}
            nodes = []
            for c in test_ablated:
                nodes.extend([c, c + n_crossings])
            nodes_t = torch.tensor(nodes, dtype=torch.long, device=device)

            logits_abl, vraw_abl = intervenor.ablated_forward(
                g, device, final_layer, nodes_t)
            pred_abl = pred_from_outputs(logits_abl, vraw_abl, codec)
            change = (pred_clean - pred_abl).abs().sum().item()

            if change < best_change:
                best_change = change
                best_ci = ci

        # Check if removing this crossing exceeds tolerance
        # (use per-coefficient MAE, not sum)
        test_ablated = current_ablated | {best_ci}
        nodes = []
        for c in test_ablated:
            nodes.extend([c, c + n_crossings])
        nodes_t = torch.tensor(nodes, dtype=torch.long, device=device)

        logits_abl, vraw_abl = intervenor.ablated_forward(
            g, device, final_layer, nodes_t)
        pred_abl = pred_from_outputs(logits_abl, vraw_abl, codec)
        mae = (pred_clean - pred_abl).abs().mean().item()

        removal_order.append((best_ci, best_change, mae))

        if mae > tolerance:
            break

        current_ablated.add(best_ci)
        active.remove(best_ci)

    intervenor.cleanup()
    return sorted(active), removal_order, pred_clean


class SingleGraphLayerExtractor:
    """
    Hooks conv layers and returns pooled graph embeddings (mean over nodes) for a single graph.
    """
    def __init__(self, model):
        self.model = model
        self.layer_outputs = {}
        self.hooks = []

        encoder = model.encoder if hasattr(model, 'encoder') else model
        if hasattr(encoder, 'convs'):
            for i, conv in enumerate(encoder.convs):
                h = conv.register_forward_hook(self._make_hook(f'conv_{i}'))
                self.hooks.append(h)
                print(f"  Hooked layer conv_{i}")

    def _make_hook(self, name):
        def hook_fn(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            self.layer_outputs[name] = output.detach()
        return hook_fn

    @torch.no_grad()
    def get_pooled_embeddings(self, pd_list, device):
        """Returns dict: layer_name -> np.array([d]) pooled embedding for that layer."""
        self.model.eval()
        self.layer_outputs = {}

        g = build_graph_for_inference(pd_list).to(device)
        _ = self.model(g)

        out = {}
        for name, node_emb in self.layer_outputs.items():
            pooled = node_emb.mean(dim=0)  # [d]
            out[name] = pooled.cpu().numpy()
        return out

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

def predict_folded_tensor(model, g, value_mode="classification", codec=None, decode_mode="expected"):
    """Differentiable prediction (no torch.no_grad). Returns tensor [L_folded]."""
    logits, value_raw = model(g)
    p = torch.sigmoid(logits)  # [1, L]
    v = decode_values(value_raw, value_mode=value_mode, codec=codec, decode_mode=decode_mode)  # [1, L]
    return (p * v).squeeze(0)  # [L]

def crossing_to_node_mask(m_cross, N_cross):
    """Expand crossing mask [N] -> node mask [2N, 1] (under + over)."""
    m_node = torch.cat([m_cross, m_cross], dim=0).view(2 * N_cross, 1)
    return m_node
