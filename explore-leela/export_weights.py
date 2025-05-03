import gzip
import os
import weights_pb2  # Import the generated module
import graphviz  # Import the graphviz library

# --- Configuration ---
WEIGHTS_FILE_PATH = 't1-256x10-distilled-swa-2432500.pb.gz'  # <--- CHANGE THIS
OUTPUT_GRAPH_FILE = 'lc0_network_graph'  # Output file name (without extension)
OUTPUT_FORMAT = 'pdf'  # Output format ('pdf', 'png', 'svg', etc.)


# --- Helper function to add nodes/edges ---
#     Simplifies adding common blocks and managing connections

def add_node(dot, node_id, label, shape='box', style='', parent_id=None):
    """Adds a node and optionally an edge from parent_id."""
    dot.node(node_id, label=label, shape=shape, style=style)
    if parent_id:
        dot.edge(parent_id, node_id)
    return node_id  # Return the id of the node just added


def add_conv_block(dot, block_data, node_prefix, label_prefix, parent_id):
    """Adds nodes representing a ConvBlock (weights, biases, bn)"""
    block_id = f"{node_prefix}_convblock"
    # Represent ConvBlock as a single node for simplicity
    # You could expand this to show weights, bias, bn layers if needed
    details = []
    if block_data.HasField('weights'): details.append('W')
    if block_data.HasField('biases'): details.append('B')
    if block_data.HasField('bn_means'): details.append('BN')
    label = f"{label_prefix} ConvBlock\n({', '.join(details)})"
    return add_node(dot, block_id, label, shape='box', parent_id=parent_id)


def add_layer(dot, layer_data, node_prefix, label_prefix, parent_id):
    """Adds a node representing a simple Layer (like dense layer weights/biases)"""
    layer_id = f"{node_prefix}_layer"
    details = []
    if layer_data.HasField('params'): details.append(f"{len(layer_data.params)} bytes")
    if layer_data.HasField('min_val'): details.append(f"min={layer_data.min_val:.2f}")
    if layer_data.HasField('max_val'): details.append(f"max={layer_data.max_val:.2f}")
    label = f"{label_prefix}\n({', '.join(details)})"
    return add_node(dot, layer_id, label, shape='ellipse', parent_id=parent_id)


# --- Main Graph Generation Logic ---

def create_computation_graph(net_message, filename='network_graph', format='pdf'):
    """
    Generates a computation graph visualization from the parsed Net message.
    """
    dot = graphviz.Digraph(comment='LCZero Network Structure', format=format)
    dot.attr(rankdir='TB')  # Top-to-Bottom layout
    dot.attr(splines='ortho')  # Use orthogonal lines for edges if possible
    # dot.attr(concentrate='true') # Try to merge parallel edges

    # Get Network Structure Type for conditional logic
    network_type = None
    network_type_name = "UNKNOWN"
    if net_message.HasField('format') and net_message.format.HasField(
            'network_format') and net_message.format.network_format.HasField('network'):
        network_type = net_message.format.network_format.network
        network_type_name = weights_pb2.NetworkFormat.NetworkStructure.Name(network_type)

    dot.attr(label=f'LCZero Network Structure ({network_type_name})')
    dot.attr(fontsize='20')

    if not net_message.HasField('weights'):
        print("Error: No 'weights' field found in the message.")
        if net_message.HasField('onnx_model'):
            print("Found 'onnx_model' field. Graph generation from ONNX bytes is not supported by this script.")
        return None

    weights = net_message.weights
    last_node_id = add_node(dot, 'input_data', 'Input Planes', shape='folder')
    trunk_output_id = last_node_id  # Track the output of the main network trunk

    # --- 1. Input Block ---
    if weights.HasField('input'):
        trunk_output_id = add_conv_block(dot, weights.input, 'input', 'Input', trunk_output_id)

    # --- Architecture Specific Trunk ---

    # ATTENTION BODY Networks (NETWORK_ATTENTIONBODY_*)
    is_attention_body = network_type in [
        weights_pb2.NetworkFormat.NetworkStructure.NETWORK_ATTENTIONBODY_WITH_HEADFORMAT,
        weights_pb2.NetworkFormat.NetworkStructure.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT,
        weights_pb2.NetworkFormat.NetworkStructure.NETWORK_AB_LEGACY_WITH_MULTIHEADFORMAT,
    ]

    if is_attention_body:
        # Optional Preprocessing Layer
        if weights.HasField('ip_emb_preproc_w'):
            parent = trunk_output_id
            if weights.HasField('ip_emb_preproc_w'):
                parent = add_layer(dot, weights.ip_emb_preproc_w, 'emb_pre_w', 'Emb Preproc W', parent)
            if weights.HasField('ip_emb_preproc_b'):
                parent = add_layer(dot, weights.ip_emb_preproc_b, 'emb_pre_b', 'Emb Preproc B', parent)
            trunk_output_id = parent

        # Main Embedding Layer
        if weights.HasField('ip_emb_w') or weights.HasField('ip_emb_b'):
            parent = trunk_output_id
            emb_id = f"embedding_block"
            add_node(dot, emb_id, "Embedding", shape='rect', style='dashed', parent_id=parent)  # Grouping node
            parent_internal = emb_id
            if weights.HasField('ip_emb_w'):
                parent_internal = add_layer(dot, weights.ip_emb_w, 'emb_w', 'Emb W', parent_internal)
            if weights.HasField('ip_emb_b'):
                parent_internal = add_layer(dot, weights.ip_emb_b, 'emb_b', 'Emb B', parent_internal)
            if weights.HasField('ip_emb_ln_gammas'):
                parent_internal = add_layer(dot, weights.ip_emb_ln_gammas, 'emb_ln_g', 'Emb LN Gamma', parent_internal)
            if weights.HasField('ip_emb_ln_betas'):
                parent_internal = add_layer(dot, weights.ip_emb_ln_betas, 'emb_ln_b', 'Emb LN Beta', parent_internal)
            # Optional Input Gating
            if weights.HasField('ip_mult_gate') or weights.HasField('ip_add_gate'):
                parent_internal = add_node(dot, 'input_gate', 'Input Gate', shape='invhouse', parent_id=parent_internal)
            # Optional Embedding FFN
            if weights.HasField('ip_emb_ffn'):
                ffn_id = 'emb_ffn'
                add_node(dot, ffn_id, "Emb FFN", shape='rect', style='dashed', parent_id=parent_internal)
                parent_internal = ffn_id  # Connect sub-layers to this
                # Add FFN dense layers if needed
                if weights.ip_emb_ffn.HasField('dense1_w'): add_layer(dot, weights.ip_emb_ffn.dense1_w, f'{ffn_id}_d1w',
                                                                      'Dense1 W', parent_internal)
                # ... add other FFN components ...
            trunk_output_id = parent_internal  # Output of embedding stage

        # Encoder Stack
        num_encoders = len(weights.encoder)
        if num_encoders > 0:
            parent = trunk_output_id
            for i, encoder_layer in enumerate(weights.encoder):
                enc_id = f"encoder_{i}"
                add_node(dot, enc_id, f"Encoder Layer {i}", shape='tab', style='filled', fillcolor='lightgrey',
                         parent_id=parent)
                # Optionally visualize MHA/FFN inside encoder - Keeping it high level for now
                # You could create a subgraph for each encoder layer here
                parent = enc_id
            trunk_output_id = parent

    # CLASSICAL / SE Networks
    else:
        # Residual Tower
        num_residual_blocks = len(weights.residual)
        if num_residual_blocks > 0:
            parent = trunk_output_id
            for i, res_block in enumerate(weights.residual):
                res_id = f"res_{i}"
                label = f"Residual Block {i}"
                if res_block.HasField('se'): label += "\n(with SE)"
                # Using Mrecord shape to hint at internal structure (Conv-BN-ReLU -> Conv-BN-ReLU -> Add)
                add_node(dot, res_id, label, shape='record', style='filled', fillcolor='lightblue', parent_id=parent)
                # Add edges for skip connection visualization (optional, can clutter)
                # dot.edge(parent, res_id, style='dashed', constraint='false')
                parent = res_id
            trunk_output_id = parent

    # --- Split Point for Heads ---
    # Create a central node from which heads diverge
    split_node_id = add_node(dot, 'trunk_output', 'Trunk Output', shape='point', parent_id=trunk_output_id)

    # --- 3. Policy Head(s) ---
    policy_parent = split_node_id
    if weights.HasField('policy_heads'):  # Multi-head format
        ph = weights.policy_heads
        multihead_policy_id = add_node(dot, 'policy_multihead', 'Policy Heads', shape='diamond', style='filled',
                                       fillcolor='coral', parent_id=policy_parent)
        policy_parent = multihead_policy_id  # Heads branch from here

        # Shared layers? (e.g., ip_pol_w/b in some attention variants)
        if ph.HasField('ip_pol_w'):
            policy_parent = add_layer(dot, ph.ip_pol_w, 'policy_heads_emb_w', 'PH Emb W', policy_parent)
        # Add ip_pol_b if exists...

        # Individual heads
        if ph.HasField('vanilla'):
            add_node(dot, 'policy_out_vanilla', 'Policy Output\n(Vanilla)', shape='oval', parent_id=policy_parent)
        if ph.HasField('optimistic_st'):
            add_node(dot, 'policy_out_ost', 'Policy Output\n(Optimistic ST)', shape='oval', parent_id=policy_parent)
        # Add other named heads (soft, opponent) if present...
        for i, head_map in enumerate(ph.policy_head_map):
            head_name = head_map.key
            head_id = f'policy_out_map_{i}_{head_name}'
            add_node(dot, head_id, f'Policy Output\n({head_name})', shape='oval', parent_id=policy_parent)

    else:  # Legacy/Single Policy Head
        # Check for different structures (Conv, Attention layers)
        policy_head_id = add_node(dot, 'policy_head', 'Policy Head', shape='invtrapezium', style='filled',
                                  fillcolor='lightcoral', parent_id=policy_parent)
        parent = policy_head_id

        if weights.HasField('policy1'): parent = add_conv_block(dot, weights.policy1, 'policy1', 'Policy Conv1', parent)
        if weights.HasField('policy'): parent = add_conv_block(dot, weights.policy, 'policy', 'Policy Conv2/Final',
                                                               parent)
        if weights.HasField('ip_pol_w'): parent = add_layer(dot, weights.ip_pol_w, 'policy_ipw', 'Policy IP W', parent)
        if weights.HasField('ip_pol_b'): parent = add_layer(dot, weights.ip_pol_b, 'policy_ipb', 'Policy IP B', parent)
        # Add ip2/ip3/ip4 for policy attention if present
        if weights.HasField('ip2_pol_w'): parent = add_layer(dot, weights.ip2_pol_w, 'policy_ip2w', 'Policy IP2 W (WQ)',
                                                             parent)
        # ... etc for ip2_b, ip3_w/b, ip4_w ...

        add_node(dot, 'policy_output', 'Policy Output', shape='oval', parent_id=parent)

    # --- 4. Value Head(s) ---
    value_parent = split_node_id  # Reset parent to trunk split
    if weights.HasField('value_heads'):  # Multi-head format
        vh = weights.value_heads
        multihead_value_id = add_node(dot, 'value_multihead', 'Value Heads', shape='diamond', style='filled',
                                      fillcolor='lightgreen', parent_id=value_parent)
        value_parent = multihead_value_id  # Heads branch from here

        # Shared layers? (ip_val_w/b in attention body)
        if weights.HasField('ip_val_w'):  # Check if shared value embedding exists at top level
            value_parent = add_layer(dot, weights.ip_val_w, 'value_heads_emb_w', 'VH Emb W', value_parent)
        # Add ip_val_b if exists...

        # Individual heads
        if vh.HasField('winner'):
            # Potentially add internal layers of winner head if needed before final output
            add_node(dot, 'value_out_winner', 'Value Output\n(Winner/WDL)', shape='oval', parent_id=value_parent)
        if vh.HasField('q'):
            add_node(dot, 'value_out_q', 'Value Output\n(Q)', shape='oval', parent_id=value_parent)
        if vh.HasField('st'):
            add_node(dot, 'value_out_st', 'Value Output\n(ST)', shape='oval', parent_id=value_parent)
        for i, head_map in enumerate(vh.value_head_map):
            head_name = head_map.key
            head_id = f'value_out_map_{i}_{head_name}'
            add_node(dot, head_id, f'Value Output\n({head_name})', shape='oval', parent_id=value_parent)

    else:  # Legacy/Single Value Head
        value_head_id = add_node(dot, 'value_head', 'Value Head', shape='invtrapezium', style='filled',
                                 fillcolor='lightgreen', parent_id=value_parent)
        parent = value_head_id

        if weights.HasField('value'): parent = add_conv_block(dot, weights.value, 'value_conv', 'Value Conv', parent)
        # Dense layers
        if weights.HasField('ip_val_w'): parent = add_layer(dot, weights.ip_val_w, 'value_ipw', 'Value IP W (Emb)',
                                                            parent)  # This is embedding in Attention Body
        if weights.HasField('ip_val_b'): parent = add_layer(dot, weights.ip_val_b, 'value_ipb', 'Value IP B (Emb)',
                                                            parent)
        if weights.HasField('ip1_val_w'): parent = add_layer(dot, weights.ip1_val_w, 'value_ip1w', 'Value IP1 W',
                                                             parent)
        if weights.HasField('ip1_val_b'): parent = add_layer(dot, weights.ip1_val_b, 'value_ip1b', 'Value IP1 B',
                                                             parent)
        if weights.HasField('ip2_val_w'): parent = add_layer(dot, weights.ip2_val_w, 'value_ip2w', 'Value IP2 W',
                                                             parent)
        if weights.HasField('ip2_val_b'): parent = add_layer(dot, weights.ip2_val_b, 'value_ip2b', 'Value IP2 B',
                                                             parent)

        add_node(dot, 'value_output', 'Value Output', shape='oval', parent_id=parent)

    # --- 5. Moves Left Head (Optional) ---
    if weights.HasField('moves_left') or \
            weights.HasField('ip1_mov_w') or \
            weights.HasField('ip_mov_w'):  # Check for conv or dense layers
        ml_parent = split_node_id  # Reset parent to trunk split
        ml_head_id = add_node(dot, 'ml_head', 'MovesLeft Head', shape='invtrapezium', style='filled',
                              fillcolor='lightyellow', parent_id=ml_parent)
        parent = ml_head_id

        if weights.HasField('moves_left'): parent = add_conv_block(dot, weights.moves_left, 'ml_conv', 'ML Conv',
                                                                   parent)
        if weights.HasField('ip_mov_w'): parent = add_layer(dot, weights.ip_mov_w, 'ml_ipw', 'ML IP W (Emb)', parent)
        # Add ip_mov_b if present
        if weights.HasField('ip1_mov_w'): parent = add_layer(dot, weights.ip1_mov_w, 'ml_ip1w', 'ML IP1 W', parent)
        # Add ip1_mov_b if present
        if weights.HasField('ip2_mov_w'): parent = add_layer(dot, weights.ip2_mov_w, 'ml_ip2w', 'ML IP2 W', parent)
        # Add ip2_mov_b if present

        add_node(dot, 'ml_output', 'MovesLeft Output', shape='oval', parent_id=parent)

    # --- Render Graph ---
    try:
        dot.render(filename, view=False)  # Set view=True to automatically open
        print(f"Graph saved to {filename}.{format}")
        return dot
    except Exception as e:
        print(f"Error rendering graph with Graphviz: {e}")
        print("Make sure Graphviz is installed and in your system's PATH.")
        return None


# --- Load Data and Generate Graph ---
if not os.path.exists(WEIGHTS_FILE_PATH):
    print(f"Error: Weights file not found at {WEIGHTS_FILE_PATH}")
else:
    print(f"Loading weights from: {WEIGHTS_FILE_PATH}")
    net_message = weights_pb2.Net()
    try:
        if WEIGHTS_FILE_PATH.endswith('.gz'):
            with gzip.open(WEIGHTS_FILE_PATH, 'rb') as f:
                file_content = f.read()
        else:
            with open(WEIGHTS_FILE_PATH, 'rb') as f:
                file_content = f.read()
        net_message.ParseFromString(file_content)
        print("Successfully parsed the weights file.")

        # Generate the graph
        create_computation_graph(net_message, filename=OUTPUT_GRAPH_FILE, format=OUTPUT_FORMAT)

    except Exception as e:
        print(f"An error occurred during loading or graph generation: {e}")
        import traceback

        traceback.print_exc()