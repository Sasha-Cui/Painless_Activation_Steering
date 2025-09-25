# build_steer_vetcors_functions.py
# ─────────────────────────────── Build Steer Vectors ──────────────────────────────
from preambles import *
from util import get_layer
# Shared Utility for Extracting Activations
@torch.inference_mode()
def get_single_activation(
    model,
    tok,
    prompt: str,
    layer: int,
    steer_target: str = "residual",
    token_mode: str = "last",
    device: str = "cuda"
) -> torch.Tensor:
    
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    
    activations = {}    
    # To extract the residual hidden state, use hidden_states without the need of hooks
    if steer_target == "residual":
        out = model(**enc, output_hidden_states=True, return_dict=True)
        # Handle both tuple and list cases
        if isinstance(out.hidden_states, tuple):
            hidden = out.hidden_states[layer]# [1, seq_len, hidden_dim]
        elif isinstance(out.hidden_states, list):
            hidden = out.hidden_states[layer]
        else:
            # If it's some other structure, try direct indexing
            try:
                hidden = out.hidden_states[layer]
            except (TypeError, KeyError, IndexError) as e:
                print(f"variable layer value: {layer}")
                print(f"Unexpected hidden_states structure: {type(out.hidden_states)}")
                print(f"Available attributes: {dir(out.hidden_states)}")
                raise e

    # Otherwise, register a forward‐hook on the block’s submodule
    else:
        # “block” is your Transformer block; adjust this to your model’s API:
        block = get_layer(model, layer)   

        # pick which submodule to hook
        if steer_target == "mlp":
            # support both LlamaBlock.mlp and MistralBlock.feed_forward
            module_to_hook = getattr(block, "mlp", None) or getattr(block, "feed_forward", None)
            if module_to_hook is None:
                raise ValueError(f"No MLP/feed-forward module found on block: {block}")
        elif steer_target == "self_attn":
            module_to_hook = block.self_attn
        elif steer_target in ("post_attention_layernorm", "post_attn"):
            module_to_hook = block.post_attention_layernorm
        else:
            raise ValueError(f"Unknown steer_target: {steer_target}")

        # hook it
        handle = module_to_hook.register_forward_hook(
            lambda _mod, _inp, out: activations.setdefault("h", out)
        )

        # do a forward pass (no need for hidden_states here)
        _ = model(**enc, return_hidden_states=False, return_dict=True)

        # remove hook so we don’t leak memory
        handle.remove()

        # now our “hidden” is whatever that submodule spit out
        hidden = activations["h"]           # [1, seq_len, hidden_dim]
        if isinstance(hidden, tuple):
            hidden = hidden[0]  # Take the actual tensor output

    # Pick the token-position        
    if token_mode == "last":
        idx = enc["attention_mask"].sum(dim=1).item() - 1
        vec = hidden[0, idx]
    elif token_mode == "mid":
        idx = (enc["attention_mask"].sum(dim=1) // 2).item()
        vec = hidden[0, idx]
    elif token_mode == "mean":
        mask = enc["attention_mask"].unsqueeze(-1)
        vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        vec = vec[0]
    else:
        raise ValueError(f"Unsupported token_mode: {token_mode}")

    return vec.detach()

# Helper Functions for Constructing Positive and Negative Inputs

# # Method: (PAS (Full MCQ)). We include all answer choices. The positive input is the
# # mean activation over correctly answered multiple-choice questions; the negative input is the
# # mean over incorrectly answered ones.
def build_bas_full_mcq(df, predictions):
    pos_inputs = df.loc[predictions == df["correct_letter"], "full_prompt"]
    neg_inputs = df.loc[predictions != df["correct_letter"], "full_prompt"]
    return pos_inputs.tolist(), neg_inputs.tolist()

# # Method (iPAS (All)). We include only the answer selected by the LLM. The positive
# # input uses the correct answer; the negative input uses the model’s incorrect answer.
def build_ibas_all(df, predictions):
    pos_inputs = df.loc[predictions == df["correct_letter"]].apply(
        lambda row: f"{row['question']} {row['correct_choice']}", axis=1)
    neg_inputs = df.loc[predictions != df["correct_letter"]].apply(
        lambda row: f"{row['question']} {row['choices'][predictions[row.name]]}", axis=1)
    return pos_inputs.tolist(), neg_inputs.tolist()

# # Method (iPAS (Wrong Only)). Restricted to incorrectly answered examples. The posi-
# # tive input uses the correct answer; the negative input uses the choice selected by the LLM.
def build_ibas_wrong_only(df, predictions):
    wrong_df = df[predictions != df["correct_letter"]]
    pos_inputs = wrong_df.apply(lambda row: f"{row['question']} {row['correct_choice']}", axis=1)
    neg_inputs = wrong_df.apply(lambda row: f"{row['question']} {row['choices'][predictions[row.name]]}", axis=1)
    return pos_inputs.tolist(), neg_inputs.tolist()


# SteeringVectorBuilder Class 
@dataclass
class SteeringVectorBuilder:
    model: torch.nn.Module
    tok: any
    layer: int = 15
    steer_target: str = "residual"
    token_mode: Literal["last", "mid", "mean"] = "last"
    device: str = "cuda" 

    def _get_activations(self, prompts: List[str], label: str) -> torch.Tensor:
        self.model.eval()
        activations = []

        for prompt in tqdm(prompts, desc=f"Computing {label} activations", disable=not sys.stdout.isatty(), leave=False, file = sys.__stdout__):
            act = get_single_activation(self.model, self.tok, prompt, self.layer, self.steer_target, self.token_mode, self.device)
            activations.append(act)

        if len(activations) == 0:
            raise ValueError(f"No {label} activations to compute steer vector.")
        return torch.stack(activations).mean(dim=0)
        
    def _log_vector_status(self, pos_inputs: List[str], neg_inputs: List[str]):
        if len(pos_inputs) == 0:
            raise ValueError("Insufficient correct activations to build steering vector.")
        if len(neg_inputs) == 0:
            raise ValueError("Insufficient incorrect activations to build steering vector.")
    
    def build_BAS_full_mcq(self, df, predictions):
        pos_inputs, neg_inputs = build_bas_full_mcq(df, predictions)
        self._log_vector_status(pos_inputs, neg_inputs)
        pos_avg = self._get_activations(pos_inputs, "correct")
        neg_avg = self._get_activations(neg_inputs, "incorrect")
        steer_vector = pos_avg - neg_avg
        print(f"✅ BAS (Full MCQ) steer vector built.")
        return steer_vector
    
    def build_iBAS_all(self, df, predictions):
        pos_inputs, neg_inputs = build_ibas_all(df, predictions)
        self._log_vector_status(pos_inputs, neg_inputs)
        pos_avg = self._get_activations(pos_inputs, "correct")
        neg_avg = self._get_activations(neg_inputs, "incorrect")
        steer_vector = pos_avg - neg_avg
        print(f"✅ iBAS (All) steer vector built.")
        return steer_vector
    
    def build_iBAS_wrong_only(self, df, predictions):
        pos_inputs, neg_inputs = build_ibas_wrong_only(df, predictions)
        self._log_vector_status(pos_inputs, neg_inputs)
        pos_avg = self._get_activations(pos_inputs, "correct")
        neg_avg = self._get_activations(neg_inputs, "incorrect")
        steer_vector = pos_avg - neg_avg
        print(f"✅ iBAS (Wrong Only) steer vector built.")
        return steer_vector

@torch.inference_mode()
def _hidden_batch(
    model,
    tok,
    prompts: Sequence[str],
    layer: int,
    steer_target: str = "residual",      # "residual" | "mlp" | "self_attn" | "post_attn" or "post_attention_layernorm"
    token_mode: str = "last",            # "last" | "mid" | "mean"
    device: str = "cuda"
) -> Tensor:
    """
    Extract activations at one of four locations for a batch of prompts:
      - residual  : hidden_states[layer]
      - mlp       : the block's feed-forward output
      - self_attn : the block's self-attention output
      - post_attn : the block's post-attn layernorm output

    Returns a [batch, hidden_dim] tensor according to token_mode.
    """
    # 1️⃣ Tokenize
    device = next(model.parameters()).device
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}

    # 2️⃣ Decide where to grab activations
    if steer_target == "residual":
        out = model(**enc, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[layer]  # [batch, seq_len, hidden_dim]

    else:
        # helper to get the right block
        block = get_layer(model, layer)

        # pick submodule
        if steer_target == "self_attn":
            module_to_hook = block.self_attn
        elif steer_target in ("post_attn", "post_attention_layernorm"):
            module_to_hook = block.post_attention_layernorm
        elif steer_target == "mlp":
            module_to_hook = getattr(block, "mlp", None) or getattr(block, "feed_forward", None)
        else:
            raise ValueError(f"Unknown steer_target={steer_target!r}")

        # hook it
        activations = {}
        handle = module_to_hook.register_forward_hook(
            lambda _m, _i, out: activations.setdefault("h", out)
        )

        # single forward pass for the batch
        _ = model(**enc, return_dict=True)

        handle.remove()
        hidden = activations["h"]           # [batch, seq_len, hidden_dim]
        if isinstance(hidden, tuple):
            hidden = hidden[0]  # take the first tensor (usually the attention output)

    # 3️⃣ Pool over tokens
    mask = enc["attention_mask"]
    if token_mode == "last":
        idxs = mask.sum(dim=1) - 1       # [batch]
        return torch.stack([hidden[i, idx.item()] for i, idx in enumerate(idxs)])

    elif token_mode == "mid":
        mids = (mask.sum(dim=1) // 2).long()
        return torch.stack([hidden[i, idx.item()] for i, idx in enumerate(mids)])

    elif token_mode == "mean":
        mask_unsq = mask.unsqueeze(-1)   # [batch, seq_len, 1]
        summed = (hidden * mask_unsq).sum(dim=1)
        counts = mask_unsq.sum(dim=1)
        return summed / counts            # [batch, hidden_dim]

    else:
        raise ValueError(f"Unsupported token_mode={token_mode!r}")


def build_steer_vec_handcrafted_prompt_pair(
    model,
    tok,
    pairs: Sequence[Tuple[str, str]],
    layer: int,
    steer_target: str = "residual", 
    device: str = "cuda",
    batch: int = 64,
    token_mode: str = "last"
) -> Tensor:
    diffs = []
    total_batches = (len(pairs) + batch - 1) // batch
    for i in tqdm(range(total_batches), desc="Building Pair-CAA", disable=not sys.stdout.isatty(), leave=False, file = sys.__stdout__):
        chunk = pairs[i * batch: (i + 1) * batch]
        truths, fictions = zip(*chunk)
        h_t = _hidden_batch(model, tok, truths, layer, token_mode=token_mode, steer_target=steer_target, device=device)
        h_f = _hidden_batch(model, tok, fictions, layer, token_mode=token_mode, steer_target=steer_target, device=device)
        diffs.append(h_t - h_f)
    print(f"✅ Pair Steer vector built out of {len(pairs)} prompt pairs.")
    return torch.cat(diffs).mean(0).detach()


