#!/usr/bin/env python3
"""Arm B: trunk + branches tree attention for Qwen3.5 (GatedDeltaNet 24 layers + full attention 8 layers).

One sample = one trajectory::

    [trunk]      positions 0 .. trunk_len-1        every assistant turn think-stripped (generation-time view)
    [branch_k]   forks after trunk position fork_k  the FULL assistant turn k (think + output), supervised

Attention semantics
    trunk q   : trunk kv <= q
    branch q  : trunk kv < fork_k  (i.e. the visible stream only)  +  its own branch kv <= q
    branches never see each other, nor trunk tokens at/after their fork.

Implementation
    * full-attention layers (8): torch.nn.attention.flex_attention with a hand-built sparse BlockMask
      (never a dense [T,T] tensor: 262K**2 bool would be 68 GB).
    * linear-attention layers (24, GatedDeltaNet): the trunk is scanned continuously and split at the fork
      points, chaining `chunk_gated_delta_rule` initial_state -> output_final_state so the state at every fork
      is exact and gradients flow back into the trunk (fla returns dh0); each branch starts from S(fork_k).
      All branches are run as ONE packed varlen call (`cu_seqlens` + per-sequence `initial_state`), and the
      depthwise causal conv (W=4) runs on the same packing with the trunk's W-1 raw mixed_qkv columns before
      each fork prepended (then cut off), so every branch sees exactly the left context it had in the trunk.

Segments are aligned to the 128-token flex block grid by inserting pad tokens (labels=-100, excluded from
every scan and from every attention query) so a block never straddles two segments.

Layout / API::

    TreeDataCollator      cache row -> model batch   (input_ids, labels, position_ids, tree_* metadata)
    apply_tree_patch()    class-level monkey patch of Qwen3_5TextModel / Qwen3_5Attention / Qwen3_5GatedDeltaNet
    build_block_mask()    (seg, vis_end, T) -> BlockMask  (exported for the checks)
"""

import os

import torch
import torch.nn.functional as F

BLOCK = 128
# T_pad is rounded up to a multiple of this so flex_attention (compiled static-shape) sees at most
# 262144 / T_MULT distinct sequence lengths instead of one per row (= one recompile per row).
T_MULT = int(os.environ.get("TREE_T_MULT", "8192"))
MAX_SEG = int(os.environ.get("TREE_MAX_SEG", "1073741824"))  # safety valve: chain state over segments this long
# TREE_ATTN_SPLIT=1: trunk queries go through FlashAttention-2 (plain causal over the trunk, 2-3x faster than
# compiled flex) and only queries from (tr // T_MULT) * T_MULT onwards (trunk tail + pads + branches) go through
# flex with the tree mask. Equivalent by construction: a trunk query's tree mask IS the causal mask, so the
# overlap region [q0, tr) computed by flex matches what FA2 would give (bf16 noise apart).
ATTN_SPLIT = os.environ.get("TREE_ATTN_SPLIT", "0") == "1"
# grid for the FA2/flex query boundary (multiple of T_MULT). Coarser -> fewer distinct (Q_len, KV_len) flex
# shapes to compile (at most a few per T_pad bucket) at the cost of flex re-doing a bit more trunk tail.
SPLIT_MULT = max(T_MULT, int(os.environ.get("TREE_SPLIT_MULT", str(4 * T_MULT))))
# (only needed if a kernel refuses a very long T; the real fix for the fla autotune crash was
#  contiguous slices — non-contiguous q/k/v made an autotune config an invalid CUDA launch)
STATS = {"gdn_tree": 0, "gdn_orig": 0, "attn_tree": 0, "attn_orig": 0}


# --------------------------------------------------------------------------------------- layout
class TreeLayout:
    """Physical layout of one tree sample (all offsets are indices into the padded sequence)."""

    def __init__(self, trunk_len: int, forks: list[int], lens: list[int], pad_id: int):
        self.trunk_len = trunk_len
        self.forks = list(forks)
        self.lens = list(lens)
        self.pad_id = pad_id
        self.starts: list[int] = []
        pos = trunk_len
        for n in self.lens:
            pos += (-pos) % BLOCK  # branch starts on a block boundary
            self.starts.append(pos)
            pos += n
        self.T = pos
        # bucket T_pad to a multiple of T_MULT (bounded number of flex graph shapes); rows that are tiny
        # (dummy / no-branch rows) only go to one block so they cost nothing
        mult = max(BLOCK, (T_MULT // BLOCK) * BLOCK) if pos > BLOCK else BLOCK
        self.T_pad = pos + ((-pos) % mult)

    def build(self, input_ids: list[int], labels: list[int], position_ids: list[int]) -> dict:
        """Assemble the padded tensors from the cache row (trunk ++ branches, unpadded).

        Labels are placed at the position whose PREVIOUS position is the one that must predict the token
        (HF's shift-CE convention), which for the first token of a branch is the trunk position ``fork-1``::

            labels[fork_b]        = branch_b[0]      # predicted by logit[fork_b - 1]  (trunk, visible prefix)
            labels[s_b+1 .. s_b+L]= branch_b[1 .. L] # predicted by the branch's own previous position

        Without this the shifted CE would predict each branch's first token from a pad position.
        """
        n_b = len(self.lens)
        assert len(self.forks) == n_b, (len(self.forks), n_b)
        assert len(input_ids) == self.trunk_len + sum(self.lens), (len(input_ids), self.trunk_len, sum(self.lens))
        ids = list(input_ids[: self.trunk_len])
        lab = [-100] * self.T_pad
        pos = list(position_ids[: self.trunk_len])
        seg = [0] * self.trunk_len
        vis = list(range(1, self.trunk_len + 1))
        off = self.trunk_len
        n_lab = 0
        for b in range(n_b):
            n_pad = self.starts[b] - len(ids)
            ids += [self.pad_id] * n_pad
            pos += [0] * n_pad
            seg += [-1] * n_pad
            vis += [0] * n_pad
            L = self.lens[b]
            s = self.starts[b]
            br = list(input_ids[off: off + L])
            assert list(labels[off: off + L]) == br, "cache labels must equal the branch token ids"
            lab[self.forks[b]] = br[0]
            lab[s + 1: s + L] = br[1:]
            ids += br
            pos += list(position_ids[off: off + L])
            seg += [b + 1] * L
            vis += [self.forks[b]] * L
            off += L
            n_lab += L
        n_pad = self.T_pad - len(ids)
        ids += [self.pad_id] * n_pad
        pos += [0] * n_pad
        seg += [-1] * n_pad
        vis += [0] * n_pad
        assert n_lab == sum(self.lens)
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "labels": torch.tensor([lab], dtype=torch.long),
            "position_ids": torch.tensor([pos], dtype=torch.long),
            "tree_seg": torch.tensor([seg], dtype=torch.int32),
            "tree_vis_end": torch.tensor([vis], dtype=torch.int32),
            "tree_forks": torch.tensor([self.forks], dtype=torch.long),
            "tree_branch_starts": torch.tensor([self.starts], dtype=torch.long),
            "tree_branch_lens": torch.tensor([self.lens], dtype=torch.long),
            "tree_trunk_len": torch.tensor(self.trunk_len, dtype=torch.long),
        }


def tree_layout_from_row(row: dict, pad_id: int) -> TreeLayout:
    lens = row["tree_branch_lens"]
    forks = row["tree_forks"]
    if torch.is_tensor(lens):
        lens = lens.tolist()
    if torch.is_tensor(forks):
        forks = forks.tolist()
    trunk_len = int(row["trunk_len"])
    return TreeLayout(trunk_len, [int(f) for f in forks], [int(n) for n in lens], pad_id)


# ----------------------------------------------------------------------------------- block mask
def build_block_mask(seg: torch.Tensor, vis_end: torch.Tensor, forks: list[int], starts: list[int],
                     lens: list[int], trunk_len: int, device, block: int = BLOCK, q_start: int = 0):
    """Sparse BlockMask for the tree structure (seg: [T] int32, 0 = trunk, b+1 = branch b, -1 = pad).

    With ``q_start`` > 0 the mask covers only query positions [q_start, T) (KV stays the full range):
    the ATTN_SPLIT path computes trunk queries with FA2 and hands flex only the tail. q_start must be
    block-aligned."""
    T = seg.numel()
    assert T % block == 0, (T, block)
    assert q_start % block == 0, (q_start, block)
    n_q = T // block
    q0b = q_start // block
    seg_blk = seg[: n_q * block : block].tolist()  # segment of every query block (block-aligned segments)
    # kv blocks per query block
    rows: list[torch.Tensor] = []
    for i in range(q0b, n_q):
        s = seg_blk[i]
        if s is None:
            s = -1
        if s == 0:  # trunk: causal within the trunk
            rows.append(torch.arange(0, i + 1, dtype=torch.int32, device=device))
        elif s > 0:
            b = s - 1
            k_end = (forks[b] - 1) // block  # last trunk block holding a position < fork_b
            own = i - starts[b] // block
            rows.append(torch.cat([torch.arange(0, k_end + 1, dtype=torch.int32, device=device),
                                   torch.arange(i - own, i + 1, dtype=torch.int32, device=device)]))
        else:  # pad block: self only
            rows.append(torch.tensor([i], dtype=torch.int32, device=device))
    # `_transpose_ordered` derives the dense width from kv_indices.shape[-1], so it must be n_q (the number of
    # kv blocks), not the max row length: sparse rows are zero-padded and `kv_num_blocks` limits them.
    max_kv = n_q
    kv_idx = torch.zeros((1, 1, n_q - q0b, max_kv), dtype=torch.int32, device=device)
    kv_num = torch.zeros((1, 1, n_q - q0b), dtype=torch.int32, device=device)
    for i, r in enumerate(rows):
        kv_idx[0, 0, i, : r.numel()] = r
        kv_num[0, 0, i] = r.numel()

    from torch.nn.attention.flex_attention import BlockMask

    def mask_mod(b_idx, h_idx, q_idx, kv_idx_: torch.Tensor):
        # Trunk keys are visible to ANY query whose vis_end says so (vis_end[q] = q+1 for trunk queries,
        # fork_k for branch k), branch keys only to their own branch. Restricting the first clause to
        # trunk queries (sq == 0) would cut the branch -> trunk path entirely.
        q_abs = q_idx + q_start
        sq = seg[q_abs]
        sk = seg[kv_idx_]
        allowed = ((sk == 0) & (kv_idx_ < vis_end[q_abs])) | \
                  ((sq > 0) & (sq == sk) & (kv_idx_ <= q_abs))
        return allowed | ((sq < 0) & (q_abs == kv_idx_))

    return BlockMask.from_kv_blocks(
        kv_num, kv_idx, BLOCK_SIZE=block, mask_mod=mask_mod, seq_lengths=(T - q_start, T)
    )


# ---------------------------------------------------------------------------------- model patches
def _conv(x: torch.Tensor, weight, bias, activation):
    """Depthwise causal conv, channel-first [1, C, L]."""
    from causal_conv1d import causal_conv1d_fn

    return causal_conv1d_fn(x=x, weight=weight, bias=bias, activation=activation)


# fla's varlen causal_conv1d computes `y + bos*D + t*D` in int32: with D = 8192 channels (Qwen3.5-4B GDN) the
# offset wraps once the packed length passes 2^31 / D = 262144 tokens -> garbage reads (NaN grads at flash
# step 1, rows with 251K-275K branch tokens) or an illegal memory access (check_conv_len.py at 275K).
# The packed sequence is therefore convolved in chunks of whole segments below this limit; segments are
# independent (each restarts the receptive field), so the chunked result is identical.
CONV_MAX_ELEMS = 2**31 - 2**24


class _ContigGrad(torch.autograd.Function):
    """Identity whose backward hands the producer a contiguous grad: a chunk's grad otherwise arrives as a
    strided view of the full [1, C, L] grad, whose element offsets overflow fla's int32 indexing past 2^31."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.contiguous()


def _conv_varlen(x: torch.Tensor, weight, bias, activation, cu_seqlens: torch.Tensor):
    """Depthwise causal conv on packed sequences (channel-first in/out); each [cu[i], cu[i+1]) restarts the
    receptive field. Uses fla's varlen kernel (token-first layout), verified exact vs per-sequence calls."""
    from fla.modules.convolution import causal_conv1d as fla_causal_conv1d

    C, L = x.shape[1], x.shape[2]
    max_len = CONV_MAX_ELEMS // C
    xt = x.transpose(1, 2).contiguous()  # [1, L, C]
    if L <= max_len:
        y, _ = fla_causal_conv1d(x=xt, weight=weight, bias=bias, activation=activation, cu_seqlens=cu_seqlens)
        return y.transpose(1, 2)
    cu = cu_seqlens.tolist()
    assert max(b - a for a, b in zip(cu, cu[1:])) <= max_len, "a single packed segment exceeds the conv length limit"
    outs = []
    i = 0
    while i < len(cu) - 1:
        j = i + 1
        while j + 1 < len(cu) and cu[j + 1] - cu[i] <= max_len:
            j += 1
        a, b = cu[i], cu[j]
        cu_c = torch.tensor([c - a for c in cu[i: j + 1]], dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        y_c, _ = fla_causal_conv1d(x=xt[:, a:b], weight=weight, bias=bias, activation=activation, cu_seqlens=cu_c)
        outs.append(_ContigGrad.apply(y_c))
        i = j
    return torch.cat(outs, dim=1).transpose(1, 2)





def tree_gdn_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    ctx = _current_ctx()
    if ctx is None or cache_params is not None or hidden_states.shape[0] != 1 or ctx.trunk_len >= hidden_states.shape[1]:
        STATS["gdn_orig"] += 1
        # keyword args: llamafactory's replacement has an extra positional slot (cache_position) between
        # cache_params and attention_mask, so a positional call would silently mis-bind attention_mask.
        return _orig_gdn_forward(
            self, hidden_states=hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs
        )
    STATS["gdn_tree"] += 1

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    B, T, _ = hidden_states.shape
    hs = hidden_states
    mixed = self.in_proj_qkv(hs).transpose(1, 2)  # [1, C, T]
    z = self.in_proj_z(hs).reshape(B, T, -1, self.head_v_dim)
    beta = self.in_proj_b(hs).sigmoid()
    g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hs).float() + self.dt_bias)
    weight = self.conv1d.weight.squeeze(1)
    bias = self.conv1d.bias
    W = self.conv_kernel_size
    tr = ctx.trunk_len
    forks = ctx.forks
    starts = ctx.starts
    lens = ctx.lens

    def split_qkv(c: torch.Tensor):
        """[1, C, L] (channel-first conv output) -> q/k/v [1, L, H, D] (contiguous)."""
        q, k, v = torch.split(c.transpose(1, 2), [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(1, -1, self.num_k_heads, self.head_k_dim)
        k = k.reshape(1, -1, self.num_k_heads, self.head_k_dim)
        v = v.reshape(1, -1, self.num_v_heads, self.head_v_dim)
        if self.num_v_heads // self.num_k_heads > 1:
            q = q.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            k = k.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        return q.contiguous(), k.contiguous(), v.contiguous()

    out = torch.zeros((B, T, self.value_dim), dtype=hidden_states.dtype, device=hidden_states.device)

    # ---- trunk: one continuous causal conv, then a chained scan cut at every fork so S(fork_k) is exact -----
    c_trunk = _conv(mixed[:, :, :tr], weight, bias, self.activation)
    q_t, k_t, v_t = split_qkv(c_trunk)
    trunk_cuts = sorted({0, *[f for f in forks if 0 < f < tr], *range(0, tr, MAX_SEG)}) + [tr]
    tr_states: list = [None]  # tr_states[i] = state after consuming trunk[:trunk_cuts[i]]
    state = None
    for i in range(len(trunk_cuts) - 1):
        a_, b_ = trunk_cuts[i], trunk_cuts[i + 1]
        o_i, state = chunk_gated_delta_rule(
            q_t[:, a_:b_], k_t[:, a_:b_], v_t[:, a_:b_], g=g[:, a_:b_].contiguous(), beta=beta[:, a_:b_].contiguous(),
            initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        out[:, a_:b_] = o_i.reshape(1, b_ - a_, -1).to(out.dtype)
        tr_states.append(state)
    # A branch forking at trunk position f sees trunk[:f] -> starts from the state after consuming trunk[:f].
    # (Taking the NEXT cut's state leaks trunk[f:next_fork] into the branch: the 0.06-0.16 nat regression.)
    cut_index = {c: i for i, c in enumerate(trunk_cuts)}
    fork_states = [tr_states[-1] if f >= tr else tr_states[cut_index[f]] for f in forks]

    # ---- branches: ONE packed varlen conv + ONE packed varlen scan (bit-exact vs a per-branch loop, see
    # runs/ablation/scratch/fla_varlen_state.py). Each branch is prepended with the W-1 raw trunk columns
    # before its fork so the depthwise causal conv sees exactly the left context it had inside the trunk;
    # those W-1 positions are then cut off again before the scan. -------------------------------------------
    if lens:
        idx_ext, idx_br, cu_ext, cu_b = ctx.gdn_index(W, hidden_states.device)
        # gather [W-1 trunk columns before fork_b] ++ branch_b for every b in one index_select (a zero column is
        # appended to `mixed` so forks < W-1 can be left-padded by indexing it)
        mixed_z = torch.cat([mixed, mixed.new_zeros((1, mixed.shape[1], 1))], dim=-1)
        packed = mixed_z.index_select(-1, idx_ext)  # [1, C, sum(W-1+len_b)]
        c_br = _conv_varlen(packed, weight, bias, self.activation, cu_ext)  # [1, C, sum(ext)]
        c_br = c_br.index_select(-1, ctx.keep_index)  # drop the W-1 context columns of every segment
        q_b, k_b, v_b = split_qkv(c_br)
        g_b = g.index_select(1, idx_br).contiguous()
        beta_b = beta.index_select(1, idx_br).contiguous()
        S0 = torch.cat([s_ if s_ is not None else q_b.new_zeros((1, self.num_v_heads, self.head_k_dim, self.head_v_dim), dtype=torch.float32)
                        for s_ in fork_states], dim=0)
        o_b, _ = chunk_gated_delta_rule(
            q_b, k_b, v_b, g=g_b, beta=beta_b, initial_state=S0, output_final_state=False,
            use_qk_l2norm_in_kernel=True, cu_seqlens=cu_b,
        )
        out[:, idx_br] = o_b.reshape(1, -1, self.num_v_heads * self.head_v_dim).to(out.dtype)

    core = self.norm(out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    return self.out_proj(core.reshape(B, T, -1))


def tree_attn_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    ctx = _current_ctx()
    if ctx is None or past_key_values is not None:
        STATS["attn_orig"] += 1
        return _orig_attn_forward(self, hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask,
                                  past_key_values=past_key_values, **kwargs)
    STATS["attn_tree"] += 1

    from torch.nn.attention.flex_attention import flex_attention

    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    input_shape = hidden_states.shape[:-1]
    head_dim = self.head_dim
    hidden_shape = (*input_shape, -1, head_dim)
    query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    # liger's RMSNorm can return fp32 for q/k while v stays bf16; flex requires one dtype.
    dt = value_states.dtype
    query_states = query_states.to(dt)
    key_states = key_states.to(dt)
    # flex_attention handles GQA natively (enable_gqa) and returns [B, H, T, D] -- the transformers
    # attention interface returns [B, T, H, D], so transpose back before flattening the head dim.
    # kernel_options: the autotuned default picks BLOCK 64/32 tiles; TREE_FLEX_BLOCK=128 forces 128x128 forward
    # tiles (fewer, fatter tiles on L20X). Set via env so the bench can A/B it without a code change.
    q0 = ctx.q_split
    if q0 > 0:
        # trunk queries [0, q0): causal over the trunk prefix -> FlashAttention-2 (varlen-free, [B, T, H, D])
        from flash_attn import flash_attn_func

        o_tr = flash_attn_func(query_states[:, :, :q0].transpose(1, 2), key_states[:, :, :q0].transpose(1, 2),
                               value_states[:, :, :q0].transpose(1, 2), softmax_scale=self.scaling, causal=True)
        # tail queries [q0, T) (trunk remainder + pads + branches): flex with the tree mask over the full KV
        o_tl = _flex()(query_states[:, :, q0:], key_states, value_states, block_mask=ctx.block_mask, scale=self.scaling,
                       enable_gqa=query_states.shape[1] != key_states.shape[1], kernel_options=_flex_kernel_options())
        attn = torch.cat([o_tr, o_tl.transpose(1, 2)], dim=1).reshape(*input_shape, -1).contiguous()
    else:
        attn = _flex()(query_states, key_states, value_states, block_mask=ctx.block_mask, scale=self.scaling,
                       enable_gqa=query_states.shape[1] != key_states.shape[1], kernel_options=_flex_kernel_options())
        attn = attn.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn * torch.sigmoid(gate).to(attn.dtype)), None


def tree_text_forward(self, *args, **kwargs):
    """Pop the tree metadata, build the context, run the original TextModel forward.

    The context is kept in a module-level slot (not a scoped contextvar) because gradient checkpointing
    re-runs the decoder layers during backward(), i.e. after this function has returned; a scoped context
    would be gone by then and the recompute would silently take the non-tree path. A forward WITHOUT tree
    metadata clears the slot, so reference (arm-A) forwards in the checks never see a stale tree context.
    """
    seg = kwargs.pop("tree_seg", None)
    vis = kwargs.pop("tree_vis_end", None)
    forks = kwargs.pop("tree_forks", None)
    starts = kwargs.pop("tree_branch_starts", None)
    lens = kwargs.pop("tree_branch_lens", None)
    trunk_len = kwargs.pop("tree_trunk_len", None)
    if seg is None:
        _CTX["ctx"] = None
        return _orig_text_forward(self, *args, **kwargs)

    ctx = _Ctx(
        seg=seg[0].to(torch.int32),
        vis_end=vis[0].to(torch.int32),
        forks=[int(x) for x in forks[0].tolist()],
        starts=[int(x) for x in starts[0].tolist()],
        lens=[int(x) for x in lens[0].tolist()],
        trunk_len=int(trunk_len),
    )
    T = int(ctx.seg.numel())
    if ATTN_SPLIT and ctx.lens and ctx.trunk_len < T:
        # flex query range starts at the T_MULT boundary at/below the trunk end (bounded set of shapes, like
        # T_pad); the trunk part before it is plain causal -> FA2.  T_MULT is a multiple of BLOCK.
        mult = max(BLOCK, (SPLIT_MULT // BLOCK) * BLOCK)
        ctx.q_split = (ctx.trunk_len // mult) * mult
    ctx.block_mask = build_block_mask(
        ctx.seg, ctx.vis_end, ctx.forks, ctx.starts, ctx.lens, ctx.trunk_len, ctx.seg.device, q_start=ctx.q_split
    )
    _CTX["ctx"] = ctx
    return _orig_text_forward(self, *args, **kwargs)


def clear_tree_ctx() -> None:
    _CTX["ctx"] = None


class _Ctx:
    __slots__ = ("seg", "vis_end", "forks", "starts", "lens", "trunk_len", "block_mask", "_gdn_idx", "keep_index", "q_split")

    def __init__(self, seg, vis_end, forks, starts, lens, trunk_len):
        self.seg = seg
        self.vis_end = vis_end
        self.forks = forks
        self.starts = starts
        self.lens = lens
        self.trunk_len = trunk_len
        self.block_mask = None
        self._gdn_idx = None
        self.keep_index = None
        self.q_split = 0  # ATTN_SPLIT: flex handles queries [q_split, T); FA2 handles trunk [0, q_split)

    def gdn_index(self, W: int, device):
        """Index tensors shared by all 24 GDN layers of one forward (built once per sample, on the CPU).

        idx_ext : positions of [W-1 trunk columns before fork_b] ++ branch_b, for every branch; a fork < W-1 is
                  left-padded with index T (a zero column appended by the caller)
        idx_br  : positions of the branch tokens only (same order)
        cu_ext / cu_b : cu_seqlens for the extended / plain packing
        keep_index    : positions inside the extended packing that are branch tokens (drops the W-1 columns)
        """
        if self._gdn_idx is None:
            T = int(self.seg.numel())
            ext, br, keep, cu_e, cu_b = [], [], [], [0], [0]
            for f, s, n in zip(self.forks, self.starts, self.lens):
                left = list(range(max(0, f - (W - 1)), f))
                left = [T] * (W - 1 - len(left)) + left
                base = len(ext)
                ext += left + list(range(s, s + n))
                keep += range(base + W - 1, base + W - 1 + n)
                br += range(s, s + n)
                cu_e.append(len(ext))
                cu_b.append(len(br))
            mk = lambda x: torch.tensor(x, dtype=torch.long, device=device)
            self._gdn_idx = (mk(ext), mk(br), mk(cu_e), mk(cu_b))
            self.keep_index = mk(keep)
        return self._gdn_idx


_CTX: dict = {"ctx": None}


def _current_ctx():
    return _CTX["ctx"]


_FLEX = {"fn": None, "compile": None}


def _flex_kernel_options():
    blk = int(os.environ.get("TREE_FLEX_BLOCK", "0"))
    if not blk:
        return None
    return {"BLOCK_M": blk, "BLOCK_N": blk, "BLOCK_M1": blk // 2, "BLOCK_N1": blk, "BLOCK_M2": blk, "BLOCK_N2": blk // 2}


def _flex():
    """flex_attention, torch.compile'd once (the C++ fallback is ~10x slower and needs autotune per shape).

    Every tree row has a different T_pad, so the graph is compiled with a large recompile
    budget (T_pad bucketed by TREE_T_MULT); a silent fall-back to the eager math kernel would materialise [T, T] scores (77 GB at 140K) and
    OOM, so dynamo is told to raise instead of falling back.
    """
    mode = os.environ.get("TREE_FLEX_COMPILE", "1")
    if mode == "0":
        from torch.nn.attention.flex_attention import flex_attention

        return flex_attention
    if _FLEX["fn"] is None:
        import torch._dynamo
        from torch.nn.attention.flex_attention import flex_attention

        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
        if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
            torch._dynamo.config.accumulated_cache_size_limit = max(
                torch._dynamo.config.accumulated_cache_size_limit, 1024
            )
        if hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
            torch._dynamo.config.fail_on_recompile_limit_hit = True
        # static shapes (one graph per distinct T_pad, bounded by TREE_T_MULT bucketing); TREE_FLEX_COMPILE=dynamic
        # opts into symbolic shapes instead.
        _FLEX["fn"] = torch.compile(flex_attention, dynamic=mode == "dynamic", mode="max-autotune-no-cudagraphs")
    return _FLEX["fn"]


_orig_gdn_forward = None
_orig_attn_forward = None
_orig_text_forward = None
_PATCHED = {"done": False}


def apply_tree_patch() -> None:
    global _orig_gdn_forward, _orig_attn_forward, _orig_text_forward
    if _PATCHED["done"]:
        return
    import transformers.models.qwen3_5.modeling_qwen3_5 as M

    _orig_gdn_forward = M.Qwen3_5GatedDeltaNet.forward
    _orig_attn_forward = M.Qwen3_5Attention.forward
    _orig_text_forward = M.Qwen3_5TextModel.forward
    M.Qwen3_5GatedDeltaNet.forward = tree_gdn_forward
    M.Qwen3_5Attention.forward = tree_attn_forward
    M.Qwen3_5TextModel.forward = tree_text_forward
    _PATCHED["done"] = True
    print("[tree] patched Qwen3_5TextModel / Qwen3_5Attention / Qwen3_5GatedDeltaNet", flush=True)


def reassert_tree_patch() -> None:
    """Re-install the tree forwards if something replaced them after apply_tree_patch().

    LlamaFactory's patcher.patch_qwen3_5_forward_gpu runs at model-load time (after our patch) and assigns its
    own cu_seqlens GDN forward to the class -- which silently turned the 24 linear layers back into a plain
    causal scan over the padded [trunk, branch, ...] sequence (dbg step-1 loss 1.609 vs 1.410).
    Whatever forward is found installed becomes the non-tree fallback.
    """
    global _orig_gdn_forward, _orig_attn_forward, _orig_text_forward
    import transformers.models.qwen3_5.modeling_qwen3_5 as M

    assert _PATCHED["done"], "apply_tree_patch() first"
    if M.Qwen3_5GatedDeltaNet.forward is not tree_gdn_forward:
        _orig_gdn_forward = M.Qwen3_5GatedDeltaNet.forward
        M.Qwen3_5GatedDeltaNet.forward = tree_gdn_forward
        print(f"[tree] re-patched Qwen3_5GatedDeltaNet.forward (was {_orig_gdn_forward.__qualname__})", flush=True)
    if M.Qwen3_5Attention.forward is not tree_attn_forward:
        _orig_attn_forward = M.Qwen3_5Attention.forward
        M.Qwen3_5Attention.forward = tree_attn_forward
        print(f"[tree] re-patched Qwen3_5Attention.forward (was {_orig_attn_forward.__qualname__})", flush=True)
    if M.Qwen3_5TextModel.forward is not tree_text_forward:
        _orig_text_forward = M.Qwen3_5TextModel.forward
        M.Qwen3_5TextModel.forward = tree_text_forward
        print(f"[tree] re-patched Qwen3_5TextModel.forward (was {_orig_text_forward.__qualname__})", flush=True)


# -------------------------------------------------------------------------------------- collator
class TreeDataCollator:
    """Cache row (lists) -> model batch with tree metadata; pads to the 128 block grid."""

    def __init__(self, pad_token_id: int = 248044):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        assert len(features) == 1, f"tree collator is batch-size 1 (got {len(features)})"
        row = features[0]
        layout = tree_layout_from_row(row, self.pad_token_id)
        batch = layout.build(
            [int(x) for x in row["input_ids"]],
            [int(x) for x in row["labels"]],
            [int(x) for x in row["position_ids"]],
        )
        return batch