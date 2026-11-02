"""Strict partial transfer: move what is compatible, prove it, and refuse the rest.

Why this exists
---------------
`transfer_style=True` in `common.construct` copies EVERY `style_representation.*` and
`style_injections.*` tensor from the source checkpoint and raises if any key is left
over. That is exactly wrong for the P0 -> S4 migration, where the target deliberately
changes architecture:

    P0   style_matrix_repr = 2 (R2, LocalCNN)   style_matrix_injection = 3 (I3)   47 writers
    S4   style_matrix_repr = 2 (R2, LocalCNN)   style_matrix_injection = 1 (I1)   10 writers

R2 is identical and must be inherited. I1 and I3 share NO parameter names at all --
I1 is `local.0/local.1`, I3 is `down/basis/output/mix` -- so the injection side is a
replacement, not a transfer, and `transfer_style=True` would fail on 18 missing keys.

What this module guarantees
---------------------------
The danger in a migration like this is silence: a typo'd prefix, a renamed module, or a
repr/injection mismatch that quietly starts the run from partially-random weights. So the
contract is invertible and enforced:

  1. Every source style tensor is either TRANSFERRED or explicitly DROPPED, by name.
  2. Every target style tensor is either TRANSFERRED or explicitly FRESH, by name.
  3. Any tensor in neither list raises -- there is no default.
  4. Shape mismatches raise instead of being skipped.
  5. Transferred tensors are byte-compared after loading, not just assumed.

`drop` must be justified as a known architectural difference; it is not a catch-all.
"""
import torch


# Which style_representation prefixes survive the P0 -> S4 migration. R2 is unchanged
# between the two configs, so this is the whole repr side.
REPR_PREFIXES = ('style_representation.local.', 'style_representation.pool.',
                 'style_representation.project.')

# Two distinct embedding tables exist and BOTH depend on num_writers:
#   backbone.writer.weight              the null-augmented writer table the U-Net reads
#   style_representation.table.weight   the R2 projection's writer table
# P0 has 48/48 rows (47 writers + null); S4 has 11/11 (10 writers + null). Neither can be
# copied row-wise -- the rows index different writer sets -- so both are re-seeded.
WRITER_TABLES = ('backbone.writer.weight', 'style_representation.table.weight')


def _style_keys(state):
    return sorted(k for k in state if k.startswith(('style_representation.',
                                                    'style_injections.')))

def plan_transfer(source_state, model, *, expect_repr, expect_injection,
                  writer_table_mode='drop', writer_rows=None):
    """Decide every style tensor's fate. Pure inspection; loads nothing.

    Also classifies the two writer tables, which are NOT style-prefixed on the backbone
    side (`backbone.writer.weight`) and would otherwise be picked up by the plain
    backbone copy and silently mis-shaped. They are reported under `writer_tables`, not
    under the style counts, so the audit table stays honest about which is which.
    """
    target = model.state_dict()
    src_keys = set(_style_keys(source_state))
    tgt_keys = set(_style_keys(target))

    # The target's own parameter names tell us what architecture it really is. Trusting
    # the requested config instead of the built module is how a migration silently
    # mismatches, so derive the families from real tensors.
    has_local = any(k.startswith('style_representation.local.') for k in tgt_keys)
    has_basis = any('.basis.weight' in k for k in tgt_keys)
    if has_local != (expect_repr == 2):
        raise ValueError(f'repr mismatch: model has_local={has_local} but expect_repr={expect_repr}')
    inj_keys = [k for k in tgt_keys if k.startswith('style_injections.')]
    if any('.local.0.weight' in k for k in inj_keys):
        actual_inj = 1
    elif any('.basis.weight' in k for k in inj_keys):
        actual_inj = 3
    else:
        actual_inj = None
    if actual_inj != expect_injection:
        raise ValueError(f'injection mismatch: model looks like I{actual_inj}, '
                         f'expected I{expect_injection}')

    to_load, transferred, dropped, fresh = {}, [], [], []
    reasons = {}

    for k in sorted(src_keys | tgt_keys):
        in_src, in_tgt = k in src_keys, k in tgt_keys
        if k in WRITER_TABLES:
            # num_writers-dependent; handled by _classify_writer_tables and
            # init_writer_tables, never by a shape-checked copy. Dropping it here is not
            # a loss -- the rows denote different writer sets and no row is meaningful
            # to carry across.
            if in_src:
                dropped.append(k)
                reasons[k] = (f'writer table: {source_state[k].shape[0]} rows in source, '
                              f'{target[k].shape[0]} in target; re-seeded from the source '
                              f'row mean (see init_writer_tables)')
            continue
        if in_src and not in_tgt:
            # A real architectural difference is the only acceptable reason to drop.
            if k.startswith('style_injections.'):
                dropped.append(k)
                reasons[k] = ('source injection parameter has no counterpart in the '
                              'target injection kind; I1 (local.0/local.1) and I3 '
                              '(down/basis/output/mix) share no names')
            else:
                raise ValueError(f'source style tensor {k} has no target counterpart and '
                                 f'is not a known architectural difference -- refusing to '
                                 f'drop it silently')
            continue
        if in_tgt and not in_src:
            fresh.append(k)
            reasons[k] = 'new in target architecture (starts from module init)'
            continue
        if source_state[k].shape != target[k].shape:
            raise ValueError(f'{k}: shape {tuple(source_state[k].shape)} -> '
                             f'{tuple(target[k].shape)} cannot be transferred')
        to_load[k] = source_state[k]
        transferred.append(k)

    # Nothing may be unaccounted for. This is the property that makes the migration
    # auditable rather than merely plausible.
    accounted = set(transferred) | set(dropped) | set(fresh)
    if accounted != src_keys | tgt_keys:
        missing = (src_keys | tgt_keys) - accounted
        raise AssertionError(f'unclassified style tensors: {sorted(missing)}')

    report = dict(expect_repr=expect_repr, expect_injection=expect_injection,
                  transferred=transferred, dropped=dropped, fresh=fresh,
                  reasons=reasons,
                  writer_tables=_classify_writer_tables(source_state, target,
                                                        writer_table_mode),
                  counts=dict(transferred=len(transferred), dropped=len(dropped),
                              fresh=len(fresh)))
    return to_load, report


def _classify_writer_tables(source_state, target, mode):
    """Report the two num_writers-dependent tables separately from the style audit."""
    out = {}
    for k in WRITER_TABLES:
        if k not in source_state or k not in target:
            out[k] = dict(present=False)
            continue
        s_rows, t_rows = source_state[k].shape[0], target[k].shape[0]
        out[k] = dict(present=True, source_rows=s_rows, target_rows=t_rows,
                      same_shape=source_state[k].shape == target[k].shape,
                      action=('transfer' if (mode == 'keep'
                                             and source_state[k].shape == target[k].shape)
                              else 'reinit'))
    return out


def apply_transfer(model, source_state, report, to_load):
    """Load the plan, then PROVE the transferred tensors arrived unchanged.

    `strict=False` on a filtered dict is unavoidable (we are deliberately loading a
    subset), which is precisely why the equality check below is not optional: without it
    a key-name typo or a silently-skipped entry looks identical to success.
    """
    result = model.load_state_dict(to_load, strict=False)

    residue = [k for k in result.missing_keys if k in set(report['fresh'])]
    if len(residue) != len(report['fresh']):
        unexpected_missing = [k for k in result.missing_keys if k not in set(report['fresh'])]
        raise AssertionError(f'unexpected missing keys after partial load: {unexpected_missing}')
    if result.unexpected_keys:
        raise AssertionError(f'unexpected keys after partial load: {result.unexpected_keys}')

    after = model.state_dict()
    worst, worst_key = 0.0, None
    for k in report['transferred']:
        a, b = after[k].float(), to_load[k].float()
        if a.shape != b.shape:
            raise AssertionError(f'{k}: shape changed during load')
        d = float((a - b).abs().max())
        if d > worst:
            worst, worst_key = d, k
    if worst != 0.0:
        raise AssertionError(f'transferred tensor {worst_key} changed by {worst:.3e} '
                             f'during load; transfer is not exact')
    return dict(verified=len(report['transferred']), max_diff=worst, worst_key=worst_key)


def load_backbone(model, source_state, report, *, exclude=()):
    """Copy P0's trained weights for everything outside the changed namespaces.

    This is the other half of the migration and the half that carries the value: the
    plan's whole premise is that P0's generic content prior survives. Style tensors are
    excluded because `plan_transfer` owns them, and the writer tables are excluded
    because `init_writer_tables` owns them.

    Like `apply_transfer`, the equality check afterwards is not optional -- a filtered
    `load_state_dict(strict=False)` reports nothing about tensors it did not receive, so
    a missed prefix would look exactly like success.
    """
    style = set(report['transferred']) | set(report['dropped']) | set(report['fresh'])
    skip = style | set(WRITER_TABLES) | set(exclude)
    after = model.state_dict()

    load, mismatched = {}, []
    for k, v in source_state.items():
        if k in skip or k not in after:
            continue
        if after[k].shape != v.shape:
            mismatched.append((k, tuple(v.shape), tuple(after[k].shape)))
            continue
        load[k] = v
    if mismatched:
        raise ValueError(f'backbone shape mismatches: {mismatched}')

    result = model.load_state_dict(load, strict=False)
    unexpected = [k for k in result.unexpected_keys if k not in skip]
    # missing_keys here is expected to include the style/writer tensors we deliberately
    # withheld, so only complain about ones we intended to supply.
    not_supplied = [k for k in load if k not in model.state_dict()]
    if unexpected or not_supplied:
        raise AssertionError(f'backbone load anomalies: unexpected={unexpected} '
                             f'not_supplied={not_supplied}')

    after = model.state_dict()
    worst, worst_key, identical = 0.0, None, 0
    for k in load:
        d = float((after[k].float() - load[k].float()).abs().max())
        if d > worst:
            worst, worst_key = d, k
        if d == 0.0:
            identical += 1
    if worst != 0.0:
        raise AssertionError(f'backbone tensor {worst_key} changed by {worst:.3e} during load')
    return dict(loaded=len(load), identical=identical, max_diff=worst,
                worst_key=worst_key, skipped=len(skip))


def init_writer_tables(model, source_state, *, jitter=0.01, seed=0):
    """Seed BOTH writer tables from the mean of P0's trained rows.

    S4 is a Top10 study while P0 was pretrained on 47 writers, so no row can be copied
    directly even where shapes allowed. The mean of the trained table is a far better
    starting point than random init: it already sits in the right feature space and
    already encodes "a generic trained writer". The small jitter stops the rows from
    being exactly identical, which would make their gradients identical on step 1 and
    leave the writer dimension momentarily degenerate.

    The last row of each table is the null slot (num_writers + 1); it is seeded the same
    way, which preserves the "null means generic" reading P0 ended with.
    """
    after = model.state_dict()
    out = {}
    g = torch.Generator(device='cpu').manual_seed(seed)
    for k in WRITER_TABLES:
        if k not in source_state or k not in after:
            out[k] = dict(present=False)
            continue
        src, tgt = source_state[k].float(), after[k]
        n = tgt.shape[0]
        if src.shape[1] != tgt.shape[1]:
            raise ValueError(f'{k}: feature dim {src.shape[1]} -> {tgt.shape[1]} cannot '
                             f'be re-seeded from the source mean')
        mean_row = src.mean(0, keepdim=True)
        noise = torch.randn(n, src.shape[1], generator=g) * float(jitter)
        new = (mean_row + noise).to(tgt.dtype)
        with torch.no_grad():
            tgt.copy_(new)
        out[k] = dict(present=True, rows=n, source_rows=src.shape[0], jitter=jitter,
                      row_spread=float(new.float().std(0).mean()),
                      source_mean=float(src.mean()), new_mean=float(new.float().mean()))
    return out
