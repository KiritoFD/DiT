# Focused S0 / S4 / S5 maturity comparison

Three existing50k checkpoints resume to85k, with35,000 extra updates and17,920,000
extra sample exposures each. S0/S4 remain local to4090, S5 remains local to48.
No checkpoint transfer, fresh architecture, new loss or Decoder training.

Prepare locally on each machine:

```sh
python -m tools.style_matrix.focus.plan --machine 4090 --root assets/style_focus_85k_20261029/4090
python -m tools.style_matrix.runtime.serial --plan assets/style_focus_85k_20261029/4090/plan.json --execute --resident-pid 2107710
```

Use machine48 and omit resident-pid on48. The existing serial runner provides GPU
locking, source sealing, atomic checkpoint completion validation, explicit resume,
25GiB minimum free-space check, fail-fast status and a24h wall limit. Restart the same
command after inspection; it resumes the latest completed checkpoint rather than
resetting the experiment. Do not resume the old generation2/later controllers.

Each run restores raw weights, optimizer, EMA and RNG from its own50k checkpoint.
The common LR restart is anchored at50k:35k horizon,200-step warmup, peak scale0.5
of the original6e-5, cosine floor3e-6. The first restored update retains the old6e-6
LR; the existing scheduler then enters its common warmup. Resume does not restart
the warmup. All other training/data/STD/VAE settings are inherited, including the
actual proven microbatch; compile remains enabled and checkpointing remains off.

Save55/60/65/70/75/80/85k. Final evaluations retain native val100, common187×3
shared-noise seeds and a separate training shared-character diagnostic panel.
Do not use that diagnostic panel as evidence of unseen-character style transfer.
Assess S0 as content control, S4 as local-fusion control and S5 as style candidate.
S5++, Writer velocity branches and conditioned Decoder extensions remain off.
