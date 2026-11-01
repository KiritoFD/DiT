"""Finite 48-side Day 1 audit chain, every transition requires actual artifacts."""
import sys


def run(queue):
    root=queue.root
    for name,module,artifact in [('F1_complete','representation','style_teacher_audit.json'),
        ('generation','generation','bank/latents.pt'),('F2_complete','gradient','gradient_transfer.json'),
        ('F3_F4','controls','content_attribution.json')]:
        queue.command(name,[sys.executable,'-u','-m','tools.three_day.audits.'+module,'--root',str(root)],
                      required=[root/artifact])
    queue.status.update(state='day1_audit_complete');queue.flush()
