from tools.analysis.prepare_style_gradient_bank import select_characters
import ast
from pathlib import Path


def test_expanded_bank_has_distinct_supported_characters_and_fixed_selection():
    rows=[{'calligrapher':w,'script':'s','character':str(c)} for w in ('A','B') for c in range(12)]
    rows.append({'calligrapher':'unsupported','script':'s','character':'bad'})
    slots=[('A','s'),('B','s')]
    selected=select_characters(rows,slots,10)
    assert selected==select_characters(rows,slots,10)
    assert len({rows[i]['character'] for i in selected})==10
    assert {rows[i]['calligrapher'] for i in selected}=={'A','B'}
    assert all((rows[i]['calligrapher'],rows[i]['script']) in slots for i in selected)


def test_output_path_is_resolved_before_canonical_chdir():
    # Structural regression guard: the GPU-backed integration smoke verifies I/O.
    tree=ast.parse(Path('tools/analysis/prepare_style_gradient_bank.py').read_text(encoding='utf-8'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    out_assign=next(n for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='out' for t in n.targets))
    assert isinstance(out_assign.value,ast.Call) and isinstance(out_assign.value.func,ast.Attribute)
    assert out_assign.value.func.attr=='resolve'
    change=next(n for n in main.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='chdir')
    assert main.body.index(out_assign)<main.body.index(change)
