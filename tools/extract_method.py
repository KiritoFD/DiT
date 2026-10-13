# -*- coding: utf-8 -*-
"""PR3 **通用抽取脚本** —— 把一个 `self.X` 方法搬成模块级函数 + 配置 dataclass + 薄封装。

用法: python scratch/pr3_generic.py ClassName.method_name [ConfigName] [func_name]

它把已手工做过 4 次的机械变换**参数化** ✓（并吸收全部已知陷阱 ✓）:
  ① 语句体按 AST 取（**不碰跨行签名** ✓ —— 我 4 次脚本失败的根因 ✗）
  ② `getattr(self, "X", DEFAULT)` 模式: 从 AST **读出 DEFAULT 字面量** ✓
     ⇒ 生成 `getattr(m, "X", DEFAULT)` ✓（默认值必须与原默认一致 ✗）
  ③ 其余 `self.X` -> `cfg.X`（字段名 = 原名 ✓，不擅自去下划线 ✗）
  ④ 语句体缩进 8 -> 4（模块级函数 ✓）；**插入与改委托一次完成** ✓（round 39 教训 ✓）
  ⑤ 若语句体里出现**裸 self**（替换不掉的）⇒ 直接**报错退出** ✗（不产出半成品 ✓）
"""
from __future__ import annotations

import ast
import io
import sys

P = 'src/model/dit.py'


def main() -> int:
    spec = sys.argv[1]
    cls_name, meth = spec.split('.', 1)
    cfg_name = sys.argv[2] if len(sys.argv) > 2 else (meth.lstrip('_').title().replace('_', '') + 'Cfg')
    func_name = sys.argv[3] if len(sys.argv) > 3 else meth.lstrip('_')

    src = io.open(P, encoding='utf-8').read()
    L = src.split('\n')
    t = ast.parse(src)
    cls = next(n for n in t.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == meth)
    b0, b1 = fn.body[0].lineno, fn.body[-1].end_lineno
    sig_src = L[fn.lineno - 1:fn.body[0].lineno - 1]
    print('  %s: def L%d..L%d | 语句体 L%d..%d（%d 行）' % (spec, fn.lineno, fn.end_lineno, b0, b1, b1 - b0 + 1))

    body = '\n'.join(L[b0 - 1:b1])
    # ② 读出 getattr(self, "X", DEFAULT) 的默认值（字面量 ✓）
    defaults = {}
    for x in ast.walk(ast.parse('if True:\n' + body)):
        if (isinstance(x, ast.Call) and getattr(x.func, 'id', '') == 'getattr' and len(x.args) >= 3
                and isinstance(x.args[0], ast.Name) and x.args[0].id == 'self'
                and isinstance(x.args[1], ast.Constant)):
            key = x.args[1].value
            try:
                defaults[key] = ast.literal_eval(x.args[2])
            except Exception:                                    # noqa: BLE001
                print('    ⚠ 无法解析默认值: %s ✗' % ast.unparse(x))
    if defaults:
        print('    陷阱默认值（从 AST 读出 ✓）: %s' % defaults)
    # ③' ★ 写入陷阱（第 4 类 ✓，2026-10-13 实测踩到 ✗）:
    #    `self.X = ...`（**store**）不能变成 `cfg.X = ...` ✗（cfg 是 frozen ✓）
    #    它本质是"改模型的属性" ⇒ 生成 `setattr(cfg.owner, "X", ...)` ✓✓（语义等价 ✓）
    stores = sorted({tg.attr for x in ast.walk(ast.parse('if True:\n' + body))
                     if isinstance(x, ast.Assign)
                     for tg in x.targets
                     if isinstance(tg, ast.Attribute) and isinstance(tg.value, ast.Name)
                     and tg.value.id == 'self'})
    if stores:
        print('    写入陷阱（self.X = …）: %s' % stores)
    for s in stores:
        for lit in ('None', 'True', 'False'):
            body = body.replace('self.%s = %s' % (s, lit), 'setattr(cfg.owner, "%s", %s)' % (s, lit))
    # 若仍有 `self.X =` 形式的写入未处理 ⇒ **中止**（不产出半成品 ✓）
    for x in ast.walk(ast.parse('if True:\n' + body)):
        if isinstance(x, ast.Assign) and any(
                isinstance(tg, ast.Attribute) and isinstance(tg.value, ast.Name) and tg.value.id == 'self'
                for tg in x.targets):
            print('    ✗ 有未处理的 self.X = 写入（行 %d）⇒ **中止，不写文件** ✓' % x.lineno)
            return 2
    # ③ 替换
    for k in list(defaults):
        for pat in ('getattr(self, "%s", %r)' % (k, defaults[k]), "getattr(self, '%s', %r)" % (k, defaults[k])):
            body = body.replace(pat, 'cfg.%s' % k)
    names = sorted({x.attr for x in ast.walk(ast.parse('if True:\n' + body))
                    if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id == 'self'})
    for n in names:
        body = body.replace('self.' + n, 'cfg.' + n)
    # ⑤ 裸 self 检查（决定性 ✓ —— 不产出半成品 ✗）
    left = [x.lineno for x in ast.walk(ast.parse('if True:\n' + body))
            if isinstance(x, ast.Name) and x.id == 'self']
    if left:
        print('    ✗ 仍有裸 self（行 %s）⇒ **中止，不写文件** ✓' % left)
        return 2
    body = '\n'.join((l[4:] if l.startswith('    ') else l) for l in body.split('\n'))

    fields = sorted(set(names) | set(defaults) | ({'owner'} if stores else set()))
    flds = ['    owner: Any             # ★ 模型自身（用于 self.X = … 的写入 ✓）' if n == 'owner'
            else '    %s: Any' % n for n in fields]
    froms = []
    for n in fields:
        if n == 'owner':
            froms.append('                   owner=m,')
        elif n in defaults:
            froms.append('                   %s=getattr(m, "%s", %r),' % (n, n, defaults[n]))
        else:
            froms.append('                   %s=getattr(m, "%s", None),' % (n, n))
    helper = ('@dataclass(frozen=True)\nclass %s:\n'
              '    """`%s` 用到的**全部** `self.*`（2026-10-13 [审查 PR3] 通用抽取 ✓）。\n\n'
              '    审查: "只移动逻辑, 不移动权重注册位置" ✓ ⇒ 不注册模块 ⇒ **state_dict 键不变** ✓✓。\n'
              '    ⚠ 全部用 getattr 兜底 ✓（防**惰性访问** ✗）；陷阱项默认值与原文一致 ✓。\n'
              '    """\n%s\n\n    @classmethod\n    def from_model(cls, m):\n        return cls(\n%s\n        )\n\n\n'
              'def %s(cfg, %s):\n'
              '    """**逐行等价**于 `%s.%s` ✓（[审查 PR3] 通用抽取 ✓）。"""\n'
              % (cfg_name, meth, '\n'.join(flds), '\n'.join(froms), func_name,
                 ', '.join(a.arg for a in fn.args.args if a.arg != 'self'), cls_name, meth))
    # 委托（按原参数顺序 ✓，全用关键字以免顺序错 ✗）
    kw = ', '.join('%s=%s' % (a.arg, a.arg) for a in fn.args.args if a.arg != 'self')
    delegate = ('        """薄封装 ✓ —— 实现已抽成模块级 `%s`（[审查 PR3] ✓）。"""\n'
                '        return %s(%s.from_model(self), %s)'
                % (func_name, func_name, cfg_name, kw)).split('\n')

    cls_lines = L[cls.lineno - 1:]
    new_cls = cls_lines[:fn.lineno - cls.lineno] + sig_src + delegate + cls_lines[(b1 - cls.lineno) + 1:]
    out = '\n'.join(L[:cls.lineno - 1]) + '\n' + helper + body + '\n\n\n' + '\n'.join(new_cls)
    io.open(P, 'w', encoding='utf-8', newline='').write(out)
    import py_compile
    py_compile.compile(P, doraise=True)
    print('  ✓ 完成: %s + %s + 薄封装（文件 %d 行）'
          % (cfg_name, func_name, len(out.split('\n'))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
