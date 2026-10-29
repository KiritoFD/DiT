"""Build a transparent multi-axis report from compact cross-machine exports."""
import argparse
import gzip
import json
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np


def paired(a,b,field='top1'):
    if a['rows']!=b['rows']:return '条件清单不同，不计算配对区间'
    p=a.get('generation_protocol',{});q=b.get('generation_protocol',{})
    if any(p.get(k)!=q.get(k) for k in ['cfg','seed','sampler','num_steps','gamma']):return '采样协议不同，不计算配对区间'
    groups={}
    for row,x,y in zip(a['rows'],a.get('E-G',[]),b.get('E-G',[])):
        if not x['valid'] or not y['valid']:continue
        groups.setdefault(row['calligrapher'],{}).setdefault(row['character'],[]).append(x[field]-y[field])
    if not groups:return '无有效配对'
    values=[np.array([np.mean(v) for v in chars.values()]) for chars in groups.values()]
    rng=np.random.default_rng(90);boot=np.zeros(2000)
    for v in values:boot+=v[rng.integers(len(v),size=(len(boot),len(v)))].mean(1)/len(values)
    scale=100 if field=='top1' else 1
    return f"{np.mean([v.mean() for v in values])*scale:+.2f} [{np.quantile(boot,.025)*scale:+.2f}, {np.quantile(boot,.975)*scale:+.2f}]"


def mm(score,key='E-G'):
    result=score.get(key+'_style_distribution',{});by={}
    for name,s in result.get('slots',{}).items():by.setdefault(name.split('|')[0],[]).append(s['mmd2'])
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None


def fmt(v,digits=4):return '—' if v is None else f'{v:.{digits}f}'


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='docs/1029/eval_unpaired');a=p.parse_args();root=Path(a.data)
    bundles=[]
    for file in root.glob('*.json.gz'):bundles+=json.loads(gzip.decompress(file.read_bytes()))
    models={};decisions={};clean={};completion=[]
    for bundle in bundles:
        completion.append((bundle['root'],bundle['complete'],len(bundle['summary'])))
        for r in bundle['summary']:
            j=r['job'];key=(j['arm'],j['step'],j['protocol'])
            if 'clean_fid' in r:clean[key]=r;continue
            row=models.setdefault(key,dict(job=j,styles=None,quality={}))
            if 'paired_content' in r:row['paired_content']=r['paired_content']
            if row['styles'] is None or 'rescored' in bundle['root']:row['styles']=r
            for metric in ['musiq-ava','musiq','topiq_nr']:
                if metric in r['g']:row['quality'][metric]=r['g'][metric]
        for d in bundle['decisions']:
            key=(d['arm'],d['step'],d['protocol'])
            if key not in decisions or 'rescored' in bundle['root']:decisions[key]=d
    def pick(arm,protocol=None):
        values=[(k,v) for k,v in models.items() if k[0]==arm and (protocol is None or k[2]==protocol)]
        return sorted(values)[-1] if values else (None,None)
    def style_line(arm,protocol=None):
        key,row=pick(arm,protocol)
        if row is None:return f'| {arm} | 待完成 | — | — | — | — | — | — |'
        s=row['styles']['g'];e=s['E-G_retrieval_K64'];local=s.get('E-L_retrieval_K64',{})
        return f"| {arm}@{key[1]//1000}k | {e['n']}/{row['job']['n']} | {fmt(e['macro_top1']*100,2)} | {fmt(e['margin'])} | {fmt(e['afc']*100,2)} | {fmt(mm(s))} | {fmt(local.get('macro_top1')*100 if local.get('macro_top1') is not None else None,2)} | {fmt(mm(s,'E-L'))} |"
    header='| 模型 | 有效N | E-G Macro%↑ | Margin↑ | Bank-2AFC%↑ | E-G MMD²↓ | E-L Macro%↑ | E-L MMD²↓ |\n|---|---:|---:|---:|---:|---:|---:|---:|'
    lines=['# 新评估网络与两阶段重评：eval_unpaired_v1','',
      '快照时间：'+datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')+'（北京时间）。', '',
      '数据与结论按本目录紧凑导出冻结；网络均冻结，未训练新生成器。全部指标独立报告，不合成总分。',
      '', '## 核心结论','',
      '1. **通用IQA校准失败，不能据此宣布书法美学胜者。** MUSIQ-AVA偏好横带擦除，TOPIQ对空白图异常偏好；TTF普遍得到更高技术质量评分。',
      '2. **SSIM优势不等于所有风格指标优势。** 同100字、K64异字检索中S5/S4高于S0，但另一套D1特征并不支持相同排序。',
      '3. **J组没有出现稳定的新增条件收益。** J2/J3的MMD有描述性改善，Writer检索未同步提高；J4–J7仍接近。',
      '4. **P1→P2有跨字检索收益。** 历史权重的原生采样、数据和初始化不同，只作参照，不作S/J架构因果比较。',
      '5. OCR权重缺失、条件多样性缺多seed。当前不生成每模型6000张，不以无效评价器决定新训练。',
      '', '## 1. 校准先于排名','',
      '24张固定训练真迹，比较原图/高斯模糊/5px增粗/擦除中央8px横带/纯白。表中为原图得分更高的比例。擦除操作见下图，不能将摄影领域的分数直接称为书法美学。','',
      '| 评价器 | 原图>模糊 | 原图>增粗 | 原图>擦除 | 原图>纯白 | 原图>同字TTF |',
      '|---|---:|---:|---:|---:|---:|']
    cal=next(b['calibration'] for b in bundles if b['root'].endswith('/4090') and b['calibration'])
    ttf=json.loads((root/'ttf_controls.json').read_text(encoding='utf-8'))['metrics']
    for name,gate in cal['gates'].items():
        v=gate['real_preferred_fraction'];lines.append('| '+name+' | '+' | '.join(f'{100*v[k]:.1f}%' for k in ['blur','thick','broken','blank'])+f" | {100*ttf[name]['real_preferred']:.1f}% |")
    lines+=['','![评价器反例与TTF对照](calibration.jpg)','',
      'MUSIQ-KONIQ可以响应模糊，不等于能评价书风；TTF更高并非单独的失效证明，擦除/纯白反例与人工校验缺失共同限制主排名。以上比例只基于24张，不是总体准确率。',
      '', '## 2. S系列：同100字原过程面板（S10未训练）','',header]
    for arm in ['S0','S1','S2','S3','S4','S5','S6','S7','S8','S9','S11']:
        lines.append(style_line(arm,'original_protocol' if arm in ['S9','S11'] else None))
    lines+=['','E-G为严格加载的ResNet Writer特征；E-L为历史跨字Pixel D1的多尺度统计特征。K=64，每个书家用异字参考图，top5相似度均值竞争，同书体比较。有效N不足100/187时显式排除参考不足槽位，所有模型使用同一规则。',
      '', 'E-G真实GT检索上限约80.75%（100字）、68.25%（187字）；E-L分别约63.0%、44.50%。E-L在187字上的可靠性尤其有限。原有原型分类Top-1、这里的kNN检索与历史困难三元组2AFC是不同指标，不得混称。',
      '', 'MMD按writer先对书体取平均、再对writer宏平均；报告不沿用早期原型的slot宏平均。参考来自生成器训练真迹，排除本面板目标字符，因此明确属于**Train-reference Style MMD**。字符分布、作品来源仍有混杂，点估计更低不等于显著改善。',
      '', '### 新速度分支：统一187字seed0','',header]
    for arm in ['S9','S11']:lines.append(style_line(arm,'heldout187_seed0'))
    lines+=['','S11在100字上SSIM0.59396、LPIPS0.35151；187字SSIM0.60237与S9接近，书家检索没有改善。独立Writer分支目前不能证明优于S0；其容量、条件路径与优化均改变，不归因为单一理论机制错误。']
    lines+=['','## 3. J0–J3：旧187字同协议','',header]
    for arm in ['J0','J1','J2','J3']:lines.append(style_line(arm,'heldout187'))
    lines+=['','## 4. J4–J7：新统一187字seed0','',header]
    for arm in ['J4','J5','J6','J7']:lines.append(style_line(arm,'heldout187_seed0'))
    lines+=['','J4–J7另有original_protocol原生噪声重评，保留在压缩JSON；不把这两种seed取各自最好值。旧J0–J3与新J4–J7分表，避免把噪声变化归因于Decoder结构。',
      '', '## 5. 历史强基线：只比较各自原生面板','',header]
    for arm in ['P1','P2','DeepSym4','U100Base']:
        for key,row in sorted(models.items()):
            if key[0]==arm:
                # Deep75 and85 share the arm name but have distinct checkpoint endpoints.
                s=row['styles']['g'];e=s['E-G_retrieval_K64'];local=s.get('E-L_retrieval_K64',{})
                lines.append(f"| {arm}@{key[1]//1000}k | {e['n']}/{row['job']['n']} | {fmt(e['macro_top1']*100,2)} | {fmt(e['margin'])} | {fmt(e['afc']*100,2)} | {fmt(mm(s))} | {fmt(local.get('macro_top1')*100 if local.get('macro_top1') is not None else None,2)} | {fmt(mm(s,'E-L'))} |")
    lines+=['','历史样本逐张RGB GT像素匹配到已审计CSV，匹配失败不纳入。旧protocol.json保留EMA、采样器、seed、STD sigma；不能把旧原生STD与新模型的绝对差解释为架构效应。',
      '', '## 6. 配对不确定性','',
      '固定这10位书家，在每位书家内部按字符重采样，重复noise按同字符聚类。下表是E-G Macro Top-1差值的95%区间，单位百分点；单seed筛选区间不含训练seed不确定性。',
      '', '| 对照（前者−后者） | Δ Macro Top-1 [95%CI] |','|---|---:|']
    pairs=[('S4','S0',None),('S5','S0',None),('S9','S0','100'),('S11','S0','100'),('S11','S9','heldout187_seed0'),('S8','S4',None),('J2','J0','heldout187'),('J3','J0','heldout187'),('J5','J4','heldout187_seed0'),('J6','J5','heldout187_seed0'),('J7','J6','heldout187_seed0'),('P2','P1',None)]
    for x,y,protocol in pairs:
        def key_for(arm):
            return next((k for k,v in models.items() if k[0]==arm and (v['job']['n']==100 if protocol=='100' else protocol is None or k[2]==protocol)),None)
        xk,yk=key_for(x),key_for(y)
        if xk in decisions and yk in decisions:lines.append(f'| {x}−{y} | {paired(decisions[xk],decisions[yk])} |')
    lines+=['','这些区间为探索性、未校正多重比较；特别是接近0的边界不能宣告机制成立。跨机器比较条件字段相同，但仍应补独立noise/训练seed复核。',
      '', '### K-shot稳定性（有效样本数同时报告）','',
      '| 模型 | K32 Macro% / N | K64 Macro% / N | K128 Macro% / N |','|---|---:|---:|---:|']
    for arm,protocol in [('S0',None),('S4',None),('S5',None),('S9','original_protocol'),('J4','heldout187_seed0'),('J7','heldout187_seed0')]:
        _,r=pick(arm,protocol)
        if r:
            values=[r['styles']['g'][f'E-G_retrieval_K{k}'] for k in [32,64,128]]
            lines.append('| '+arm+' | '+' | '.join(f"{100*v['macro_top1']:.2f} / {v['n']}" if v.get('macro_top1') is not None else '不可算' for v in values)+' |')
    lines+=['','K变化可能改变可评槽位和N；正式稳定性结论应再限制到三K共同有效样本，不拿不同分母直接宣称性能变化。',
      '', '## 7. 无参考质量与非配对分布诊断','',
      '下面保留实测分数以便复核，**不形成美学排行榜**。FID/KID使用官方Clean-FID缩放和Inception；FID用等价低秩SVD计算完整经验协方差距离，已与标准协方差公式核对，无PCA。小样本偏差仍在。',
      '', '| 模型 / 面板 | AVA | TOPIQ | KONIQ | pooled FID | KID | SSIM↑ | LPIPS↓ |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,row in sorted(models.items()):
        q=row['quality'];c=clean.get(key,{})
        paired_content=row.get('paired_content',{})
        lines.append(f"| {key[0]}@{key[1]//1000}k / {key[2]} | {fmt(q.get('musiq-ava',{}).get('mean'))} | {fmt(q.get('topiq_nr',{}).get('mean'))} | {fmt(q.get('musiq',{}).get('mean'),2)} | {fmt(c.get('clean_fid'),2)} | {fmt(c.get('kid'),5)} | {fmt(paired_content.get('ssim'))} | {fmt(paired_content.get('lpips'))} |")
    lines+=['','Real-vs-Real分割按writer×script分层；另记录相同样本数下Gen-vs-Real对照，详见JSON的matched_n/matched_gen_fid。不能拿100-vs-100生成FID与50-vs-50 Real分割直接判高低。不计算每书家十几张的FID；KID负估计保留，不截断为0。',
      '', '## 8. 当前不能回答的问题','',
      '- 审美/自然度：通用IQA未通过任务校准，150–300组人工盲评尚未完成。当前没有书法审美金标准。',
      '- 内容：两机未找到可严格加载并验证字表的OCR权重，OCR为缺失，不是0%。已修复旧评价器“缺权重仍输出随机网络分数”的危险行为。',
      '- 风格泛化：KNN排除同字符，跨dataset-source控制可算；dataset标签不等于作品来源。现有参考库来自训练集，不假装是独立作品留出。',
      '- 多样性：尚未形成统一同条件三seed面板；部分模型的原生噪声与新固定噪声两组结果不能直接代替该协议。feature Coverage已作诊断，但依赖每槽生成数和参考数，低值不直接证明坍缩。',
      '- E-L严格权重仅在4090本地找到；48没有可核验的同源权重，按数据规则没有跨机搬模型。E-L只作现有子集的第二意见，不能据此对所有模型排名。',
      '- S10未训练，不虚构结果。S11已完成50k与新指标；一次性任务只接续评测，不恢复旧训练队列。',
      '- 严格未见字应排除整个预训练链见过的字符，不能仅排除最后一次Top10微调CSV；历史P1来自100k预训练，需要单独审计。',
      '', '## 9. 执行与证据','',
      '4090完成IQA三网、E-G/E-L与Clean-FID/KID；48先在CPU以低优先级重评，S11原训练完成后使用GPU完成S11与历史强基线的末段评测，未恢复S9-Pair/S10/H0/H1。新工具不修改生成器、VAE或旧checkpoint。',
      '', '| 结果根 | 完成 | 结果数 |','|---|---|---:|']
    for name,done,n in completion:lines.append(f'| `{name}` | {done} | {n} |')
    lines+=['','压缩JSON包含逐query判定、K32/64/128、GT ceiling、分writer/script MMD、标签交换负控、Coverage及原始质量分数。`—`表示该快照未完成或资产缺失，不是零。',
      '', '实现：[工具说明](../../../tools/eval_unpaired/README.md)。模型版本与语义参照：[IQA-PyTorch官方ModelCard](https://github.com/chaofengc/IQA-PyTorch/blob/main/docs/ModelCard.md)；分布特征实现：[Clean-FID官方仓库](https://github.com/GaParmar/clean-fid)。',
      '', '下一步先用人工盲评和真正的留出跨字书家面板验证评价器，不依据摄影美学分数继续扩展生成架构。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPORT',root/'REPORT.md','models',len(models))


if __name__=='__main__':main()
