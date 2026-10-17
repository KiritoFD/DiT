import copy
from tools.analysis.prepare_v89_crosschar_pilot import teacher_gate,build_configs


def test_high_classification_cannot_bypass_failed_matched_discrimination():
    matched={'models':{'v3':{k:{'n':1000,'accuracy':.45,'ci95':[.4,.5]}
                            for k in ('test_char','test_source')}}}
    accuracy={'full_holdout_exported_prototypes':{k:{'prototype_top1':.99}
                                                 for k in ('test_char','test_source')}}
    assert len(teacher_gate(matched,accuracy))==2
    good=copy.deepcopy(matched)
    for value in good['models']['v3'].values():
        value.update(accuracy=.7,ci95=[.6,.8])
    assert not teacher_gate(good,accuracy)
    good['models']['v3']['test_char']['accuracy']=float('nan')
    assert teacher_gate(good,accuracy)


def test_pilot_only_changes_training_loss_not_generator_architecture_or_inputs():
    base={'content_cond':'std','std_soft_cond':True,'cond_inject_at':'2,4,5,6',
          'freeze_triple_tables':True,'std_soft_sigma_min':.4,'std_soft_sigma_max':1.,'w_repa':.03}
    arms=build_configs(base,'teacher.pt','bank.pt','baseline.pt')
    for cfg in arms.values():
        for key in base:
            assert cfg[key]==base[key]
        assert cfg['writer_glyph_rank']==0 and cfg['style_ada_rank']==0
        assert not any('prototype' in key or 'reference_style' in key for key in cfg)
    assert arms['control']['crosschar_gradient_ratio']==0
    assert arms['crosschar']['crosschar_gradient_ratio']==.2
