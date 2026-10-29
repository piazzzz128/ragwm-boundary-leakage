"""CPU-only verification of this public subset; no hosted-model replay."""
import collections
import csv
import hashlib
import json
import pathlib

HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def read(name):return json.loads((HERE/name).read_text('utf-8'))
def rows(name):
    with (HERE/name).open(encoding='utf-8',newline='')as f:return list(csv.DictReader(f))


def main():
    manifest=read('PUBLICATION_MANIFEST.json');checks=0
    for name,expected in manifest['public_files_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
        checks+=1
    result=read('body_context_boundary/RESULTS.json')
    rr=rows('body_context_boundary/ownership_by_unit.csv')
    assert len(rr)==320 and len({(x['unit_index'],x['condition']) for x in rr})==320
    for condition,w in result['WSN']['conditions'].items():
        group=[int(x['flag']) for x in rr if x['condition']==condition]
        assert len(group)==40 and set(group)<={0,1,2}
        yes,no,unknown=[group.count(i) for i in [1,0,2]]
        assert (yes,no,unknown)==(w['YES'],w['NO'],w['Unknown'])
        assert w['WSN_Unknown_bounds']==[yes,yes+unknown]
        checks+=40
    assert result['low_damage_localization_proxy_joint_pass'] is False
    assert result['general_QA_or_retrieval_utility_validated'] is False
    cc=rows('body_context_boundary/localization_by_copy.csv')
    assert len(cc)==453
    for method,b in result['boundary']['boundary'].items():
        group=[x for x in cc if x['method']==method]
        assert len(group)==151 and len({x['copy_index'] for x in group})==151
        covered=sum(int(x['covered']) for x in group)
        accepted=sum(int(x['accepted_true_start']) for x in group)
        first=sum(int(x['first_action_true_start']) for x in group)
        assert (covered,accepted,first)==(b['covered'],b['TP'],b['first_action_TP'])
        assert (151-covered,151-accepted)==(b['uncovered_FN'],b['FN_including_uncovered'])
        for x in group:
            assert int(x['first_action_true_start'])<=int(x['accepted_true_start'])<=int(x['covered'])
        checks+=151
    damage=rows('body_context_boundary/damage_by_world.csv')
    clean=rows('body_context_boundary/independent_clean_by_document.csv')
    decomposition=read('body_context_boundary/ACTUAL_DAMAGE_DECOMPOSITION.json')['methods']
    for method in ['RAW','BIN','VIRTUAL_GR']:
        group=[x for x in damage if x['condition']==method]
        assert len(group)==40
        chars=sum(int(x['normal_original_characters_deleted']) for x in group)/40
        cost=sum(float(x['normal_char_fraction']) for x in group)/40
        assert abs(chars-result['damage_utility']['methods'][method]['mean_normal_original_characters_deleted'])<1e-9
        assert abs(cost-result['damage_utility']['methods'][method]['mean_normal_char_fraction'])<1e-12
        assert sum(int(x['complete_payloads_removed']) for x in group)==decomposition[method]['total_complete_payloads_removed']
        for x in group:
            assert sum(int(x[k]) for k in ['normal_original_characters_deleted','watermark_characters_deleted','separator_characters_deleted'])==int(x['all_characters_deleted'])
        normal=[x for x in clean if x['method']==method]
        assert len(normal)==1000 and len({x['source_index'] for x in normal})==1000
        fp=sum(int(x['modified']) for x in normal)
        deleted=sum(int(x['characters_deleted']) for x in normal)
        total=sum(int(x['original_characters']) for x in normal)
        expected=result['clean']['methods'][method]
        assert (fp,deleted,total)==(expected['FP'],expected['original_characters_deleted'],expected['total_original_characters'])
        checks+=1040
    early=read('virtual_boundary_clean_gate/RESULTS.json')
    assert early['WSN']=='NOT_RUN_GATE_FAILED'
    assert early['scientific_status']=='CLEAN_GATE_FAILED_NO_WATERMARK_EFFICACY_MEASURED'
    seam=read('sentence_seam_intervention/RESULTS.json')
    assert all(seam['gates'].values())
    assert seam['WSN_conditions']['PERIOD_BIN']['YES']==5
    assert seam['causal_scope']!=result['scientific_status']
    print(json.dumps({'status':'PASS_PUBLIC_SUBSET_RECOMPUTATION','checks':checks,
                      'all320_ownership_slots_retained':True,'scientific_joint_pass':False,
                      'new_GPU_API_NLL_WSN':0,'withheld_API_or_corpus_replayed':False}))


if __name__=='__main__':main()
