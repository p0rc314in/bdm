"""Verify recorded evidence; does not execute the research experiments."""
from pathlib import Path
import json
import math
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / 'data' / name).read_text())


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (a, b)


def pick(m, *names):
    for n in names:
        if n in m:
            return m[n]
    raise AssertionError(('missing field', names))


def aggregate(r, family, split='test'):
    m = r[split]
    if family == 'overall' and 'nll' in m:
        return m
    names = {'overall': ('overall', 'all'), 'span': ('span_recall', 'span'),
             'overwrite': ('overwrite_recall', 'overwrite'),
             'one_hop': ('pointer_h1', 'one_hop')}
    return pick(m, *names[family])


def conditions(m):
    if 'conditions' in m:
        return {x['condition_id']: x for x in m['conditions']}
    return {k: v for k, v in m.items() if re.fullmatch(
        r'(pointer_h(1|2|4|8)|span_s(1|4|8|16)|overwrite_v(1|2|4))_n\d+', k)}


def verify_quality():
    rows = load('quality-endpoints.json')
    assert len(rows) == 36
    ids = {(r['arm'], r['benchmark'], r['seed']) for r in rows}
    assert len(ids) == len(rows)
    for r in rows:
        assert r['step'] == {'wiki': 21603, 'recall': 30000}[r['benchmark']]
        for split in ('validation', 'test'):
            overall = aggregate(r, 'overall', split)
            if r['benchmark'] == 'wiki':
                targets, windows, bytes_ = ((247416, 484, 1145546) if split == 'validation'
                                            else (283426, 554, 1289122))
                assert pick(overall, 'targets', 'scored_targets') == targets
                assert overall['windows'] == windows
                assert pick(overall, 'bytes', 'scored_bytes') == bytes_
                close(pick(overall, 'loss_sum', 'total_nll') / targets, overall['nll'])
                close(math.exp(overall['nll']), overall['perplexity'])
            else:
                cc = conditions(r[split])
                assert len(cc) == 30, (r['arm'], split, len(cc))
                assert sum(c['examples'] for c in cc.values()) == 61440
                assert all(c['examples'] == 2048 for c in cc.values())
                exact = sum(pick(c, 'exact', 'exact_sets') for c in cc.values())
                correct = sum(pick(c, 'correct', 'correct_queries') for c in cc.values())
                close(exact / 61440, overall['exact_set_accuracy'])
                close(correct / 983040, overall['accuracy'])
                close(sum(c['loss_sum'] for c in cc.values()) / 983040, overall['nll'])
    summaries = load('quality-summary.json')
    for summary in summaries:
        rr = [r for r in rows if r['arm'] == summary['arm']]
        for task in ('wiki', 'recall'):
            ss = sorted((r for r in rr if r['benchmark'] == task), key=lambda r: r['seed'])
            assert [r['seed'] for r in ss] == [0, 1, 2]
            values = {'wiki_nll': [aggregate(r, 'overall')['nll'] for r in ss]} if task == 'wiki' else {
                f'{family}_{metric}': [aggregate(r, family)[metric] * 100 for r in ss]
                for family in ('overall', 'span', 'overwrite', 'one_hop')
                for metric in ('accuracy', 'exact_set_accuracy')}
            for key, vals in values.items():
                close(statistics.mean(vals), summary[key]['mean'])
                close(statistics.stdev(vals), summary[key]['sd'])
                assert vals == summary[key]['seeds']
    assert {s['arm'] for s in summaries} == {
        'attention', 'gdn1', 'gdn2_nvidia', 'mom_m4_k2_shared_ffn240',
        'sdm_native_k64_n2048', 'bsdm_init_variance_matched'}
    assert {r['arm'] for r in rows} == {s['arm'] for s in summaries}
    print('36 complete quality endpoints; six recomputed three-seed model summaries: OK')


def verify_performance():
    for file in ('training-performance.json', 'inference-performance.json'):
        rows = load(file)
        assert len(rows) == 40
        assert len({(r['family'], r['context']) for r in rows}) == 40
        assert len({r['family'] for r in rows}) == 5
        assert {r['context'] for r in rows} == {8192 * 2**i for i in range(8)}
        for r in rows:
            if r['status'] != 'ok':
                continue
            assert (r['layers'], r['width']) == (16, 2048)
            for key, expected in [('batch', 1), ('ffn_width', 5632)]:
                if key in r:
                    assert r[key] == expected
            if file.startswith('training'):
                assert len(r['samples_seconds']) == r['timed_steps']
                close(statistics.mean(r['samples_seconds']) * 1000, r['mean_ms'])
                assert r['new_timed_graphs'] == 0
            else:
                assert len(r['prefill_seconds']) == (3 if r['family'] == 'bsdm' else 5)
                close(statistics.mean(r['prefill_seconds']), r['prefill_mean_seconds'])
                if 'decode_seconds_per_token' in r:
                    assert len(r['decode_seconds_per_token']) == 5
                    close(statistics.mean(r['decode_seconds_per_token']), r['decode_mean_seconds'])
            if r['family'] in ('bsdm', 'sdm', 'sdm_native'):
                model = r.get('model', r.get('mixer'))
                n = model.get('logical_rows', model.get('slots_per_head'))
                assert n == r['context']
        print(f'{file}: 40 cells, means and sparse capacity identities OK')


def verify_sustained_decode():
    rows = load('decode-performance.json')
    assert len(rows) == len({(r['family'], r['context']) for r in rows}) == 40
    assert len({r['family'] for r in rows}) == 5
    assert {r['context'] for r in rows} == {8192 * 2**i for i in range(8)}
    assert sum(r['status'] == 'ok' for r in rows) == 38
    for r in rows:
        if r['status'] != 'ok':
            assert r['context'] == 1048576
            assert r['family'] in ('attention', 'sdm_native')
            assert r['status'] == 'unmeasured: prefill OOM'
            assert r['decode_ms'] is None and not r['samples_ms']
            continue
        assert len(r['samples_ms']) == len(r['cuda_event_samples_ms']) == 3
        close(statistics.mean(r['samples_ms']), r['decode_ms'])
        assert r['peak_allocated_bytes'] > 0
        assert all(end - start == 256 for start, end in zip(r['start_context'], r['end_context']))
    print('Corrected sustained decode: 38 measured cells, two prefill OOM endpoints OK')


def verify_initialization_and_babylm():
    scale = load('initialization-scale-three-seed.json')
    for r in scale:
        for group in ('current', 'reduced', 'paired'):
            v = r[group]['values']
            assert len(v) == 3
            close(statistics.mean(v), r[group]['mean'])
            close(statistics.stdev(v), r[group]['sd'])
        for a, b, delta in zip(r['current']['values'], r['reduced']['values'], r['paired']['values']):
            close(b - a, delta)
    b = load('babylm.json')
    assert set(b['models']) == {'Attention', 'SDM', 'BSDM'}
    assert b['configuration']['steps'] == 102852
    assert b['configuration']['target_tokens'] == 1685114880
    close(b['configuration']['initial_factor_multiplier'], 2**-.5)
    assert b['accounting']['total_parameters'] == sum(b['accounting'][k] for k in (
        'input_embedding', 'output_embedding', 'learned_initial_memory', 'processing_parameters'))
    tasks = {'boolq', 'mnli', 'mrpc', 'multirc', 'qqp', 'rte', 'wsc'}
    for model in b['models'].values():
        assert set(model['finetunes']) == tasks
        assert len(model['terminal']) == 7
    assert b['configuration']['memory_rows'] == 2048
    assert b['configuration']['memory']['bank_count'] * b['configuration']['memory']['bank_size'] == 2048
    for name in ('SDM', 'BSDM'):
        c = b['model_configurations'][name]
        assert c['memory_rows'] == c['context'] == 2048
        assert c['accessed_rows_per_role'] == 64 and c['elastic_coefficient'] == 0
        a = b['accounting_by_model'][name]
        assert a['total_parameters'] == sum(a[k] for k in ('input_embedding', 'output_embedding', 'learned_initial_memory', 'processing_parameters'))
        for task, ft in b['models'][name]['finetunes'].items():
            assert len(ft['epochs']) == (30 if task == 'wsc' else 10)
            metric = ft['selection_metric']
            close(ft['best_validation'][metric] - ft['selection_validation'][metric], ft['reopened_score_delta'])
            if (name, task) != ('SDM', 'multirc'):
                close(ft['reopened_score_delta'], 0)
    close(b['models']['SDM']['finetunes']['multirc']['reopened_score_delta'], 1 / 2424)
    assert len(b['training_curve']) == 1029
    assert b['training_curve'][-1]['step'] == 102852
    for name, key in [('Attention', 'attention'), ('SDM', 'native_sdm'), ('BSDM', 'bsdm_n2048')]:
        close(b['training_curve'][-1][name], b['matched_training_nll'][key])
        assert [p['exposure_millions'] for p in b['checkpoint_curves'][name]] == list(range(100, 1001, 100))
    for ft in b['models']['BSDM']['finetunes'].values():
        assert ft['reopened_score_delta'] == 0
    print('Three-seed initialization arithmetic and complete BabyLM evaluation: OK')


def verify_trained_memory():
    d = load('trained-bsdm-memory.json')
    c = d['configuration']
    assert (c['layers'], c['width'], c['memory_rows'], c['selected_banks'],
            c['bank_size'], c['seed']) == (8, 128, 16384, 8, 8, 0)
    assert c['steps'] * c['input_positions_per_update'] == c['input_positions']
    dense = c['layers'] * c['memory_rows'] * c['width'] * d['element_bytes']
    assert dense == 32 * 2**20
    assert [a['elastic_coefficient'] for a in d['arms']] == [0, .01]
    terminal = {}
    for arm in d['arms']:
        for split, rows in arm['splits'].items():
            assert split in ('validation', 'test') and len(rows) == 20
            for r in rows:
                banks = r['written_banks_by_request_and_layer']
                assert len(banks) == r['occupancy_requests'] == 64
                assert all(len(x) == c['layers'] for x in banks)
                assert all(8 <= b <= c['bank_count'] for x in banks for b in x)
                mean_bank_count = sum(map(sum, banks)) / len(banks)
                close(r['bank_fraction'], mean_bank_count / (c['layers'] * c['bank_count']))
                close(r['bank_value_bytes'], mean_bank_count * c['bank_size'] * c['width'] * 2)
                assert r['examples'] == 2048
                close(r['query_accuracy'], r['correct_queries'] / (r['examples'] * 16))
                close(r['exact_accuracy'], r['exact_requests'] / r['examples'])
                if split == 'test':
                    terminal[arm['elastic_coefficient'], r['family'], r['sequence_length']] = r
    assert len(d['requests']) == 20
    for p in d['requests']:
        h = p['request_positions'] - 16
        assert p['history_positions'] == h
        close(p['attention_kv_bytes'], 2 * h * c['layers'] * c['width'] * 2)
        close(p['dense_sdm_bsdm_bytes'], dense)
        close(p['cow_worst_case_bytes'], min(c['memory_rows'], 64 * h) * c['layers'] * c['width'] * 2)
        for coefficient, byte_key, prefix in [(0, 'cow_bytes', 'baseline'), (.01, 'elastic_cow_bytes', 'elastic')]:
            r = terminal[coefficient, p['task'], p['request_positions']]
            close(p[byte_key], r['bank_value_bytes'])
            close(p[prefix + '_query_accuracy'], r['query_accuracy'])
            close(p[prefix + '_exact_accuracy'], r['exact_accuracy'])
    for p in d['prefixes']:
        for coefficient, key in [(0, 'cow_bytes'), (.01, 'elastic_cow_bytes')]:
            r = terminal[coefficient, p['task'], 16384]
            v = next(x for x in r['prefixes'] if x['tokens'] == p['history_positions'])
            close(p[key], v['bank_value_bytes'])
    assert [a['elastic_coefficient'] for a in d['lambda_sweep']] == [0, .001, .003, .01, .03, .06, .12]
    for arm in d['lambda_sweep']:
        assert len(arm['validation']) == len(arm['test']) == 20
        for r in arm['validation'] + arm['test']:
            close(r['bank_value_bytes'], r['bank_fraction'] * dense)
    print('Trained memory: both terminal splits, bank-count/accuracy arithmetic and plotted bytes OK')


def verify_initial_memory_geometry():
    d = load('initial-memory-geometry.json')
    c = d['configuration']
    assert c['bank_count'] * c['bank_size'] == c['memory_rows'] == 4096
    assert math.prod(c['router_factors']) == c['bank_count']

    def additive(families, examples):
        for k in ('examples', 'exact_sets', 'correct_queries', 'queries'):
            assert families['all'][k] == sum(families[f][k] for f in
                                             ('span_recall', 'overwrite_recall', 'pointer_chase'))
        assert families['all']['examples'] == examples
        assert families['one_hop']['examples'] == 4 * examples // 30
        for f in families.values():
            assert f['queries'] == 16 * f['examples'] and 0 <= f['exact_sets'] <= f['examples']

    vectors = {'near_square_rows': 64 + 64, 'bank_row_column_slot': 16 + 32 + 8,
               'bank_matrix': 16 * 8 + 32 * 8}
    for arm in d['complete_runs']:
        assert arm['learned_vectors_per_layer'] == vectors[arm['arm']]
        for split in ('validation', 'test'):
            additive(arm[split], 61440)
    print('Initial-memory geometry: additive counts and factor sizes OK')


if __name__ == '__main__':
    verify_quality()
    verify_performance()
    verify_sustained_decode()
    verify_initialization_and_babylm()
    verify_trained_memory()
    verify_initial_memory_geometry()
    print('Recorded-evidence verification passed. No experiment was rerun.')
