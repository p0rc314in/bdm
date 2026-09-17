"""Full WikiText/Recall training: ordinary PyTorch, native dstack, hosted W&B."""

from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from reproduction import local_store as wandb

from data import DATASETS, PROTOCOLS, STEPS, VALIDATION_STEPS, Dataset, sha256
from model import ComparisonModel
from optim import AdamW, learning_rate


def save_checkpoint(path, network, optimizer, step, config, history, evaluations):
    state = dict(model=network.state_dict(), optimizer=optimizer.state_dict(), step=step,
                 config=config, history=history, evaluations=evaluations,
                 python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                 torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def restore_checkpoint(path, network, optimizer, config):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['config'] != config:
        raise ValueError('Configuration changed: use a new dstack run name for a new experiment')
    network.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    # The immutable stream position and learning-rate schedule follow exactly from step.
    return state['step'], state['history'], state['evaluations']


def publish_checkpoint(run, path, previous, step, days):
    artifact = wandb.Artifact(f'{run.id}-checkpoints', type='checkpoint', metadata={'step': step})
    artifact.ttl = None
    artifact.add_file(str(path), name='checkpoint.pt')
    uploaded = run.log_artifact(artifact, aliases=['latest'])
    uploaded.wait(timeout=300)
    if previous is not None and previous.id != uploaded.id:
        previous.ttl = timedelta(days=days)
        previous.save()
    print(f'CHECKPOINT step={step} artifact={uploaded.qualified_name}', flush=True)
    return uploaded


def loss_function(network, tokens, labels):
    return network(tokens, target=labels)


def verify_fresh_normalizer_initialization(network, benchmark, arm, seed, constructor_options):
    """Each experiment entry point replaces this with its arm's initialization check."""
    raise NotImplementedError('Launch training through an experiment entry point')


def update(network, forward_loss, optimizer, tokens, labels, benchmark, step, microbatch):
    network.train()
    optimizer.zero_grad()
    targets = int((labels != -100).sum())
    total_loss = torch.zeros((), device=tokens.device)
    for begin in range(0, len(tokens), microbatch):
        y = labels[begin:begin + microbatch]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=tokens.is_cuda):
            loss = forward_loss(tokens[begin:begin + microbatch], y)
        reported_loss = loss
        if isinstance(loss, tuple):
            loss, reported_loss, auxiliary = loss
            network.last_auxiliary_loss = float(auxiliary.detach())
        if microbatch < len(tokens):
            loss = loss * int((y != -100).sum()) / targets
            reported_loss = reported_loss * int((y != -100).sum()) / targets
        loss.backward()
        total_loss += reported_loss.detach()
    norm = optimizer.step(learning_rate(benchmark, step))
    return float(total_loss), norm, targets


def metrics_from_totals(totals, benchmark):
    values = dict(totals)
    values['nll'] = values['loss_sum'] / values['targets']
    values['accuracy'] = values['correct'] / values['targets']
    if benchmark == 'wiki':
        values['perplexity'] = math.exp(values['nll'])
        values['bits_per_byte'] = values['loss_sum'] / (math.log(2) * values['bytes'])
    else:
        values['exact_set_accuracy'] = values['exact'] / values['examples']
    return values


@torch.no_grad()
def evaluate(network, dataset, split, device, save_predictions=False, limit_per_condition=None):
    network.eval()
    totals = defaultdict(lambda: defaultdict(float))
    predictions, labels_saved, losses_saved = [], [], []
    families = ({c['id']: c['family'] for c in dataset.conditions} if dataset.benchmark == 'recall' else {})
    started = time.monotonic()
    for tokens, labels, condition in dataset.evaluation_batches(split, limit_per_condition):
        x, y = torch.from_numpy(tokens).to(device), torch.from_numpy(labels).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            logits = network(x)
        if dataset.benchmark == 'recall':
            logits = logits[:, -16:]
        losses = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction='none').view_as(y)
        predicted, valid = logits.argmax(-1), y != -100
        correct = predicted.eq(y)
        row = dict(loss_sum=float(losses[valid].double().sum()), correct=int(correct[valid].sum()),
                   targets=int(valid.sum()), examples=len(y))
        if dataset.benchmark == 'wiki':
            row['bytes'] = int(dataset.byte_lengths[labels[labels != -100]].sum(dtype=np.uint64))
            row['windows'] = 1
        else:
            row['exact'] = int(correct.all(-1).sum())
        groups = ['overall'] if condition == 'all' else ['overall', condition, families[condition]]
        if condition.startswith('pointer_'):
            groups.append('pointer_' + condition.split('_')[1])
        for group in groups:
            for key, value in row.items():
                totals[group][key] += value
        if save_predictions:
            predictions.append(predicted[valid].cpu().numpy().astype(np.uint16))
            labels_saved.append(y[valid].cpu().numpy().astype(np.uint16))
            losses_saved.append(losses[valid].cpu().numpy())
    result = {name: metrics_from_totals(row, dataset.benchmark) for name, row in totals.items()}
    result['seconds'] = time.monotonic() - started
    overall = result['overall']
    if dataset.benchmark == 'wiki':
        expected = dataset.manifest['evaluation']['splits'][split]
        assert (overall['targets'], overall['bytes'], overall['windows']) == (
            expected['scored_targets'], expected['scored_bytes'], expected['windows'])
    else:
        examples = limit_per_condition or 2048
        assert (overall['targets'], overall['examples']) == (30 * examples * 16, 30 * examples)
        assert all(result[c['id']]['examples'] == examples for c in dataset.conditions)
    arrays = {}
    if save_predictions:
        arrays = {'predictions': np.concatenate(predictions), 'labels': np.concatenate(labels_saved),
                  'nll': np.concatenate(losses_saved)}
        if dataset.benchmark == 'recall':
            arrays = {key: value.reshape(30, 2048, 16) for key, value in arrays.items()}
    print(f'EVALUATED split={split} metrics={json.dumps(overall)}', flush=True)
    return result, arrays


def train(args):
    invocation_started = time.monotonic()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This experiment requires an NVIDIA GPU with BF16 and FlashAttention support')
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Unsupported geometry must fail instead of silently using quadratic attention.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    torch.use_deterministic_algorithms(True, warn_only=args.arm.startswith('sdm_'))
    if args.arm.startswith('bsdm_'):
        from numerics import install
        numerical = install(128)
    else:
        numerical = {'outer_softmax': 'not applicable'}
    benchmark = args.benchmark
    config = dict(benchmark=benchmark, protocol=PROTOCOLS[benchmark], tier='small_wikitext_plus_recall',
                  dataset=DATASETS[benchmark][0], manifest_sha256=DATASETS[benchmark][1],
                  width=128, layers=8, ffn_width=512, heads=1 if args.arm.startswith('bsdm_') else 4, seed=args.seed,
                  steps=STEPS[benchmark], batch_size=8 if benchmark == 'wiki' else 32,
                  microbatch=args.microbatch, model=args.arm, arm=args.arm,
                  tied_embeddings=False, mixer_backend='role' if args.arm.startswith('bsdm_') else 'NVIDIA native chunk GDN2',
                  compute_dtype='bfloat16', gradient_accumulation_dtype='bfloat16',
                  master_dtype='float32' if benchmark == 'wiki' else None,
                  moment_dtype='float32' if benchmark == 'wiki' else 'bfloat16',
                  optimizer='AdamW', betas=[0.9, 0.95], eps=1e-8, weight_decay=0.01,
                  weight_decay_exclusions='existing parameter _no_weight_decay annotations', clip_norm=1.0, lr=3e-4,
                  warmup=540 if benchmark == 'wiki' else 100,
                  schedule='cosine-to-0.1' if benchmark == 'wiki' else 'constant',
                  execution_profile='bsdm-final-torch211-x128-wandb-v1', torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, compiled=not args.no_compile)
    assert torch.__version__ == '2.11.0+cu128'
    if getattr(args, 'training_state_dtype', None) is not None:
        config.update(training_state_dtype=args.training_state_dtype,
                      state_arithmetic_dtype='float32',
                      state_gradient_accumulation_dtype='float32',
                      state_rounding='forward-and-replay agree at persisted checkpoint boundaries',
                      comparison='bsdm-training-state-storage-three-seeds-v1')
    config.update(gradient_norm='native Wiki' if benchmark == 'wiki' else 'FP32 accumulation',
                  MKL_CBWR=os.environ.get('MKL_CBWR'), sources=json.loads(Path('sources.json').read_text()))
    if getattr(args, 'campaign', False):
        config.update(campaign=('bsdm-nvidia-final-three-seeds-v1' if args.arm.startswith('bsdm_nvidia_')
                                else 'bsdm-final-three-seeds-v1'),
                      initialization_policy='Common shell/task seed; native controller seed 100000*seed+1000+layer; SDM native initialization_seed',
                      strict_determinism=not args.arm.startswith('sdm_'))
        config.update(heads=1 if args.arm.startswith(('sdm_', 'bsdm_')) else 4,
                      ffn_width=240 if args.arm=='mom_m4_k2_shared_ffn240' else 512,
                      mixer_backend=('canonical role' if args.arm.startswith('bsdm_') else
                                     'original SDM; native kernels with correctness repairs' if args.arm.startswith('sdm_') else
                                     'Flash SDPA' if args.arm=='attention' else
                                     'FLA 0.5.2 native GDN1' if args.arm=='gdn1' else
                                     'FLA 0.5.2 MoM with preserved causal corrections' if args.arm.startswith('mom_') else
                                     'NVIDIA native chunk GDN2'))
    constructor = ComparisonModel
    constructor_options = {}
    if args.compact_replay != 'auto':
        constructor_options['compact_replay'] = args.compact_replay == 'bf16'
        config.update(recurrent_replay=args.compact_replay,
                      execution_profile='bsdm-explicit-recurrent-replay-' + args.compact_replay + '-v1')
    if args.seed != 0:
        config.update(execution_profile=config['execution_profile']+'-model-seed'+str(args.seed),
                      initialization_comparison='Fresh model seed; all initial tensors, ordering and decay annotations matched against the same-seed PyTorch unit-normalizer reference. Frozen training stream and all other settings unchanged.')
    config.update(getattr(args, 'experiment_config', {}))
    run_id = os.environ.get('WANDB_RUN_ID') or os.environ.get('DSTACK_RUN_NAME')
    if not run_id:
        raise ValueError('A stable WANDB_RUN_ID or dstack run name is required for restart')
    with wandb.init(project=os.environ.get('WANDB_PROJECT', 'reproduction'),
                    entity=os.environ.get('WANDB_ENTITY', 'local'), id=run_id,
                    resume='allow', config=config, mode='online') as run:
        run.define_metric('train/*', step_metric='train/step')
        run.define_metric('validation/*', step_metric='validation/step')
        run.define_metric('matched/*', step_metric='matched/step')
        run.define_metric('probe/*', step_metric='probe/step')
        previous = None
        if run.resumed:
            prior = list(wandb.Api().run(f'{run.entity}/{run.project}/{run.id}').logged_artifacts())
            results = [a for a in prior if a.type == 'result' and 'final' in a.aliases]
            for artifact in prior:
                if artifact.type == 'checkpoint' and (results or 'latest' not in artifact.aliases):
                    if artifact.ttl is None:
                        artifact.ttl = timedelta(days=args.checkpoint_days)
                        artifact.save()
            if results:
                print(f'ALREADY_COMPLETE artifact={results[-1].qualified_name}', flush=True)
                return
            previous = next((a for a in prior if a.type == 'checkpoint' and 'latest' in a.aliases), None)
        artifact = run.use_artifact(DATASETS[benchmark][0], type='dataset')
        dataset = Dataset(artifact.download(), benchmark)
        network = constructor(benchmark, args.arm, seed=args.seed, **constructor_options)
        common_hash = network.common_hash()
        expected_common = ('9fb3872da2c49015d2e53511a9662a06ebb53753f43d33b9594519e915a31015'
                           if benchmark == 'wiki' else '6f7c9b717d3bbde7d5983d08ff2faaad674102a5383ad79c3591ae914a454360')
        if args.seed != 0 or getattr(args, 'campaign', False):
            expected_common = verify_fresh_normalizer_initialization(
                network, benchmark, args.arm, args.seed, constructor_options)
        if common_hash != expected_common:
            raise ValueError(f'Historical common initialization changed: {common_hash}')
        resolved = network.resolved_model() if hasattr(network, 'resolved_model') else network.config.record()
        run.config.update({'resolved_model': resolved, 'accounting': network.accounting(),
                           'common_parameter_sha256': common_hash})
        network = network.cuda().bfloat16()
        if args.arm.startswith('bsdm_'):
            from bdm.role_bank_kernel import _replay_settings
            mixer = network.layers[0].attention
            interval, snapshot_dtype = _replay_settings(mixer.bank_size, mixer.value_width,
                torch.bfloat16, torch.cuda.get_device_capability(), mixer.compact_replay)
            state_dtype = getattr(mixer.config, 'training_state_dtype', None)
            if state_dtype is not None:
                snapshot_dtype = getattr(torch, state_dtype)
            if getattr(args, 'training_state_dtype', None) is not None:
                assert state_dtype == args.training_state_dtype
            replay_record = dict(requested=args.compact_replay, interval=interval,
                                 snapshot_dtype=str(snapshot_dtype), capability=list(torch.cuda.get_device_capability()))
            numerical['recurrent_replay'] = replay_record
            run.config.update({'resolved_recurrent_replay': replay_record})
            print('RECURRENT_REPLAY ' + json.dumps(replay_record), flush=True)
        optimizer = AdamW(network, benchmark)
        step, history, evaluations = 0, [], {}
        if previous is not None:
            path = Path(previous.download()) / 'checkpoint.pt'
            step, history, evaluations = restore_checkpoint(path, network, optimizer, config)
        if hasattr(network,'run_diagnostics'):
            network.run_diagnostics(run,benchmark)
        print(f'{"RESUMED" if previous else "STARTED"} step={step} device={torch.cuda.get_device_name()} benchmark={benchmark}', flush=True)
        run.summary.update(dict(resumed_from_step=step, device=torch.cuda.get_device_name(),
                           dstack_run=os.environ.get('DSTACK_RUN_NAME'), parameters=sum(p.numel() for p in network.parameters()),
                           **network.accounting(), common_parameter_sha256=common_hash))
        forward_loss = lambda x, y: loss_function(network, x, y)
        if not args.no_compile:
            forward_loss = torch.compile(forward_loss, dynamic=False)
        out = Path('outputs')
        out.mkdir(exist_ok=True)
        last_checkpoint = time.monotonic()
        profile_pending = os.environ.get('PROFILE_ONE_TRAIN_STEP') == '1'
        profile_start_step, stable_graph_steps = step, 0
        while step < STEPS[benchmark]:
            tokens, labels = dataset.train_batch(step)
            x, y = torch.from_numpy(tokens).cuda(), torch.from_numpy(labels).cuda()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            from torch._dynamo.utils import counters
            graphs = counters['stats']['unique_graphs']
            profile_this_step = (profile_pending and step >= profile_start_step + 32
                                 and stable_graph_steps >= 8)
            diagnostic = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA])
                          if profile_this_step else nullcontext())
            started = time.monotonic()
            with diagnostic:
                loss, norm, targets = update(network, forward_loss, optimizer, x, y, benchmark, step + 1, args.microbatch)
            torch.cuda.synchronize()
            step += 1
            seconds = time.monotonic() - started
            row = dict(step=step, loss=loss, grad_norm=norm, targets=targets,
                       lr=learning_rate(benchmark, step), seconds=seconds,
                       new_compiled_graphs=counters['stats']['unique_graphs']-graphs,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
            stable_graph_steps = stable_graph_steps + 1 if row['new_compiled_graphs'] == 0 else 0
            if profile_this_step:
                profile_pending = False
                row['diagnostic_profile'] = True
                trace_path = out / 'training-step-trace.json'
                diagnostic.export_chrome_trace(str(trace_path))
                table = diagnostic.key_averages().table(sort_by='self_cuda_time_total', row_limit=25)
                print('ACTUAL_TRAIN_PROFILE step=' + str(step) + '\n' + table, flush=True)
                (out / 'training-step-table.txt').write_text(table)
                profile_artifact = wandb.Artifact(f'{run.id}-training-profile', type='diagnostic',
                    metadata={'step': step, 'compiled': not args.no_compile,
                              'separate_gpu_processes': False, 'optimizer_update_included': True})
                profile_artifact.ttl = None
                profile_artifact.add_file(str(trace_path))
                profile_artifact.add_file(str(out / 'training-step-table.txt'))
                profile_artifact.add_file(__file__, name='train.py')
                run.log_artifact(profile_artifact).wait(timeout=300)
            if hasattr(network, 'last_auxiliary_loss'):
                row.update(auxiliary_loss=network.last_auxiliary_loss,
                           objective=loss+getattr(network, 'auxiliary_coefficient', .01)*network.last_auxiliary_loss)
            if benchmark == 'recall':
                row['condition'] = dataset.conditions[int(dataset.arrays['train_condition_ids'][step-1])]['id']
            history.append(row)
            run.log({f'train/{key}': value for key, value in row.items()})
            if step % 250 == 0:
                recent = history[-250:]
                metrics = {'matched/step': step, 'matched/nll': sum(r['loss']*r['targets'] for r in recent)/sum(r['targets'] for r in recent),
                           'matched/steps_per_second': 250/sum(r['seconds'] for r in recent),
                           'matched/new_compiled_graphs': sum(r['new_compiled_graphs'] for r in recent),
                           'matched/peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in recent)}
                if benchmark == 'recall':
                    for family, prefix in [('span', 'span_'), ('overwrite', 'overwrite_'), ('one_hop', 'pointer_h1_')]:
                        rows = [r for r in recent if r['condition'].startswith(prefix)]
                        metrics[f'matched/{family}_nll'] = sum(r['loss']*r['targets'] for r in rows)/sum(r['targets'] for r in rows)
                run.log(metrics)
                print('MATCHED ' + json.dumps(metrics), flush=True)
            if step == 1 or step % 100 == 0:
                rate = len(history[-100:]) / sum(r['seconds'] for r in history[-100:])
                print(f'PROGRESS step={step}/{STEPS[benchmark]} loss={loss:.5f} steps_per_second={rate:.2f}', flush=True)
            if benchmark == 'recall' and step % 1000 == 0:
                if getattr(args, 'checkpoint_before_probe', False):
                    save_checkpoint(out / 'checkpoint.pt', network, optimizer, step, config, history, evaluations)
                    previous = publish_checkpoint(run, out / 'checkpoint.pt', previous, step, args.checkpoint_days)
                    last_checkpoint = time.monotonic()
                probe, _ = evaluate(network, dataset, 'validation', 'cuda', limit_per_condition=32)
                evaluations[f'probe-{step}'] = probe
                run.log({'probe/step': step, **{f'probe/{family}_exact': probe[family]['exact_set_accuracy']
                    for family in ('span_recall', 'overwrite_recall', 'pointer_h1')}})
                print('PROBE ' + json.dumps({'step': step, 'metrics': probe}), flush=True)
            if benchmark == 'wiki' and step in VALIDATION_STEPS[:-1]:
                metrics, _ = evaluate(network, dataset, 'validation', 'cuda')
                evaluations[str(step)] = metrics
                run.log({'validation/step': step, **{f'validation/{k}': v for k, v in metrics['overall'].items()}})
                torch.save(dict(model=network.state_dict(), config=config, step=step, metrics=metrics), out / 'model.pt')
                analysis = wandb.Artifact(f'{run.id}-analysis', type='model', metadata={'step': step})
                analysis.ttl = None
                analysis.add_file(str(out / 'model.pt'))
                run.log_artifact(analysis, aliases=[f'step-{step}']).wait(timeout=300)
            interrupt = step == args.interrupt_after_step
            if (time.monotonic() - last_checkpoint >= args.checkpoint_seconds or interrupt
                    or step == STEPS[benchmark] or (benchmark == 'wiki' and step in VALIDATION_STEPS)):
                save_checkpoint(out / 'checkpoint.pt', network, optimizer, step, config, history, evaluations)
                previous = publish_checkpoint(run, out / 'checkpoint.pt', previous, step, args.checkpoint_days)
                last_checkpoint = time.monotonic()
            if interrupt:
                print(f'INTENTIONAL_INTERRUPTION step={step}; dstack must retry', flush=True)
                os._exit(75)
        metrics = {}
        for split in ('validation', 'test'):
            metrics[split], arrays = evaluate(network, dataset, split, 'cuda', save_predictions=True)
            np.savez_compressed(out / f'{split}.npz', **arrays)
            run.summary.update({f'{split}/{k}': v for k, v in metrics[split]['overall'].items()})
        if benchmark == 'wiki':
            evaluations[str(step)] = metrics['validation']
        torch.save(dict(model=network.state_dict(), config=config, step=step), out / 'model.pt')
        # Reopen the actual terminal weights and check inference before task cleanup.
        x = torch.from_numpy(dataset.train_batch(0)[0][:1]).cuda()
        with torch.no_grad():
            restored = constructor(benchmark, args.arm, seed=args.seed, **constructor_options).cuda().bfloat16().eval()
            restored.load_state_dict(torch.load(out / 'model.pt', map_location='cuda', weights_only=False)['model'])
            reload_check = dict(sequence_length=x.shape[1], inference_bitwise=True)
            sdm_wiki = benchmark == 'wiki' and args.arm.startswith('sdm_')
            if sdm_wiki:
                # Native SDM prefill has nondeterministic scatter-product reductions.
                # Frozen-checkpoint evidence is in SDM_TERMINAL_RECOVERY.md. Verify
                # all restored tensors exactly; preserve full-context evaluation.
                expected = network.state_dict()
                actual = restored.state_dict()
                assert actual.keys() == expected.keys()
                for name in expected:
                    torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                reference = network(x)
                reloaded = restored(x)
                if sdm_wiki:
                    assert torch.isfinite(reference).all() and torch.isfinite(reloaded).all()
                    delta = reloaded.float() - reference.float()
                    reload_check.update(full_context_bitwise=torch.equal(reference, reloaded),
                        full_context_max_abs=float(delta.abs().max()),
                        full_context_rms=float(delta.square().mean().sqrt()),
                        full_context_argmax_differing=int((reference.argmax(-1) != reloaded.argmax(-1)).sum()))
                    # T=1 selects SDM's native fused decode, which the frozen test
                    # verified bitwise on repeats and independently loaded models.
                    short_reference = network(x[:, :1])
                    torch.testing.assert_close(restored(x[:, :1]), short_reference, rtol=0, atol=0)
                    for name, value in restored.state_dict().items():
                        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
                    reload_check.update(sequence_length=1, state_tensors_bitwise=True,
                                        method='native SDM fused decode; full prefill differences recorded')
                else:
                    torch.testing.assert_close(reloaded, reference, rtol=0, atol=0)
        report = dict(config=config, completed_steps=step, target_presentations=sum(r['targets'] for r in history),
                      metrics=metrics, validation_checkpoints=evaluations, terminal_reload_verified=True,
                      terminal_reload_check=reload_check,
                      training_seconds=sum(r['seconds'] for r in history),
                      peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in history),
                      peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in history), device=torch.cuda.get_device_name(),
                      invocation_wall_seconds=time.monotonic()-invocation_started,
                      accounting=network.accounting(), numerical=numerical,
                      model_sha256=sha256(out / 'model.pt'))
        assert report['target_presentations'] == (353941347 if benchmark == 'wiki' else 15360000)
        run.summary.update({key: report[key] for key in ('completed_steps', 'target_presentations', 'terminal_reload_verified',
                                                       'training_seconds', 'peak_allocated_bytes', 'model_sha256')})
        (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        (out / 'curve.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history))
        result = wandb.Artifact(f'{run.id}-results', type='result', metadata={'completed_steps': step})
        result.ttl = None
        for name in ('model.pt', 'metrics.json', 'curve.jsonl', 'validation.npz', 'test.npz'):
            result.add_file(str(out / name), name=name)
        result.add_file('sources.json', name='source/extraction.json')
        uploaded = run.log_artifact(result, aliases=['final'])
        uploaded.wait(timeout=300)
        if previous is not None:
            previous.ttl = timedelta(days=args.checkpoint_days)
            previous.save()
        print(f'COMPLETE steps={step} results={uploaded.qualified_name}', flush=True)
