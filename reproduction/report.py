"""Tabulate and plot NEW run outputs without consulting the paper's saved tables."""
from pathlib import Path
import csv
import json


def report(work, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    work=Path(work).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    rows=[];curves=[]
    for identity in sorted(work.glob('*/identity.json')):
        root=identity.parent;name=root.name;runargs=json.loads(identity.read_text())['arguments']
        out=root/'workspace/outputs'
        curve=out/'curve.jsonl'
        if curve.exists():
            values=[json.loads(line) for line in curve.read_text().splitlines() if line]
            curves.append((name,[v['step'] for v in values],[v['loss'] for v in values]))
        baby=out/'babylm/training.json'
        if baby.exists():
            values=json.loads(baby.read_text())['history']
            curves.append((name,[v['step'] for v in values],[v['nll'] for v in values]))
        metrics=out/'metrics.json'
        if metrics.exists():
            data=json.loads(metrics.read_text())
            rows.append({'run':name,'phase':'terminal','metric':'completed_steps','value':data['completed_steps']})
            for split,groups in data.get('metrics',{}).items():
                for group,metrics in groups.items():
                    if isinstance(metrics,dict):
                        for metric in ['nll','accuracy','query_accuracy','exact_set_accuracy','perplexity']:
                            if metric in metrics:rows.append(dict(run=name,phase=split,metric=group+'/'+metric,value=metrics[metric]))
            for split, result in data.get('terminal', {}).items():
                for condition in result['conditions']:
                    for metric in ['nll','query_accuracy','exact_accuracy','bank_fraction','bank_value_bytes']:
                        if metric in condition:
                            rows.append(dict(run=name,phase=split,metric=str(condition['id'])+'/'+metric,value=condition[metric]))
        for path in (out/'babylm/inference/1000').glob('*.json'):
            data=json.loads(path.read_text())
            def numbers(value,prefix=''):
                if isinstance(value,dict):
                    for key,item in value.items():yield from numbers(item,prefix+'/'+key)
                elif isinstance(value,(int,float)):yield prefix,value
            for metric,value in numbers(data.get('metrics',{})):
                rows.append(dict(run=name,phase='terminal',metric=path.stem+metric,value=value))
        for filename in ['COMPLETE.json']:
            path=root/filename
            if path.exists():
                data=json.loads(path.read_text())
                rows.append(dict(run=name,phase=runargs['phase'],metric='status',value=data['status']))
                if data['status']!='ok':continue
                if 'decode_mean_seconds' in data:data['decode_ms']=1000*data['decode_mean_seconds']
                for metric in ['mean_ms','tokens_per_second','decode_ms','peak_allocated_bytes','prefill_mean_seconds','prefill_peak_bytes','decode_peak_bytes']:
                    if metric in data:rows.append(dict(run=name,phase=runargs['phase'],metric=metric,value=data[metric]))
        for path in out.glob('babylm/finetune-*/COMPLETE.json'):
            data=json.loads(path.read_text())
            metric=data['selection_metric']
            rows.append(dict(run=name,phase=path.parent.name,metric=metric,value=data['best_validation'][metric]))
            # Preserve the complete selected-checkpoint record for downstream analysis.
            target=output/(name+'-'+path.parent.name+'.json');target.write_text(json.dumps(data,indent=2)+'\n')
    with (output/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['run','phase','metric','value']);writer.writeheader();writer.writerows(rows)
    if curves:
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        for name,x,y in curves:ax.plot(x,y,label=name,linewidth=1)
        ax.set(xlabel='Optimizer update',ylabel='Training language/task NLL')
        ax.legend(fontsize=6);ax.spines[['right','top']].set_visible(False)
        fig.savefig(output/'loss.png',dpi=300);fig.savefig(output/'loss.pdf');plt.close(fig)
    print(json.dumps(dict(rows=len(rows),curves=len(curves),output=str(output))))
