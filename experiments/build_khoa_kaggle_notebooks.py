"""Build Kaggle CLI submissions from the checked-in notebooks and runner."""

import argparse
import ast
import json
from pathlib import Path


paths = [
    'src/khoa/__init__.py', 'src/khoa/baselines.py', 'src/khoa/detector.py', 'src/khoa/studd.py',
    'experiments/reproduce_khoa_6datasets.py',
]
bundle = {path: Path(path).read_text() for path in paths}
# The standalone Kaggle package contains only src.khoa.
bundle["src/__init__.py"] = '"""Standalone package for the Khoa Kaggle runner."""\n'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
base = parser.parse_args().output
for notebook in sorted(Path('notebooks/khoa').glob('*.ipynb')):
    slug = notebook.stem[3:]
    source = json.loads(notebook.read_text())
    config = {}
    for line in source['cells'][2]['source']:
        if line.startswith('class '):
            break
        if line.startswith(('DATASET =','TEACHER =','STUDENT =')):
            assignment = ast.parse(line).body[0]
            config[assignment.targets[0].id] = ast.literal_eval(assignment.value)
    name = config['DATASET']
    target = base / slug
    target.mkdir(parents=True, exist_ok=True)
    setup, load = source['cells'][2:4]
    # Cell 2 contains the metadata plus the notebook-local detector. The
    # standalone runner imports its copy, so only retain metadata here.
    setup_text = ''.join(setup['source']).split('# Page–Hinkley from',1)[0]
    if 'class BaseDriftDetector' in setup_text:
        setup_text = setup_text.split('class BaseDriftDetector',1)[0]
    setup['source'] = setup_text.splitlines(keepends=True)
    run_cell = f"""from pathlib import Path
import sys

project_root = Path("/kaggle/working/studd-10methods")
sources = {bundle!r}
for relative_path, text in sources.items():
    path = project_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
sys.path.insert(0, str(project_root))
from experiments.reproduce_khoa_6datasets import evaluate_dataset

evaluate_dataset(
    X, y, dataset=DATASET, slug=SLUG, teacher=TEACHER, student=STUDENT,
    window=W, seed=SEED, source=DATA_URL, checksum=EXPECTED_SHA256,
    output=Path("/kaggle/working/reproduction"),
)
"""
    def make_cell(body, id):
        return {
            'cell_type':'code', 'id':id, 'metadata':{}, 'source':body.splitlines(keepends=True),
            'outputs':[], 'execution_count':None,
        }
    install = '%pip install -q --only-binary=:all: "xgboost==3.1.3"'
    source['cells'] = [
        {'cell_type':'markdown','id':'intro','metadata':{},'source':[f'# {name} · 10 phương pháp STUDD']},
        make_cell(install, 'dependencies'), setup, load, make_cell(run_cell, 'evaluation'),
    ]
    (target / notebook.name).write_text(json.dumps(source,ensure_ascii=False))
    sources = ['phmchnkhoa/studd-arabic-benchmark-inputs'] if slug in ('arabic','arabic-shuffled') else []
    metadata = {
        'id':f'phmchnkhoa/studd-{slug}-10-methods-khoa',
        'title':f'STUDD {slug} 10 Methods Khoa', 'code_file':notebook.name,
        'language':'python', 'kernel_type':'notebook', 'is_private':True,
        'enable_gpu':False, 'enable_tpu':False, 'enable_internet':True,
        'dataset_sources':sources, 'competition_sources':[], 'kernel_sources':[],
    }
    (target/'kernel-metadata.json').write_text(json.dumps(metadata))
    print(slug, name)
