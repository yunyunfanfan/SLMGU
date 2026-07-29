"""
Download heterogeneous graph datasets to ./data/
Datasets: LastFM, MovieLens-1M, OGB-MAG, NELL-995, Hetionet, DBpedia50k
"""
import os
import sys
import zipfile
import tarfile
import requests
from pathlib import Path

DATA_DIR = Path('./data')
DATA_DIR.mkdir(exist_ok=True)


def download_file(url: str, dest: Path, desc: str = '') -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f'  [skip] {dest.name} already exists')
        return True
    print(f'  Downloading {desc or dest.name} ...')
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f'\r  {pct:.1f}%', end='', flush=True)
        print()
        return True
    except Exception as e:
        print(f'\n  ERROR: {e}')
        if dest.exists():
            dest.unlink()
        return False


# ─────────────────────────────────────────────────────────
# 1. LastFM  (PyG built-in, HeteroData)
# ─────────────────────────────────────────────────────────
def download_lastfm():
    print('\n[1/6] LastFM (PyG built-in)')
    from torch_geometric.datasets import LastFM
    try:
        dataset = LastFM(root=str(DATA_DIR / 'LastFM'))
        data = dataset[0]
        print(f'  Done. {data}')
    except Exception as e:
        print(f'  ERROR: {e}')


# ─────────────────────────────────────────────────────────
# 2. MovieLens-1M  (PyG built-in, HeteroData)
# ─────────────────────────────────────────────────────────
def download_movielens1m():
    print('\n[2/6] MovieLens-1M (PyG built-in)')
    from torch_geometric.datasets import MovieLens1M
    try:
        dataset = MovieLens1M(root=str(DATA_DIR / 'MovieLens-1M'))
        data = dataset[0]
        print(f'  Done. {data}')
    except Exception as e:
        print(f'  ERROR: {e}')


# ─────────────────────────────────────────────────────────
# 3. OGB-MAG  (PyG built-in via OGB, HeteroData)
# ─────────────────────────────────────────────────────────
def download_ogb_mag():
    print('\n[3/6] OGB-MAG (PyG built-in via OGB)')
    from torch_geometric.datasets import OGB_MAG
    try:
        dataset = OGB_MAG(root=str(DATA_DIR / 'OGB-MAG'))
        data = dataset[0]
        print(f'  Done. {data}')
    except Exception as e:
        print(f'  ERROR: {e}')


# ─────────────────────────────────────────────────────────
# 4. NELL-995  (KG completion benchmark, raw triples)
#    Source: GraIL paper's public GitHub data release
# ─────────────────────────────────────────────────────────
def download_nell995():
    print('\n[4/6] NELL-995 (KG completion benchmark)')
    base_dir = DATA_DIR / 'NELL-995'
    base_dir.mkdir(exist_ok=True)

    # Files distributed with the GraIL / Neural LP paper benchmarks
    base_url = ('https://raw.githubusercontent.com/kkteru/grail/'
                'master/data/nell_v1/')
    files = ['train.txt', 'valid.txt', 'test.txt',
             'train2id.txt', 'valid2id.txt', 'test2id.txt',
             'entities.dict', 'relations.dict']

    success = True
    for fname in files:
        ok = download_file(base_url + fname, base_dir / fname, fname)
        if not ok:
            success = False

    if not success:
        # Fallback: try alternative GitHub source
        print('  Trying fallback source ...')
        alt_url = ('https://raw.githubusercontent.com/wencolani/GraIL/'
                   'master/data/nell-995/')
        for fname in ['train.txt', 'valid.txt', 'test.txt']:
            download_file(alt_url + fname, base_dir / fname, fname)

    present = list(base_dir.iterdir())
    print(f'  Files in {base_dir}: {[p.name for p in present]}')


# ─────────────────────────────────────────────────────────
# 5. Hetionet  (biomedical heterogeneous network)
#    Source: official hetionet GitHub release
# ─────────────────────────────────────────────────────────
def download_hetionet():
    print('\n[5/6] Hetionet')
    base_dir = DATA_DIR / 'Hetionet'
    base_dir.mkdir(exist_ok=True)

    # Nodes & edges TSV files from the official hetionet release
    release_base = ('https://github.com/hetio/hetionet/raw/main/'
                    'hetnet/tsv/')
    nodes_url = release_base + 'hetionet-v1.0-nodes.tsv'
    edges_url = release_base + 'hetionet-v1.0-edges.sif.gz'

    ok1 = download_file(nodes_url, base_dir / 'nodes.tsv', 'nodes.tsv')
    ok2 = download_file(edges_url, base_dir / 'edges.sif.gz', 'edges.sif.gz')

    if ok2 and (base_dir / 'edges.sif.gz').exists():
        import gzip, shutil
        out = base_dir / 'edges.sif'
        if not out.exists():
            print('  Decompressing edges.sif.gz ...')
            with gzip.open(base_dir / 'edges.sif.gz', 'rb') as fin, \
                 open(out, 'wb') as fout:
                shutil.copyfileobj(fin, fout)
            print('  Decompressed.')

    present = list(base_dir.iterdir())
    print(f'  Files in {base_dir}: {[p.name for p in present]}')


# ─────────────────────────────────────────────────────────
# 6. DBpedia50k  (KG completion benchmark, ~50k entities)
#    Source: Shi & Weninger (2018) "Open World KG Completion"
# ─────────────────────────────────────────────────────────
def download_dbpedia50k():
    print('\n[6/6] DBpedia50k')
    base_dir = DATA_DIR / 'DBpedia50k'
    base_dir.mkdir(exist_ok=True)

    base_url = ('https://raw.githubusercontent.com/bxshi/'
                'ConMask/master/data/dbpedia50/')
    files = ['train.txt', 'valid.txt', 'test.txt']

    success = True
    for fname in files:
        ok = download_file(base_url + fname, base_dir / fname, fname)
        if not ok:
            success = False

    if not success:
        # Fallback: alternative repo
        print('  Trying fallback source ...')
        alt_base = ('https://raw.githubusercontent.com/smartschat/'
                    'cort/master/')
        for fname in files:
            download_file(alt_base + fname, base_dir / fname, fname)

    present = list(base_dir.iterdir())
    print(f'  Files in {base_dir}: {[p.name for p in present]}')


if __name__ == '__main__':
    print('=== Downloading heterogeneous graph datasets to ./data/ ===')
    download_lastfm()
    download_movielens1m()
    # download_ogb_mag()  # skipped (2GB, download separately if needed)
    download_nell995()
    download_hetionet()
    download_dbpedia50k()
    print('\n=== Done ===')
    print('Summary of ./data/:')
    for p in sorted(DATA_DIR.iterdir()):
        if p.is_dir():
            n = sum(1 for _ in p.rglob('*') if _.is_file())
            print(f'  {p.name:25s}  ({n} files)')
