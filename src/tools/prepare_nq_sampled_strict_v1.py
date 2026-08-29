import json
import random
import shutil
from pathlib import Path

SEED = 2026
TARGET_DOCS = 20000

RAW_ROOT = Path("/root/autodl-tmp/ragwm_storage/beir_downloads")
SUBSET_ROOT = Path("/root/autodl-tmp/ragwm_storage/datasets/nq_sampled_strict_v1")
EXPECTED_ROOT = Path("/data/sunmengjie/lpz/ragwm/datasets/nq")

random.seed(SEED)

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    SUBSET_ROOT.mkdir(parents=True, exist_ok=True)

    raw_nq = RAW_ROOT / "nq"
    corpus_path = raw_nq / "corpus.jsonl"
    queries_path = raw_nq / "queries.jsonl"
    qrels_path = raw_nq / "qrels" / "test.tsv"

    if not corpus_path.exists():
        print("Raw NQ not found. Downloading BEIR NQ...")
        try:
            from beir import util
            url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"
            util.download_and_unzip(url, str(RAW_ROOT))
        except Exception as e:
            raise RuntimeError(
                "BEIR NQ download failed. Check network or manually place nq under "
                f"{raw_nq}. Original error: {repr(e)}"
            )

    assert corpus_path.exists(), f"missing {corpus_path}"
    assert queries_path.exists(), f"missing {queries_path}"
    assert qrels_path.exists(), f"missing {qrels_path}"

    print("Reading qrels...")
    qrel_doc_ids = set()
    qrel_query_ids = set()
    with open(qrels_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                qid, docid = parts[0], parts[1]
                qrel_query_ids.add(qid)
                qrel_doc_ids.add(docid)

    print("qrel docs:", len(qrel_doc_ids))
    print("qrel queries:", len(qrel_query_ids))

    selected = {}
    reservoir = []

    print("Scanning corpus...")
    for obj in read_jsonl(corpus_path):
        docid = obj.get("_id")
        if docid in qrel_doc_ids:
            selected[docid] = obj
        else:
            reservoir.append(obj)

    print("selected from qrels:", len(selected))
    need = max(0, TARGET_DOCS - len(selected))
    if need > 0:
        print("random fill:", need)
        fill = random.sample(reservoir, min(need, len(reservoir)))
        for obj in fill:
            selected[obj["_id"]] = obj

    selected_docs = list(selected.values())
    random.shuffle(selected_docs)

    # filter qrels to selected docs
    selected_ids = {x["_id"] for x in selected_docs}

    print("Writing subset corpus:", len(selected_docs))
    write_jsonl(SUBSET_ROOT / "corpus.jsonl", selected_docs)

    # keep queries that appear in qrels; this is enough for compatibility
    queries = []
    for obj in read_jsonl(queries_path):
        if obj.get("_id") in qrel_query_ids:
            queries.append(obj)
    write_jsonl(SUBSET_ROOT / "queries.jsonl", queries)

    out_qrels = SUBSET_ROOT / "qrels" / "test.tsv"
    out_qrels.parent.mkdir(parents=True, exist_ok=True)
    kept_qrels = 0
    with open(qrels_path, "r", encoding="utf-8") as fin, open(out_qrels, "w", encoding="utf-8") as fout:
        header = fin.readline()
        fout.write(header if header.strip() else "query-id\tcorpus-id\tscore\n")
        for line in fin:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[1] in selected_ids:
                fout.write(line)
                kept_qrels += 1

    print("queries:", len(queries))
    print("kept qrels:", kept_qrels)

    EXPECTED_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if EXPECTED_ROOT.exists() or EXPECTED_ROOT.is_symlink():
        backup = EXPECTED_ROOT.parent / f"nq_backup_before_sampled_strict_v1"
        if not backup.exists():
            EXPECTED_ROOT.rename(backup)
            print("old nq moved to:", backup)
        else:
            if EXPECTED_ROOT.is_symlink():
                EXPECTED_ROOT.unlink()
            else:
                shutil.rmtree(EXPECTED_ROOT)

    EXPECTED_ROOT.symlink_to(SUBSET_ROOT, target_is_directory=True)
    print("symlink created:", EXPECTED_ROOT, "->", SUBSET_ROOT)

    manifest = {
        "version": "nq_sampled_strict_v1",
        "seed": SEED,
        "target_docs": TARGET_DOCS,
        "actual_docs": len(selected_docs),
        "qrel_docs_included_first": len(qrel_doc_ids),
        "queries": len(queries),
        "kept_qrels": kept_qrels,
        "subset_root": str(SUBSET_ROOT),
        "expected_project_path": str(EXPECTED_ROOT),
        "note": "NQ sampled strict v1 keeps qrel-linked documents first, then randomly fills to 20k docs. It is designed for open-domain generalization testing, not full-scale NQ replication."
    }
    json.dump(manifest, open(SUBSET_ROOT / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
