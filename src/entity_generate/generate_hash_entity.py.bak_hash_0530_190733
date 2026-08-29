import json
import hashlib
import os
import argparse
import sys
import re
from collections import Counter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.utils import load_json, save_json

class EntityProcessor:
    def __init__(self):
        self.entity_types = {}
        self.entities = {}
        self.relation_types = {}
        self.relation_list = []

    def get_top_relation_types(self, k=20):
        relation_types = sorted(self.relation_types.items(), key=lambda item: item[1], reverse=True)
        return [item[0] for item in relation_types[:k]]

    def valid_entity(self, e):
        if not isinstance(e, str):
            return False
        t = e.strip()
        if not t:
            return False
        tl = t.lower()

        bad = [
            "john doe", "university of science", "mathematics",
            "single-family", "lyapunov", "sarcoma uk",
            "beyond its borders", "real subjects",
            "operational efficiency", "amount of training data",
            "climate change", "anthropogenic activity",
            "university of example", "reformist seasons"
        ]
        if any(k in tl for k in bad):
            return False

        if re.fullmatch(r"[\d,\.\-\s]+", t):
            return False
        if re.fullmatch(r"[\d,\.\-\s]+died", tl):
            return False
        if len(t) < 3 or len(t) > 100:
            return False

        return True

    def get_top_entities_from_relation_list(self, top_k=2000, min_freq=2):
        cnt = Counter()
        for x in self.relation_list:
            if not isinstance(x, list) or len(x) < 3:
                continue
            h, t, r = x[0], x[1], x[2]
            if self.valid_entity(h):
                cnt[h] += 1
            if self.valid_entity(t):
                cnt[t] += 1
        out = [e for e, c in cnt.most_common() if c >= min_freq]
        return out[:top_k]

    def get_hash_init(self):
        return os.urandom(32).hex()

    def hash_entity(self, filtered_entities, hash_init, hash_num=100):
        if not filtered_entities:
            raise ValueError("filtered_entities is empty")

        entity_hash_list = []
        seen = set()
        current = hashlib.sha256(hash_init.encode("utf-8")).hexdigest()
        tries = 0
        max_tries = hash_num * 50

        while len(entity_hash_list) < min(hash_num, len(filtered_entities)) and tries < max_tries:
            idx = int(current, 16) % len(filtered_entities)
            ent = filtered_entities[idx]
            if ent not in seen:
                seen.add(ent)
                entity_hash_list.append(ent)
            current = hashlib.sha256((current + ent + hash_init + str(tries)).encode("utf-8")).hexdigest()
            tries += 1

        return entity_hash_list

def deterministic_edge(s1, t1, hash_init, edge_prob):
    h = hashlib.sha256(("EDGE||" + s1 + "||" + t1 + "||" + str(hash_init)).encode("utf-8")).hexdigest()
    score = int(h[:12], 16) / float(16**12 - 1)
    return score < edge_prob

def hash_wmunit(entity_hash_list, relation_type_list, hash_init, edge_prob):
    relation_total = len(relation_type_list)

    def get_wmunit(s1, t1):
        h = hashlib.sha256(("REL||" + s1 + "||" + t1 + "||" + str(hash_init)).encode("utf-8")).hexdigest()
        return relation_type_list[int(h, 16) % relation_total]

    wmunit_list = []
    for i in range(len(entity_hash_list)):
        for j in range(i + 1, len(entity_hash_list)):
            s1 = entity_hash_list[i]
            t1 = entity_hash_list[j]
            if deterministic_edge(s1, t1, hash_init, edge_prob):
                wmunit_list.append((s1, t1, get_wmunit(s1, t1)))

    print(f"Total watermark units: {len(wmunit_list)}")
    return wmunit_list

def scratch_run():
    entityprocessor = EntityProcessor()
    entityprocessor.entities = load_json(entities_dict_path)
    entityprocessor.entity_types = load_json(entity_type_path)
    entityprocessor.relation_types = load_json(relation_type_path)
    entityprocessor.relation_list = load_json(relation_list_path)

    filter_entity = entityprocessor.get_top_entities_from_relation_list(
        top_k=args.entity_pool_size,
        min_freq=args.min_entity_freq
    )
    print("filter_entity", len(filter_entity))
    save_json(filter_entity, entity_top_list_path)

    hash_init = entityprocessor.get_hash_init()
    save_json(hash_init, init_hash_path)

    entity_hash = entityprocessor.hash_entity(filter_entity, hash_init, hash_num=args.entity_num)
    save_json(entity_hash, entity_hash_path)

    relation_top = entityprocessor.get_top_relation_types(k=20)
    save_json(relation_top, relation_top_path)

    wmunit_list = hash_wmunit(entity_hash, relation_top, hash_init, args.edge_prob)
    save_json(wmunit_list, watermark_unit_path)

def wmunit_run():
    entity_hash = load_json(entity_hash_path)
    relation_top = load_json(relation_top_path)
    hash_init = load_json(init_hash_path)
    wmunit_list = hash_wmunit(entity_hash, relation_top, hash_init, args.edge_prob)
    save_json(wmunit_list, watermark_unit_path)

def check_run():
    relation_list = load_json(relation_list_path)
    wmunit_list = load_json(watermark_unit_path)
    for wmunit in wmunit_list:
        if tuple(wmunit) in relation_list:
            print(f"{wmunit} exists in the relation list")

def cal_run():
    relation_list = load_json(relation_list_path)
    entity_list = load_json(entities_dict_path)
    print(len(relation_list), len(entity_list))

def parse_args():
    parser = argparse.ArgumentParser(description="Generate watermark unit list")
    parser.add_argument("-t", choices=["scratch", "wmunit", "check", "cal"], default="wmunit")
    parser.add_argument("--eval_dataset", type=str, default="trec-covid",
                        choices=["trec-covid", "nfcorpus", "nq", "msmarco", "hotpotqa"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--basepath", type=str, default="/workspace/ragwm/ragwm/output/wm_prepare")
    parser.add_argument("--entity_num", type=int, default=100)
    parser.add_argument("--edge_prob", type=float, default=0.05)
    parser.add_argument("--entity_pool_size", type=int, default=2000)
    parser.add_argument("--min_entity_freq", type=int, default=2)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    basepath = args.basepath

    entities_dict_path = os.path.join(basepath, args.eval_dataset, "entities_dict_llm.json")
    entity_type_path = os.path.join(basepath, args.eval_dataset, "entity_type_llm.json")
    relation_type_path = os.path.join(basepath, args.eval_dataset, "relation_type_llm.json")
    relation_list_path = os.path.join(basepath, args.eval_dataset, "relation_list_llm.json")

    entity_top_path = os.path.join(basepath, args.eval_dataset, "entity_top.json")
    entity_top_list_path = os.path.join(basepath, args.eval_dataset, "entity_top_list.json")
    relation_top_path = os.path.join(basepath, args.eval_dataset, "relation_top.json")
    init_hash_path = os.path.join(basepath, args.eval_dataset, "init_hash.json")
    entity_hash_path = os.path.join(basepath, args.eval_dataset, "entity_hash.json")
    watermark_unit_path = os.path.join(basepath, args.eval_dataset, "wmunit.json")

    if args.t == "scratch":
        scratch_run()
    elif args.t == "wmunit":
        wmunit_run()
    elif args.t == "check":
        check_run()
    elif args.t == "cal":
        cal_run()
