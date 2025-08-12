import json
from pathlib import Path

from server.agent.rag_manager import rag_manager


CENTROIDS_PATH = Path("server/agent/packs/pack_centroids.json")

def build_pack_centroids(packs_root="learning/packs"):
    centroids = {}
    for pack_dir in sorted(Path(packs_root).iterdir()):
        nodes_path = pack_dir / "nodes.jsonl"
        if not nodes_path.exists(): 
            continue
        texts = []
        with nodes_path.open("r", encoding="utf-8") as f:
            for line in f:
                node = json.loads(line)
                name = node.get("name","")
                summ = (node.get("summary") or "")[:300]
                texts.append(f"{name}\n{summ}")
        if not texts: 
            continue
        emb = rag_manager.model.encode(texts)  # (N, dim)
        centroid = emb.mean(axis=0).tolist()
        centroids[str(pack_dir)] = {"centroid": centroid}
    CENTROIDS_PATH.write_text(json.dumps(centroids), encoding="utf-8")
    return centroids

def load_centroids():
    if CENTROIDS_PATH.exists():
        return json.loads(CENTROIDS_PATH.read_text(encoding="utf-8"))
    return build_pack_centroids()