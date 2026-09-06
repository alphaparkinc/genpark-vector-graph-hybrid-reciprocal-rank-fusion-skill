import json
from client import VectorGraphHybridRRFFusion

def main():
    fusion = VectorGraphHybridRRFFusion()
    vector_results = ["doc_A", "doc_B", "doc_C", "doc_D"]
    graph_results = ["doc_C", "doc_A", "doc_E"]
    
    res = fusion.fuse_rankings(vector_results, graph_results, k=60)
    print("Hybrid RRF Fusion Result:")
    print(json.dumps(res, indent=2))
    # doc_A and doc_C appear in both lists and should take top 2 spots
    top_two = [r["document_id"] for r in res["fused_rankings"][:2]]
    assert "doc_A" in top_two
    assert "doc_C" in top_two
    print("RRF fusion verification: PASS")

if __name__ == "__main__":
    main()
