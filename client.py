from typing import Dict, Any, List, Optional

class VectorGraphHybridRRFFusion:
    """
    Merges disparate retrieval ranking lists (dense vector semantic search vs graph hop proximity)
    using the Reciprocal Rank Fusion formula: Score(d) = sum(1 / (k + rank(d))).
    """
    def fuse_rankings(
        self,
        vector_ranked_ids: List[str],
        graph_ranked_ids: List[str],
        k: int = 60
    ) -> Dict[str, Any]:
        rrf_scores: Dict[str, float] = {}
        all_ids = set(vector_ranked_ids).union(set(graph_ranked_ids))

        for doc_id in all_ids:
            score = 0.0
            v_rank = vector_ranked_ids.index(doc_id) + 1 if doc_id in vector_ranked_ids else None
            g_rank = graph_ranked_ids.index(doc_id) + 1 if doc_id in graph_ranked_ids else None

            if v_rank is not None:
                score += 1.0 / (k + v_rank)
            if g_rank is not None:
                score += 1.0 / (k + g_rank)

            rrf_scores[doc_id] = round(score, 6)

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_ranked_list = [
            {
                "rank": idx + 1,
                "document_id": doc_id,
                "rrf_score": score,
                "vector_rank": vector_ranked_ids.index(doc_id) + 1 if doc_id in vector_ranked_ids else None,
                "graph_rank": graph_ranked_ids.index(doc_id) + 1 if doc_id in graph_ranked_ids else None
            }
            for idx, (doc_id, score) in enumerate(sorted_results)
        ]

        return {
            "total_candidates": len(all_ids),
            "smoothing_constant_k": k,
            "fused_rankings": final_ranked_list
        }
