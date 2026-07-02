# scripts/evaluate_retrieval.py
#
# Measures retrieval quality: hit-rate@1/3/5 and MRR on a hand-written set of
# in-domain questions with known-correct source pages, plus a rejection-rate
# check on out-of-domain questions (using the same distance threshold the
# live pipeline uses to decide when to decline rather than hallucinate).
#
# Only exercises the retriever, not the LLM, so it runs in seconds.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retriever import Retriever
from rag.pipeline import MAX_RELEVANT_DISTANCE
from eval_dataset import IN_DOMAIN, OUT_OF_DOMAIN

K = 5


def evaluate_in_domain(retriever: Retriever):
    hits_at = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []
    rows = []

    for question, expected_url in IN_DOMAIN:
        results = retriever.search(question, k=K)
        urls = [r["url"] for r in results]

        rank = urls.index(expected_url) + 1 if expected_url in urls else None
        reciprocal_ranks.append(1 / rank if rank else 0)

        for k in hits_at:
            if rank and rank <= k:
                hits_at[k] += 1

        rows.append((question, expected_url, rank))

    n = len(IN_DOMAIN)
    print("=== Retrieval quality (in-domain) ===")
    print(f"{'Question':<65} {'Rank':<6}")
    for question, expected_url, rank in rows:
        status = f"#{rank}" if rank else "MISS"
        print(f"{question[:63]:<65} {status:<6}")

    print()
    for k, hits in hits_at.items():
        print(f"Hit-rate@{k}: {hits}/{n} ({hits / n:.0%})")
    print(f"MRR: {sum(reciprocal_ranks) / n:.3f}")
    print()


def evaluate_out_of_domain(retriever: Retriever):
    correct_declines = 0
    print("=== Rejection quality (out-of-domain) ===")
    for question in OUT_OF_DOMAIN:
        results = retriever.search(question, k=1)
        best_distance = results[0]["distance"] if results else float("inf")
        would_decline = best_distance > MAX_RELEVANT_DISTANCE
        correct_declines += would_decline
        print(f"{question[:60]:<62} distance={best_distance:.2f}  {'DECLINED (correct)' if would_decline else 'ANSWERED (false positive)'}")

    n = len(OUT_OF_DOMAIN)
    print()
    print(f"Correct rejection rate: {correct_declines}/{n} ({correct_declines / n:.0%})")


def main():
    retriever = Retriever()
    evaluate_in_domain(retriever)
    evaluate_out_of_domain(retriever)


if __name__ == "__main__":
    main()
