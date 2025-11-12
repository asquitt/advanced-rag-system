"""
Compare different retrieval strategies.
"""
from src.retrieval.retriever import BasicRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


def main():
    # Sample documents
    docs = [
        "Python programming language with dynamic typing and extensive libraries",
        "Machine learning uses statistical methods to enable systems to learn",
        "Deep learning neural networks with multiple layers for complex patterns",
        "Natural language processing analyzes and understands human language",
        "Computer vision processes and analyzes visual information from images",
    ]

    query = "machine learning systems"

    print(f"Query: {query}\n")
    print("=" * 70)

    # Dense only
    print("\n1. DENSE RETRIEVAL (Vector similarity)")
    dense = BasicRetriever(collection_name="compare_dense")
    dense.index(docs)
    results = dense.retrieve(query, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"   {i}. ({r['score']:.3f}) {r['text'][:60]}...")

    # Hybrid
    print("\n2. HYBRID RETRIEVAL (Dense + BM25)")
    hybrid = HybridRetriever(collection_name="compare_hybrid", dense_weight=0.5)
    hybrid.index(docs)
    results = hybrid.retrieve(query, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"   {i}. ({r['score']:.3f}) {r['text'][:60]}...")

    print("\n" + "=" * 70)
    print(
        "\n💡 Hybrid typically performs best by combining semantic and keyword matching"
    )


if __name__ == "__main__":
    main()
