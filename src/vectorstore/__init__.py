"""Text retrieval over document chunks.

    build  embed chunks.jsonl -> an on-disk index
    index  LocalVectorIndex (numpy, default), TigerGraphVectorIndex (chunk_emb
           vector attribute), and a hashing fallback for offline CI

The local index is what the benchmark runs on: the TigerGraph vector attribute
was unavailable on our Savanna workspace, so the deploy step probes for it and
skips cleanly rather than failing. See docs/FEEDBACK.md #11.
"""
