from utils.embeddings import get_embedding_service
from utils.qdrant_setup import get_qdrant_client, COLLECTION_NAME
from typing import List, Dict, Set, Tuple
import re

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

STOPWORDS = {
    'what', 'is', 'the', 'of', 'in', 'a', 'an', 'for', 'and', 'or',
    'how', 'does', 'do', 'explain', 'describe', 'tell', 'about', 'me',
    'with', 'its', 'their', 'this', 'that', 'are', 'which', 'where',
    'when', 'why', 'give', 'show', 'list', 'on', 'by', 'at', 'to',
    'from', 'was', 'were', 'has', 'have', 'had', 'be', 'been', 'being'
}

# Abbreviation map: expands folder abbreviations so "LUB" matches "lubrication"
ABBREV = {
    'sys': ['system'],
    'susp': ['suspension'],
    'lub': ['lubrication', 'lubrication'],
    'txn': ['transmission'],
    'eng': ['engine'],
    'ign': ['ignition'],
    'constr': ['construction'],
    'maint': ['maintenance'],
    'prev': ['preventive'],
    'brk': ['brake'],
    'fuel': ['fuel'],
    'cool': ['cooling'],
    'steer': ['steering'],
    'mc': ['motorcycle'],
    'spl': ['special'],
    'intr': ['introduction'],
    'dtls': ['details'],
    'adj': ['adjustment'],
    'cyl': ['cylinder'],
    'comp': ['compression'],
    'elec': ['electrical', 'electric'],
    'elect': ['electrical', 'electric'],
    'trg': ['training'],
    'smts': ['smts'],
    're': ['royal', 'enfield'],
    'hh': ['hero', 'honda'],
    'mg': ['mg'],
    'wm': ['workshop', 'manual'],
}


# ─────────────────────────────────────────────────────────────
# TOKEN HELPERS
# ─────────────────────────────────────────────────────────────

def tokenize(text: str) -> Set[str]:
    """Extract lowercase alphanumeric tokens, skipping stopwords and short tokens."""
    tokens = re.findall(r'[A-Za-z0-9]+', text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


def expand_tokens(tokens: Set[str]) -> Set[str]:
    """
    Expand abbreviated tokens using the ABBREV map so folder abbreviations
    like 'SYS', 'LUB', 'SUSP' can match query words like 'system',
    'lubrication', 'suspension'.
    Returns the original tokens PLUS their expansions.
    """
    expanded = set(tokens)
    for t in tokens:
        if t in ABBREV:
            expanded.update(ABBREV[t])
        # Also check if a query word is the EXPANDED form of an abbreviation
        for abbr, expansions in ABBREV.items():
            if t in expansions:
                expanded.add(abbr)
    return expanded


def path_tokens(path_part: str) -> Set[str]:
    """Tokenize a path component and expand abbreviations."""
    raw = tokenize(path_part)
    return expand_tokens(raw)


# ─────────────────────────────────────────────────────────────
# TWO-LEVEL PATH SCORING
# ─────────────────────────────────────────────────────────────

def parse_path_levels(source: str) -> Tuple[str, str, str]:
    """
    Split a source path into (vehicle_folder, system_folder, filename).

    The folder structure is:
        .../root/N. Vehicle Name/N. System Name/filename.ext

    'Numbered folders' (starting with digit + dot) are the discriminating
    hierarchy levels. Works for any depth — finds first and second numbered
    folder in path.
    """
    parts = [p for p in source.replace('\\', '/').split('/') if p]
    numbered = [p for p in parts if re.match(r'^\d+[.\s]', p)]

    vehicle_folder = numbered[0] if len(numbered) > 0 else ""
    system_folder  = numbered[1] if len(numbered) > 1 else ""
    filename       = parts[-1] if parts else ""

    return vehicle_folder, system_folder, filename


def score_source(source: str, query_tok: Set[str]) -> Tuple[int, int]:
    """
    Returns (vehicle_score, system_score) for a source path vs query tokens.

    vehicle_score = overlap between query tokens and the vehicle-folder tokens
    system_score  = overlap between query tokens and system-folder + filename tokens

    Abbreviation expansion is applied to path tokens so e.g. "SUSP SYS" expands
    to include "suspension" and "system", matching queries that use those words.
    """
    vehicle_folder, system_folder, filename = parse_path_levels(source)

    v_tok = path_tokens(vehicle_folder)
    s_tok = path_tokens(system_folder + " " + filename)

    v_score = len(query_tok & v_tok)
    s_score = len(query_tok & s_tok)

    return v_score, s_score


# ─────────────────────────────────────────────────────────────
# SOURCE-TARGETED SEARCH
# ─────────────────────────────────────────────────────────────

def source_targeted_search(query: str, top_k_per_source: int = 10) -> List[Dict]:
    """
    Directly inject chunks from files whose paths best match the query,
    using two-level (vehicle + system) scoring with abbreviation expansion.

    Algorithm:
    1. Score every ingested source path against query tokens.
    2. If any source has vehicle_score >= 2, restrict to ONLY the best-
       matching vehicle (hard vehicle filter).  This prevents cross-vehicle
       contamination (e.g. SAFARI files appearing for MG 413W queries).
    3. Within the selected vehicle, pick the top-3 sources by system_score
       so the right system subfolder wins.
    4. Fetch their chunks and return them for the reranker to evaluate.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    query_tok = expand_tokens(tokenize(query))
    if not query_tok:
        return []

    client = get_qdrant_client()

    # ── collect all unique source paths ──────────────────────
    all_sources: Set[str] = set()
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=["source"]
        )
        for r in results:
            if r.payload and "source" in r.payload:
                all_sources.add(r.payload["source"])
        if offset is None:
            break

    # ── score every source ────────────────────────────────────
    scored: List[Tuple[str, int, int]] = []   # (source, v_score, s_score)
    for source in all_sources:
        v, s = score_source(source, query_tok)
        if v + s > 0:
            scored.append((source, v, s))

    if not scored:
        return []

    max_v = max(x[1] for x in scored)

    # ── vehicle-level hard filter ─────────────────────────────
    if max_v >= 2:
        # Strong vehicle signal: keep ONLY sources from the best-matching vehicle.
        # Find the vehicle folder of the top-scoring source, then filter to all
        # sources that share that same vehicle folder.
        top_vehicle_sources = [x for x in scored if x[1] == max_v]
        top_vehicle_sources.sort(key=lambda x: x[2], reverse=True)

        # Identify the winning vehicle folder
        winning_vehicle_folder, _, _ = parse_path_levels(top_vehicle_sources[0][0])

        # Keep ALL sources under that vehicle folder (they will be further
        # ranked by system_score)
        vehicle_candidates = [
            x for x in scored
            if parse_path_levels(x[0])[0] == winning_vehicle_folder
        ]
        vehicle_candidates.sort(key=lambda x: x[2], reverse=True)
        selected = vehicle_candidates[:3]

    else:
        # No strong vehicle discriminator: rank by total score
        scored.sort(key=lambda x: x[1] + x[2], reverse=True)
        # Require at least 2 total matches
        selected = [x for x in scored if x[1] + x[2] >= 2][:3]

    if not selected:
        return []

    print(f"🎯 Targeted sources: {[(s[0].split('/')[-2]+'/'+s[0].split('/')[-1], s[1], s[2]) for s in selected]}")

    # ── fetch chunks from selected sources ────────────────────
    chunks: List[Dict] = []
    for source, v_score, s_score in selected:
        base_score = min(0.95, 0.45 + 0.1 * (v_score + s_score))
        results, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=top_k_per_source,
            with_payload=True,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            )
        )
        for r in results:
            text = r.payload.get("text", "").strip()
            if text:
                chunks.append({
                    "text": text,
                    "source": r.payload["source"],
                    "chunk_index": r.payload.get("chunk_index", 0),
                    "score": base_score,
                    "rrf_score": base_score,
                    "reranker_score": base_score,
                    "retriever": "path_match"
                })

    return chunks


# ─────────────────────────────────────────────────────────────
# DENSE SEARCH
# ─────────────────────────────────────────────────────────────

def dense_search(query: str, top_k: int = 10) -> List[Dict]:
    embedding_service = get_embedding_service()
    query_vector = embedding_service.embed_query(query)

    client = get_qdrant_client()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    return [
        {
            "text": r.payload["text"],
            "source": r.payload["source"],
            "chunk_index": r.payload["chunk_index"],
            "score": r.score,
            "rrf_score": r.score,
            "retriever": "dense"
        }
        for r in results
    ]


# ─────────────────────────────────────────────────────────────
# RRF + PATH BOOSTS
# ─────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(result_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    scores: dict = {}
    texts: dict = {}
    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            key = result["text"][:100]
            if key not in scores:
                scores[key] = 0.0
                texts[key] = result
            scores[key] += 1.0 / (k + rank + 1)
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    fused = []
    for key in sorted_keys:
        item = texts[key].copy()
        item["rrf_score"] = scores[key]
        fused.append(item)
    return fused


def path_boost_after_rerank(query: str, chunks: List[Dict], boost: float = 0.6) -> List[Dict]:
    """
    Apply two-level path scoring on top of reranker scores.

    Hard vehicle filter: if ANY chunk scores vehicle_match >= 2, all chunks
    from wrong vehicles (vehicle_match < max) are removed entirely before
    re-sorting. This prevents high-text-score chunks from wrong vehicles
    (e.g. SAFARI manual scoring high on 'steering') from surviving.
    """
    query_tok = expand_tokens(tokenize(query))
    if not query_tok:
        return chunks

    # Score every chunk
    for chunk in chunks:
        v, s = score_source(chunk.get("source", ""), query_tok)
        chunk["_v"] = v
        chunk["_s"] = s

    max_v = max((c["_v"] for c in chunks), default=0)

    # Hard vehicle filter: only apply when there's a strong vehicle signal
    if max_v >= 2:
        chunks = [c for c in chunks if c["_v"] == max_v]

    # Boost remaining chunks by system-level relevance
    for chunk in chunks:
        multiplier = 1 + boost * (chunk["_v"] + chunk["_s"])
        base = chunk.get("reranker_score", chunk.get("rrf_score", chunk.get("score", 0)))
        chunk["reranker_score"] = base * multiplier

    return sorted(chunks, key=lambda x: x.get("reranker_score", 0), reverse=True)


# ─────────────────────────────────────────────────────────────
# MAIN RETRIEVE
# ─────────────────────────────────────────────────────────────

def extract_keywords(query: str) -> List[str]:
    """For backward compat with api/main.py imports."""
    return list(tokenize(query))


def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    print(f"🔍 Retrieving: {query}")

    # Genuinely diverse query variations
    kw_query   = " ".join(tokenize(query))
    tech_query = f"{query} components parts operation mechanism procedure"
    variations = list(dict.fromkeys(
        q for q in [query, kw_query, tech_query] if q.strip()
    ))

    all_lists = []
    for q in variations:
        all_lists.append(dense_search(q, top_k=top_k * 6))

    fused = reciprocal_rank_fusion(all_lists)

    # Inject source-targeted chunks (guarantees correct vehicle files in pool)
    targeted = source_targeted_search(query)
    if targeted:
        existing = {c["text"][:100] for c in fused}
        added = 0
        for chunk in targeted:
            if chunk["text"][:100] not in existing:
                fused.append(chunk)
                existing.add(chunk["text"][:100])
                added += 1
        print(f"🎯 Injected {added} targeted chunks")

    return fused[:top_k]


# ─────────────────────────────────────────────────────────────
# MULTI-QUERY RETRIEVE (question paper)
# ─────────────────────────────────────────────────────────────

def retrieve_multi(queries: List[str], top_k: int = 20) -> List[Dict]:
    embedding_service = get_embedding_service()
    vectors = embedding_service.model.encode_queries(queries)

    import numpy as np
    if isinstance(vectors, np.ndarray):
        vectors = vectors.tolist()

    client = get_qdrant_client()
    seen_texts: Set[str] = set()
    all_chunks: List[Dict] = []
    reference_query = queries[0] if queries else ""

    for vector in vectors:
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        vector = [float(x) for x in vector]

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k,
            with_payload=True
        ).points

        for r in results:
            text = r.payload["text"].strip()
            if text and text not in seen_texts:
                seen_texts.add(text)
                all_chunks.append({
                    "text": text,
                    "source": r.payload["source"],
                    "chunk_index": r.payload["chunk_index"],
                    "score": r.score,
                    "rrf_score": r.score,
                    "reranker_score": r.score,
                    "retriever": "dense"
                })

    if reference_query:
        targeted = source_targeted_search(reference_query)
        for chunk in targeted:
            if chunk["text"] not in seen_texts:
                seen_texts.add(chunk["text"])
                all_chunks.append(chunk)

    return all_chunks


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test path scoring without hitting Qdrant
    test_cases = [
        ("steering system for MG 413W",
         "/home/hpc25/Downloads/LVG/LVG/3. MG 413W MPFI/9. Steering sys/Field manual of MG 413 MPFI.pdf"),
        ("steering system for MG 413W",
         "/home/hpc25/Downloads/LVG/LVG/4. TATA SAFARI STORME UPDATE/7. SUSP SYS/SAFARI WM PART-I.pdf"),
        ("lubrication system Hero Honda CD 100",
         "/home/hpc25/Downloads/LVG/LVG/2. Hero Honda CD 100/3. Lubrication sys/Hero Honda Precie.pdf"),
        ("fuel system Royal Enfield 350",
         "/home/hpc25/Downloads/LVG/LVG/1. MC Royal Enfield TCI/3. FUEL SYS, DECOMP VALVE/Preci MCRE 350 TCI.pdf"),
    ]

    for query, source in test_cases:
        qtok = expand_tokens(tokenize(query))
        v, s = score_source(source, qtok)
        vf, sf, fn = parse_path_levels(source)
        print(f"\nQuery : {query}")
        print(f"Source: .../{vf}/{sf}/{fn}")
        print(f"Scores: vehicle={v}  system={s}  total={v+s}")
