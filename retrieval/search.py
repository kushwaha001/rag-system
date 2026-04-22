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

# Tokens that appear in almost every folder/filename and must NOT be the
# sole discriminator when selecting which system folder to target.
# e.g. "Fuel System" → "system"/"sys" match LUB SYS, BRAKE SYS, etc.
# Only SPECIFIC content words ("fuel", "lubrication", "brake"...) should pick the folder.
GENERIC_FOLDER_TOKENS = {'system', 'sys', 'manual', 'wm', 'workshop'}

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


def score_source(source: str, query_tok: Set[str]) -> Tuple[int, int, int]:
    """
    Returns (vehicle_score, system_score, sys_folder_specific) for a source path.

    vehicle_score      = overlap between query tokens and the vehicle-folder tokens
    system_score       = overlap between query tokens and (system-folder + filename) tokens
    sys_folder_specific = overlap between (query − generic − vehicle) tokens and the
                          system-folder tokens ONLY (filename excluded)

    Why sys_folder_specific matters: the same filename often appears in every
    system subfolder (e.g. `Field manual of MG 413 MPFI.pdf` is copied into
    FUEL SYS, LUB SYS, BRAKE SYS, etc.), so filename tokens inflate system_score
    equally across folders. sys_folder_specific ignores the filename and strips
    generic "system/sys/manual/workshop" tokens plus the vehicle tokens, leaving
    only the *content-specific* words ("fuel", "lubrication", "brake") that
    actually discriminate one system folder from another.
    """
    vehicle_folder, system_folder, filename = parse_path_levels(source)

    v_tok      = path_tokens(vehicle_folder)
    sys_tok    = path_tokens(system_folder)
    s_tok      = sys_tok | path_tokens(filename)

    v_score = len(query_tok & v_tok)
    s_score = len(query_tok & s_tok)

    # Specific score: drop generic folder words and the vehicle tokens from the
    # query, then intersect with the system-folder tokens only.
    specific_query = query_tok - GENERIC_FOLDER_TOKENS - v_tok
    specific_sys   = sys_tok   - GENERIC_FOLDER_TOKENS
    sys_specific = len(specific_query & specific_sys)

    return v_score, s_score, sys_specific


# ─────────────────────────────────────────────────────────────
# SOURCE-TARGETED SEARCH
# ─────────────────────────────────────────────────────────────

def source_targeted_search(query: str, top_k_per_source: int = 10, section_filter: str = None) -> List[Dict]:
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
    src_scroll_filter = None
    if section_filter:
        src_scroll_filter = Filter(must=[FieldCondition(key="section", match=MatchValue(value=section_filter))])

    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=["source"],
            scroll_filter=src_scroll_filter
        )
        for r in results:
            if r.payload and "source" in r.payload:
                all_sources.add(r.payload["source"])
        if offset is None:
            break

    # ── score every source ────────────────────────────────────
    scored: List[Tuple[str, int, int, int]] = []   # (source, v, s, sp)
    for source in all_sources:
        v, s, sp = score_source(source, query_tok)
        if v + s > 0:
            scored.append((source, v, s, sp))

    if not scored:
        return []

    max_v = max(x[1] for x in scored)

    # ── vehicle-level hard filter ─────────────────────────────
    if max_v >= 2:
        # Strong vehicle signal: keep ONLY sources from the best-matching vehicle.
        top_vehicle_sources = [x for x in scored if x[1] == max_v]
        top_vehicle_sources.sort(key=lambda x: (x[3], x[2]), reverse=True)

        # Identify the winning vehicle folder
        winning_vehicle_folder, _, _ = parse_path_levels(top_vehicle_sources[0][0])

        # Keep ALL sources under that vehicle folder
        vehicle_candidates = [
            x for x in scored
            if parse_path_levels(x[0])[0] == winning_vehicle_folder
        ]

        # Within this vehicle: if any source has a positive sys_folder_specific
        # match, HARD-filter to only those (e.g. "Fuel System" keeps FUEL SYS
        # and drops LUB SYS / BRAKE SYS / COOLING SYS even though they share
        # the same filename).
        max_sp = max((x[3] for x in vehicle_candidates), default=0)
        if max_sp > 0:
            vehicle_candidates = [x for x in vehicle_candidates if x[3] == max_sp]

        vehicle_candidates.sort(key=lambda x: (x[3], x[2]), reverse=True)
        selected = vehicle_candidates[:3]

    else:
        # No strong vehicle discriminator: rank by specific-match first, then total
        scored.sort(key=lambda x: (x[3], x[1] + x[2]), reverse=True)
        # Require either a specific system match OR >=2 total tokens
        selected = [x for x in scored if x[3] >= 1 or (x[1] + x[2]) >= 2][:3]

    if not selected:
        return []

    print(f"🎯 Targeted sources: {[(s[0].split('/')[-2]+'/'+s[0].split('/')[-1], s[1], s[2], s[3]) for s in selected]}")

    # ── fetch chunks from selected sources ────────────────────
    chunks: List[Dict] = []
    for source, v_score, s_score, sp_score in selected:
        base_score = min(0.95, 0.45 + 0.08 * (v_score + s_score) + 0.12 * sp_score)
        chunk_must = [FieldCondition(key="source", match=MatchValue(value=source))]
        if section_filter:
            chunk_must.append(FieldCondition(key="section", match=MatchValue(value=section_filter)))
        results, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=top_k_per_source,
            with_payload=True,
            scroll_filter=Filter(must=chunk_must)
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

def dense_search(query: str, top_k: int = 10, section_filter: str = None) -> List[Dict]:
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    embedding_service = get_embedding_service()
    query_vector = embedding_service.embed_query(query)

    client = get_qdrant_client()
    qfilter = None
    if section_filter:
        qfilter = Filter(must=[FieldCondition(key="section", match=MatchValue(value=section_filter))])

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        query_filter=qfilter
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
    Apply two-level path scoring on top of reranker scores, with two hard filters:

    1. Vehicle hard filter: if ANY chunk has v_score >= 2, drop all chunks
       from other vehicles. Prevents cross-vehicle contamination.

    2. System-folder hard filter: within the surviving vehicle, if ANY chunk
       has sys_folder_specific >= 1, drop all chunks with sp < max_sp.
       This discriminates folders whose filenames are identical (same manual
       copied into every system folder): only the folder whose NAME matches
       the content-specific query tokens survives.
    """
    query_tok = expand_tokens(tokenize(query))
    if not query_tok:
        return chunks

    # Score every chunk
    for chunk in chunks:
        v, s, sp = score_source(chunk.get("source", ""), query_tok)
        chunk["_v"]  = v
        chunk["_s"]  = s
        chunk["_sp"] = sp

    max_v = max((c["_v"] for c in chunks), default=0)

    # Hard vehicle filter
    if max_v >= 2:
        chunks = [c for c in chunks if c["_v"] == max_v]

    # Hard system-folder filter (within the surviving vehicle set)
    max_sp = max((c["_sp"] for c in chunks), default=0)
    if max_sp >= 1:
        chunks = [c for c in chunks if c["_sp"] == max_sp]

    # Boost remaining chunks — specific matches are worth more than raw overlap
    for chunk in chunks:
        multiplier = 1 + boost * (chunk["_v"] + chunk["_s"]) + 1.2 * boost * chunk["_sp"]
        base = chunk.get("reranker_score", chunk.get("rrf_score", chunk.get("score", 0)))
        chunk["reranker_score"] = base * multiplier

    return sorted(chunks, key=lambda x: x.get("reranker_score", 0), reverse=True)


# ─────────────────────────────────────────────────────────────
# MAIN RETRIEVE
# ─────────────────────────────────────────────────────────────

def extract_keywords(query: str) -> List[str]:
    """For backward compat with api/main.py imports."""
    return list(tokenize(query))


def retrieve(query: str, top_k: int = 5, section_filter: str = None) -> List[Dict]:
    print(f"🔍 Retrieving: {query}" + (f" [section={section_filter}]" if section_filter else ""))

    # Genuinely diverse query variations
    kw_query   = " ".join(tokenize(query))
    tech_query = f"{query} components parts operation mechanism procedure"
    variations = list(dict.fromkeys(
        q for q in [query, kw_query, tech_query] if q.strip()
    ))

    all_lists = []
    for q in variations:
        all_lists.append(dense_search(q, top_k=top_k * 6, section_filter=section_filter))

    fused = reciprocal_rank_fusion(all_lists)

    # Inject source-targeted chunks first so they are never cut off by top_k slice
    targeted = source_targeted_search(query, section_filter=section_filter)
    if targeted:
        existing_dense = {c["text"][:100] for c in fused}
        targeted_unique = [c for c in targeted if c["text"][:100] not in existing_dense]
        print(f"🎯 Injecting {len(targeted_unique)} targeted chunks (prepended)")
        targeted_keys = {c["text"][:100] for c in targeted_unique}
        dense_remainder = [c for c in fused if c["text"][:100] not in targeted_keys]
        combined = targeted_unique + dense_remainder
    else:
        combined = fused

    return combined[:top_k]


# ─────────────────────────────────────────────────────────────
# MULTI-QUERY RETRIEVE (question paper)
# ─────────────────────────────────────────────────────────────

def retrieve_multi(
    queries: List[str],
    top_k: int = 20,
    topic: str = "",
    selected_sources: List[str] = None,
) -> List[Dict]:
    """
    Multi-query retrieval with RRF fusion for question paper generation.

    Each query angle is embedded and searched independently; results are
    fused with RRF so chunks that rank well across multiple angles get a
    higher combined score.  Source-targeted chunks (path-matched) are
    prepended so they are never cut off by the top_k slice.

    If `selected_sources` is a non-empty list, retrieval is hard-restricted
    to only those source paths — path_match and dense search are both
    filtered, and if no chunks remain the top-k from a pure source scroll
    is returned as fallback.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

    embedding_service = get_embedding_service()
    vectors = embedding_service.model.encode_queries(queries)

    import numpy as np
    if isinstance(vectors, np.ndarray):
        vectors = vectors.tolist()

    client = get_qdrant_client()

    # Build Qdrant filter restricting to selected sources (OR across source paths)
    source_filter = None
    if selected_sources:
        source_filter = Filter(
            must=[FieldCondition(key="source", match=MatchAny(any=selected_sources))]
        )

    # Collect one result list per query angle
    per_query_lists: List[List[Dict]] = []
    for vector in vectors:
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        vector = [float(x) for x in vector]

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k,
            with_payload=True,
            query_filter=source_filter,
        ).points

        per_query_lists.append([
            {
                "text": r.payload["text"].strip(),
                "source": r.payload["source"],
                "chunk_index": r.payload["chunk_index"],
                "score": r.score,
                "rrf_score": r.score,
                "reranker_score": r.score,
                "retriever": "dense"
            }
            for r in results if r.payload.get("text", "").strip()
        ])

    # RRF fusion across all query angles
    fused = reciprocal_rank_fusion(per_query_lists)

    # When user selected specific docs, skip path-match targeting (they
    # already told us exactly which files to use) and guarantee coverage
    # by scrolling extra chunks from the selected sources if dense search
    # returned too few.
    if selected_sources:
        if len(fused) < top_k * 2:
            extra_results, _ = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=top_k * 4,
                with_payload=True,
                scroll_filter=source_filter,
            )
            existing = {c["text"][:100] for c in fused}
            for r in extra_results:
                text = (r.payload or {}).get("text", "").strip()
                if text and text[:100] not in existing:
                    fused.append({
                        "text": text,
                        "source": r.payload.get("source", ""),
                        "chunk_index": r.payload.get("chunk_index", 0),
                        "score": 0.3,
                        "rrf_score": 0.3,
                        "reranker_score": 0.3,
                        "retriever": "scroll_fill",
                    })
                    existing.add(text[:100])
        return fused

    # Source-targeted chunks for the raw topic (prepended, never cut off)
    raw_topic = topic or (queries[0] if queries else "")
    targeted = source_targeted_search(raw_topic)
    if targeted:
        fused_keys = {c["text"][:100] for c in fused}
        targeted_unique = [c for c in targeted if c["text"][:100] not in fused_keys]
        targeted_keys = {c["text"][:100] for c in targeted_unique}
        dense_remainder = [c for c in fused if c["text"][:100] not in targeted_keys]
        return targeted_unique + dense_remainder

    return fused


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test path scoring without hitting Qdrant.
    # Critical discrimination test: "Fuel System" on MG 413W. The same file
    # (Field manual of MG 413 MPFI.pdf) lives in ALL 9 system folders, so the
    # filename tokens tie them. Only sys_folder_specific should separate them.
    test_cases = [
        ("Fuel System",
         "/home/hpc25/Downloads/LVG/LVG/3. MG 413W MPFI/3. FUEL SYS/Field manual of MG 413 MPFI.pdf"),
        ("Fuel System",
         "/home/hpc25/Downloads/LVG/LVG/3. MG 413W MPFI/4. LUB SYS/Field manual of MG 413 MPFI.pdf"),
        ("Fuel System",
         "/home/hpc25/Downloads/LVG/LVG/3. MG 413W MPFI/5. COOLING SYS/Field manual of MG 413 MPFI.pdf"),
        ("Fuel System",
         "/home/hpc25/Downloads/LVG/LVG/3. MG 413W MPFI/6. BRAKE SYS/Field manual of MG 413 MPFI.pdf"),
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
        v, s, sp = score_source(source, qtok)
        vf, sf, fn = parse_path_levels(source)
        print(f"\nQuery : {query}")
        print(f"Source: .../{vf}/{sf}/{fn}")
        print(f"Scores: vehicle={v}  system={s}  specific={sp}")
