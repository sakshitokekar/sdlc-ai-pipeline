from codebase_search_rag.models import Chunk


MAX_SYMBOLS = 30  # context budget — prevents LLM context window overflow


class CodeRetriever:
#     __init__(collection, embedder, indexer)
#         - collection: ChromaDB collection
#         - embedder: Embedder instance for query embedding
#         - indexer: Indexer instance for graph traversal and symbol lookup
    def __init__(self, collection, embedder, indexer):
        self.collection = collection
        self.embedder = embedder
        self.indexer = indexer


#     semantic_search(query, top_k=10) -> dict
#         - embeds query using embedder.embed_query()
#         - queries ChromaDB with query_embeddings
#         - returns raw ChromaDB result with ids, documents, metadatas, distances
    def semantic_search(self, query: str, top_k: int = 10) -> dict:
        embedded_query = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[embedded_query],
            n_results=top_k
        )
        return results


#     graph_expand(symbol_ids, file_paths, depth=2) -> dict
#         - BFS traversal of call graph starting from seed symbol_ids
#         - each hop follows callers + callees of current frontier
#         - scores symbols by graph distance: seed=1.0, depth1=0.8, depth2=0.6 etc
#         - applies MAX_SYMBOLS budget — keeps highest scored symbols only
#         - also collects file-level dependencies and dependents
#         - returns expanded_symbol_ids (scored), dependencies, dependents
    def graph_expand(self, symbol_ids: list[str], file_paths: list[str], depth: int = 2) -> dict:
        # Score symbols by graph distance — seed symbols are most relevant
        scores = {sym: 1.0 for sym in symbol_ids}
        frontier = set(symbol_ids)
        visited = set(symbol_ids)

        for hop in range(depth):
            next_frontier = set()
            hop_score = 1.0 - (0.2 * (hop + 1))  # 0.8 at depth1, 0.6 at depth2
            for sym in frontier:
                neighbors = set(self.indexer.get_callers(sym)) | set(self.indexer.get_callees(sym))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        # Keep highest score if already seen via another path
                        scores[neighbor] = max(scores.get(neighbor, 0), hop_score)
            frontier = next_frontier - visited
            visited.update(frontier)

        # Apply context budget — sort by score, keep top MAX_SYMBOLS
        sorted_symbols = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [sym for sym, _ in sorted_symbols[:MAX_SYMBOLS]]

        # File-level retrieval — collect related files from dependencies + dependents
        related_files = set()
        for fp in file_paths:
            related_files.update(self.indexer.get_dependencies(fp))
            related_files.update(self.indexer.get_dependents(fp))

        dependencies = list({dep for fp in file_paths for dep in self.indexer.get_dependencies(fp)})
        dependents = list({dep for fp in file_paths for dep in self.indexer.get_dependents(fp)})

        return {
            "expanded_symbol_ids": top_symbols,
            "symbol_scores": dict(sorted_symbols[:MAX_SYMBOLS]),
            "dependencies": dependencies,
            "dependents": dependents,
            "related_files": list(related_files)
        }


#     retrieve(query, top_k=10) -> dict
#         - calls semantic_search to get top K semantically similar chunks
#         - extracts symbol_ids and file_paths from results
#         - calls graph_expand to find related symbols via call graph
#         - fetches graph-expanded chunks from ChromaDB using chunk_ids from symbol_table
#         - ranks results: semantic hits first (score 1.0), then graph expansions by score
#         - deduplicates by chunk ID
#         - applies MAX_SYMBOLS context budget
#         - returns rich context object for Agent 2
    def retrieve(self, query: str, top_k: int = 10) -> dict:
        # Step 1: Semantic search
        semantic_results = self.semantic_search(query, top_k)
        seed_symbol_ids = [meta["symbol_id"] for meta in semantic_results["metadatas"][0]]
        file_paths = [meta["file_path"] for meta in semantic_results["metadatas"][0]]

        # Step 2: Graph expansion
        graph_results = self.graph_expand(seed_symbol_ids, file_paths)

        # Step 3: Fetch graph-expanded chunks using chunk_id stored in symbol_table
        # Avoids double ChromaDB lookup — chunk_id is directly in symbol_table
        graph_chunk_ids = [
            self.indexer.symbol_table[sym_id]["chunk_id"]
            for sym_id in graph_results["expanded_symbol_ids"]
            if sym_id in self.indexer.symbol_table
            and "chunk_id" in self.indexer.symbol_table[sym_id]
        ]
        graph_chunks = self.collection.get(
            ids=graph_chunk_ids,
            include=["documents", "metadatas"]
        ) if graph_chunk_ids else {"ids": [], "documents": [], "metadatas": []}

        # Step 4: Combine and rank — semantic hits first, then graph expansions by score
        combined = {}

        # Semantic hits: highest priority (score = distance-based, lower = better)
        distances = semantic_results.get("distances", [[]])[0]
        for i, (id, doc, meta) in enumerate(zip(
            semantic_results["ids"][0],
            semantic_results["documents"][0],
            semantic_results["metadatas"][0]
        )):
            combined[id] = {
                "id": id,
                "content": doc,
                "metadata": meta,
                "score": 1.0,
                "source": "semantic",
                "distance": distances[i] if distances else None
            }

        # Graph expansion hits: scored by graph distance
        symbol_scores = graph_results["symbol_scores"]
        for id, doc, meta in zip(
            graph_chunks["ids"],
            graph_chunks["documents"],
            graph_chunks["metadatas"]
        ):
            if id not in combined:  # don't overwrite semantic hits
                sym_id = meta.get("symbol_id", "")
                combined[id] = {
                    "id": id,
                    "content": doc,
                    "metadata": meta,
                    "score": symbol_scores.get(sym_id, 0.5),
                    "source": "graph"
                }

        # Sort by score descending — semantic hits naturally float to top
        ranked_chunks = sorted(combined.values(), key=lambda x: x["score"], reverse=True)

        return {
            "query": query,
            "seed_symbols": seed_symbol_ids,
            "expanded_symbols": graph_results["expanded_symbol_ids"],
            "symbol_scores": graph_results["symbol_scores"],
            "dependencies": graph_results["dependencies"],
            "dependents": graph_results["dependents"],
            "related_files": graph_results["related_files"],
            "chunks": ranked_chunks
        }