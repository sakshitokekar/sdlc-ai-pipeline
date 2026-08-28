from codebase_search_rag.models import Chunk
import os
import hashlib
from pathlib import Path
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython
import json
import networkx as nx


class Indexer:
#     __init__(self, repo_path, embedder, collection):
#         - stores repo path
#         - stores embedder instance (injected, not created here)
#         - stores ChromaDB collection (injected, not created here)
#         - loads existing indexes from disk if available
#         - initialises single global NetworkX DiGraph
#         - initialises name_index for fast symbol resolution by short name
    def __init__(self, repo_path, embedder, collection):
        self.repo_path = repo_path
        self.embedder = embedder
        self.collection = collection
        self.file_hashes_path = Path("codebase_search_rag/data/file_hashes.json")
        self.call_graph_path = Path("codebase_search_rag/data/call_graph.json")
        self.symbol_table_path = Path("codebase_search_rag/data/symbol_table.json")
        self.dependency_graph_path = Path("codebase_search_rag/data/dependency_graph.json")
        self.name_index_path = Path("codebase_search_rag/data/name_index.json")
        self.file_hashes = {}
        self.symbol_table = {}
        self.dependency_graph = {}

        # name_index: short_name -> list of fully qualified symbol IDs
        # e.g. {"process_payment": ["services.payment.process_payment"]}
        # used to resolve raw call strings to qualified symbol IDs
        self.name_index = {}

        # Single global graph — never overwritten per file
        self.graph = nx.DiGraph()

        self.target_types = ['function_definition', 'class_definition']
        self.parser = Parser(Language(tspython.language()))

        # Load existing indexes from disk
        if self.file_hashes_path.exists():
            with open(self.file_hashes_path, 'r') as f:
                self.file_hashes = json.load(f)

        if self.call_graph_path.exists():
            with open(self.call_graph_path, 'r') as f:
                self.graph = nx.node_link_graph(json.load(f))

        if self.symbol_table_path.exists():
            with open(self.symbol_table_path, 'r') as f:
                self.symbol_table = json.load(f)

        if self.dependency_graph_path.exists():
            with open(self.dependency_graph_path, 'r') as f:
                self.dependency_graph = json.load(f)

        if self.name_index_path.exists():
            with open(self.name_index_path, 'r') as f:
                self.name_index = json.load(f)


#     _make_symbol_id(file_path, symbol_name, class_name=None) -> str:
#         - generates a fully qualified symbol ID to avoid cross-file name collisions
#         - if symbol is a method, includes class name: services.payment.PaymentService.process
#         - if symbol is a top-level function: services.payment.process_payment
    def _make_symbol_id(self, file_path: str, symbol_name: str, class_name: str = None) -> str:
        relative = Path(file_path).relative_to(self.repo_path)
        module = str(relative).replace("/", ".").replace("\\", ".").removesuffix(".py")
        if class_name:
            return f"{module}.{class_name}.{symbol_name}"
        return f"{module}.{symbol_name}"


#     _make_chunk_id(file_path, symbol_name) -> str:
#         - generates deterministic chunk ID based on file path + symbol name
#         - ensures re-indexing same symbol produces same ID
#         - allows ChromaDB upsert to overwrite stale embeddings
    def _make_chunk_id(self, file_path: str, symbol_name: str) -> str:
        raw = f"{file_path}::{symbol_name}"
        return hashlib.md5(raw.encode()).hexdigest()


#     parse_and_chunk(file_path) -> list[Chunk]:
#         - reads .py file
#         - parses full file AST once (for imports + top-level structure)
#         - recursively collects function_definition + class_definition nodes
#         - handles class methods: symbol ID includes class name
#         - returns list of Chunk objects with deterministic IDs
    def parse_and_chunk(self, file_path: str) -> list[Chunk]:
        with open(file_path, "rb") as f:
            source_code = f.read()
        tree = self.parser.parse(source_code)
        root = tree.root_node
        chunks = []

        # Collects every top-level node that is NEITHER a function nor a
        # class — imports, module-level constants, and critically things
        # like `if __name__ == "__main__":` blocks. Without this, RAG
        # structurally could never retrieve module-level code (this was a
        # real bug: app.py's `app.run(host=..., debug=...)` line lives
        # inside an `if __name__` block and was invisible to retrieval,
        # forcing Gemini to guess its exact bytes for a code-fix — which
        # failed the exact-match replacement in agent2_dev.py). These are
        # consolidated into ONE chunk per file rather than one per
        # statement, since they're typically small and read together.
        module_level_nodes = []

        # Walk top-level nodes — handle classes and top-level functions separately
        for node in root.children:
            if node.type == "class_definition":
                class_name_node = node.child_by_field_name("name")
                class_name = source_code[class_name_node.start_byte:class_name_node.end_byte].decode()
                symbol_id = self._make_symbol_id(file_path, class_name)

                # Add class-level chunk
                chunks.append(Chunk(
                    id=self._make_chunk_id(file_path, class_name),
                    content=source_code[node.start_byte:node.end_byte].decode("utf-8"),
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                    path=file_path,
                    metadata={
                        "symbol_id": symbol_id,
                        "function_name": class_name,
                        "class_name": class_name,
                        "file_path": file_path,
                        "start_line": node.start_point[0],
                        "end_line": node.end_point[0],
                        "chunk_type": "class_definition",
                        "language": "python"
                    }
                ))

                # Add method-level chunks inside the class
                for child in node.children:
                    if child.type == "block":
                        for method in child.children:
                            if method.type == "function_definition":
                                method_name_node = method.child_by_field_name("name")
                                method_name = source_code[method_name_node.start_byte:method_name_node.end_byte].decode()
                                method_symbol_id = self._make_symbol_id(file_path, method_name, class_name)
                                chunks.append(Chunk(
                                    id=self._make_chunk_id(file_path, f"{class_name}.{method_name}"),
                                    content=source_code[method.start_byte:method.end_byte].decode("utf-8"),
                                    start_line=method.start_point[0],
                                    end_line=method.end_point[0],
                                    path=file_path,
                                    metadata={
                                        "symbol_id": method_symbol_id,
                                        "function_name": method_name,
                                        "class_name": class_name,
                                        "file_path": file_path,
                                        "start_line": method.start_point[0],
                                        "end_line": method.end_point[0],
                                        "chunk_type": "function_definition",
                                        "language": "python"
                                    }
                                ))

            elif node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                symbol = source_code[name_node.start_byte:name_node.end_byte].decode()
                symbol_id = self._make_symbol_id(file_path, symbol)
                chunks.append(Chunk(
                    id=self._make_chunk_id(file_path, symbol),
                    content=source_code[node.start_byte:node.end_byte].decode("utf-8"),
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                    path=file_path,
                    metadata={
                        "symbol_id": symbol_id,
                        "function_name": symbol,
                        "class_name": None,
                        "file_path": file_path,
                        "start_line": node.start_point[0],
                        "end_line": node.end_point[0],
                        "chunk_type": "function_definition",
                        "language": "python"
                    }
                ))
            else:
                # Everything that isn't a class or function definition —
                # import statements, module-level assignments, if/for/with
                # blocks at the top level (including `if __name__ ==
                # "__main__":`), decorators on nothing, etc. Skip pure
                # comment/whitespace-only nodes implicitly (tree-sitter
                # doesn't emit separate nodes for those at this level).
                module_level_nodes.append(node)

        # Consolidate all collected module-level nodes into ONE chunk
        # spanning from the first to the last such node, so retrieval can
        # find (for example) the `if __name__ == "__main__":` block that
        # would otherwise never be chunked at all.
        if module_level_nodes:
            start_byte = module_level_nodes[0].start_byte
            end_byte = module_level_nodes[-1].end_byte
            start_line = module_level_nodes[0].start_point[0]
            end_line = module_level_nodes[-1].end_point[0]
            module_symbol_id = self._make_symbol_id(file_path, "__module_level__")

            chunks.append(Chunk(
                id=self._make_chunk_id(file_path, "__module_level__"),
                content=source_code[start_byte:end_byte].decode("utf-8"),
                start_line=start_line,
                end_line=end_line,
                path=file_path,
                metadata={
                    "symbol_id": module_symbol_id,
                    "function_name": "__module_level__",
                    "class_name": None,
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "chunk_type": "module_level",
                    "language": "python"
                }
            ))

        return chunks


#     _cleanup_file_symbols(file_path) -> None:
#         - removes all graph nodes/edges for symbols belonging to file_path
#         - removes all symbol_table entries for file_path
#         - removes file from dependency_graph
#         - removes file's symbols from name_index
#         - deletes stale embeddings from ChromaDB
#         - called before re-indexing a changed file to prevent stale data
    def _cleanup_file_symbols(self, file_path: str) -> None:
        stale_symbol_ids = [
            sym_id for sym_id, info in self.symbol_table.items()
            if info["file"] == file_path
        ]

        # Remove from ChromaDB
        stale_chunk_ids = [
            self._make_chunk_id(file_path, info["symbol_name"])
            for sym_id, info in self.symbol_table.items()
            if info["file"] == file_path
        ]
        if stale_chunk_ids:
            self.collection.delete(ids=stale_chunk_ids)

        # Remove from graph and symbol table
        for sym_id in stale_symbol_ids:
            del self.symbol_table[sym_id]
            if self.graph.has_node(sym_id):
                self.graph.remove_node(sym_id)

            # Remove from name_index
            short_name = sym_id.split(".")[-1]
            if short_name in self.name_index:
                self.name_index[short_name] = [
                    s for s in self.name_index[short_name] if s != sym_id
                ]
                if not self.name_index[short_name]:
                    del self.name_index[short_name]

        # Remove from dependency graph
        if file_path in self.dependency_graph:
            del self.dependency_graph[file_path]


#     build_graphs(chunks, file_path, file_source_code) -> None:
#         - cleans up stale symbols for file_path before rebuilding
#         - extracts imports from FULL FILE AST (not chunk AST) to fix import extraction bug
#         - adds caller -> callee edges using resolved symbol IDs
#         - updates name_index with short_name -> [qualified_symbol_id] mappings
#         - updates symbol_table with location info
#         - updates dependency_graph with file-level import dependencies
#         - NOTE: does NOT persist to disk here — persisted once after index_repository() finishes
    def build_graphs(self, chunks: list[Chunk], file_path: str, file_source_code: bytes) -> None:
        self._cleanup_file_symbols(file_path)

        # Parse full file AST for imports (imports are outside function chunks)
        full_tree = self.parser.parse(file_source_code)
        imports = self._extract_imports(full_tree.root_node)
        self.dependency_graph[file_path] = imports

        # First pass: register all symbols in name_index and symbol_table
        for chunk in chunks:
            symbol_id = chunk.metadata["symbol_id"]
            short_name = chunk.metadata["function_name"]

            self.graph.add_node(symbol_id)
            self.symbol_table[symbol_id] = {
                "file": chunk.path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "type": chunk.metadata["chunk_type"],
                "symbol_name": short_name,
            }

            # Register in name_index for fast resolution
            if short_name not in self.name_index:
                self.name_index[short_name] = []
            if symbol_id not in self.name_index[short_name]:
                self.name_index[short_name].append(symbol_id)

        # Second pass: build call edges (name_index now populated)
        for chunk in chunks:
            symbol_id = chunk.metadata["symbol_id"]
            tree = self.parser.parse(chunk.content.encode())
            raw_calls = self._extract_calls(tree.root_node)
            for raw_call in raw_calls:
                resolved = self._resolve_call(raw_call, file_path)
                self.graph.add_edge(symbol_id, resolved)


#     _extract_calls(node) -> list[str]:
#         - recursively walks AST node to find all function call expressions
#         - returns list of raw call strings for later resolution
    def _extract_calls(self, node: Node) -> list[str]:
        calls = []
        if node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node:
                calls.append(func_node.text.decode())
        for child in node.children:
            calls.extend(self._extract_calls(child))
        return calls


#     _extract_imports(node) -> list[str]:
#         - recursively walks FULL FILE AST to find import statements
#         - handles both "import x" and "from x import y" forms
#         - returns list of raw import strings
    def _extract_imports(self, node: Node) -> list[str]:
        imports = []
        if node.type in ("import_statement", "import_from_statement"):
            imports.append(node.text.decode())
        for child in node.children:
            imports.extend(self._extract_imports(child))
        return imports


#     _resolve_call(raw_call, file_path) -> str:
#         - resolves raw call string to fully qualified symbol ID using name_index
#         - handles simple calls: "process_payment" -> "services.payment.process_payment"
#         - handles method calls: "payment_service.process" -> looks up "process" in name_index
#         - if exactly one candidate found -> return it
#         - if multiple candidates -> return raw_call (ambiguous, leave for LLM to reason)
#         - if no candidate -> return raw_call as fallback
    def _resolve_call(self, raw_call: str, file_path: str) -> str:
        # Handle method calls like "obj.method" — extract just the method name
        short_name = raw_call.split(".")[-1]

        candidates = self.name_index.get(short_name, [])
        if len(candidates) == 1:
            return candidates[0]

        # Multiple candidates — try to disambiguate using current file's module
        relative = Path(file_path).relative_to(self.repo_path)
        module = str(relative).replace("/", ".").replace("\\", ".").removesuffix(".py")
        for candidate in candidates:
            if candidate.startswith(module):
                return candidate

        # Fall back to raw call
        return raw_call


#     _check_index_status(file_path, indexed_size, indexed_mtime, indexed_hash) -> str:
#         - checks if a file has changed since it was last indexed
#         - two-step: fast metadata pre-check (size + mtime), fallback to MD5
#         - returns "DELETED", "CHANGED", or "UNCHANGED"
    def _check_index_status(self, file_path: str, indexed_size: int, indexed_mtime: float, indexed_hash: str) -> str:
        try:
            stat_info = os.stat(file_path)
            current_size = stat_info.st_size
            current_mtime = stat_info.st_mtime
        except FileNotFoundError:
            return "DELETED"

        if current_size != indexed_size or current_mtime != indexed_mtime:
            current_hash = self._get_file_md5(file_path)
            if current_hash != indexed_hash:
                return "CHANGED"

        return "UNCHANGED"


#     _get_file_md5(file_path, chunk_size) -> str:
#         - computes MD5 hash of file contents in 8192-byte chunks
#         - memory efficient for large files
#         - returns empty string if file not found
    def _get_file_md5(self, file_path: str, chunk_size: int = 8192) -> str:
        md5_hash = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    md5_hash.update(chunk)
            return md5_hash.hexdigest()
        except FileNotFoundError:
            return ""


#     get_callers(symbol_id) -> list[str]:
#         - returns list of symbol IDs that call the given symbol
#         - uses NetworkX graph.predecessors()
#         - used by retriever for graph traversal: "who calls this function?"
    def get_callers(self, symbol_id: str) -> list[str]:
        if self.graph.has_node(symbol_id):
            return list(self.graph.predecessors(symbol_id))
        return []


#     get_callees(symbol_id) -> list[str]:
#         - returns list of symbol IDs that the given symbol calls
#         - uses NetworkX graph.successors()
#         - used by retriever for graph traversal: "what does this function call?"
    def get_callees(self, symbol_id: str) -> list[str]:
        if self.graph.has_node(symbol_id):
            return list(self.graph.successors(symbol_id))
        return []


#     get_dependencies(file_path) -> list[str]:
#         - returns list of import strings for the given file
#         - used by retriever: "what does this file import?"
    def get_dependencies(self, file_path: str) -> list[str]:
        return self.dependency_graph.get(file_path, [])


#     get_dependents(file_path) -> list[str]:
#         - returns list of files that import the given file
#         - used for impact analysis: "if I change this file, what else might break?"
    def get_dependents(self, file_path: str) -> list[str]:
        relative = Path(file_path).relative_to(self.repo_path)
        module = str(relative).replace("/", ".").replace("\\", ".").removesuffix(".py")
        dependents = []
        for f, imports in self.dependency_graph.items():
            for imp in imports:
                if module in imp:
                    dependents.append(f)
                    break
        return dependents


#     find_symbol(name) -> list[dict]:
#         - looks up a short symbol name in name_index
#         - returns list of matching symbol info dicts from symbol_table
#         - used by retriever and Agent 2 to find a symbol's location
    def find_symbol(self, name: str) -> list[dict]:
        candidates = self.name_index.get(name, [])
        return [self.symbol_table[c] for c in candidates if c in self.symbol_table]


#     index_repository() -> None:
#         - walks all .py files in repo_path
#         - for each file: check hash, process if NEW or CHANGED, skip if UNCHANGED
#         - passes full file source code to build_graphs for correct import extraction
#         - persists ALL indexes to disk ONCE after all files processed (not per file)
    def index_repository(self) -> None:
        for file_path in Path(self.repo_path).rglob("*.py"):
            file_path_str = str(file_path)
            stored = self.file_hashes.get(file_path_str, None)

            if stored is None:
                status = 'NEW'
            else:
                status = self._check_index_status(
                    file_path_str,
                    stored["size"],
                    stored["mtime"],
                    stored["hash"]
                )

            if status in ('NEW', 'CHANGED'):
                with open(file_path_str, "rb") as f:
                    source_code = f.read()
                chunks = self.parse_and_chunk(file_path_str)
                self.build_graphs(chunks, file_path_str, source_code)
                self._store_chunks(chunks)
                stat_info = os.stat(file_path_str)
                self.file_hashes[file_path_str] = {
                    "size": stat_info.st_size,
                    "mtime": stat_info.st_mtime,
                    "hash": self._get_file_md5(file_path_str)
                }
            elif status == 'DELETED':
                self._cleanup_file_symbols(file_path_str)
                del self.file_hashes[file_path_str]

        # Persist ALL indexes once after processing all files
        with open(self.file_hashes_path, 'w') as f:
            json.dump(self.file_hashes, f)
        with open(self.call_graph_path, 'w') as f:
            json.dump(nx.node_link_data(self.graph), f)
        with open(self.symbol_table_path, 'w') as f:
            json.dump(self.symbol_table, f)
        with open(self.dependency_graph_path, 'w') as f:
            json.dump(self.dependency_graph, f)
        with open(self.name_index_path, 'w') as f:
            json.dump(self.name_index, f)


#     _store_chunks(chunks) -> None:
#         - embeds chunks and upserts to ChromaDB in batches of 300
#         - upsert overwrites stale embeddings instead of duplicating
    def _store_chunks(self, chunks: list[Chunk]) -> None:
        BATCH_SIZE = 300
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            texts = [chunk.content for chunk in batch]
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=texts,
                embeddings=self.embedder.embed_chunks(texts),
                metadatas=[chunk.metadata for chunk in batch]
            )