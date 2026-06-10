from codebase_search_rag.models import Chunk
import os
import hashlib
from pathlib import Path
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython
from uuid import uuid4
import json
import networkx as nx

class Indexer:
#     __init__(self, repo_path, embedder, collection):
#         - stores repo path
#         - stores embedder instance (injected, not created here)
#         - stores ChromaDB collection (injected, not created here)
#         - loads existing call_graph.json and symbol_table.json if they exist
    def __init__(self, repo_path, embedder, collection):
        self.repo_path = repo_path
        self.embedder = embedder
        self.collection = collection
        self.file_hashes_path = Path("codebase_search_rag/data/file_hashes.json")
        self.call_graph_path = Path("codebase_search_rag/data/call_graph.json")
        self.symbol_table_path = Path("codebase_search_rag/data/symbol_table.json")
        self.file_hashes = {}
        self.call_graph = {}
        self.symbol_table = {}
        self.target_types = ['function_definition', 'class_definition']
        self.parser = Parser(Language(tspython.language()))
        if self.file_hashes_path.exists():            
            # Open the file in read mode ('r')
            with open(self.file_hashes_path, 'r') as file:
            # Load the JSON data into a dictionary
                self.file_hashes = json.load(file)
                
        if self.call_graph_path.exists():            
            # Open the file in read mode ('r')
            with open(self.call_graph_path, 'r') as file:
            # Load the JSON data into a dictionary
                self.call_graph = json.load(file)
        
        if self.symbol_table_path.exists():            
            # Open the file in read mode ('r')
            with open(self.symbol_table_path, 'r') as file:
            # Load the JSON data into a dictionary
                self.symbol_table = json.load(file)


#     parse_and_chunk(file_path) -> list[Chunk]:
#         - reads .py file
#         - tree-sitter AST parse
#         - recursively collect function_definition + class_definition nodes
#         - for each node: extract name, code, start_line, end_line, metadata
#         - returns list of Chunk objects
    def parse_and_chunk(self, file_path: str) -> list[Chunk]:
        with open(file_path, "rb") as f:
            source_code = f.read()
        tree = self.parser.parse(source_code)
        root = tree.root_node
        nodes = self._collect_nodes(root)
        chunks = []
        for node in nodes:
            name_node = node.child_by_field_name("name")
            symbol = source_code[name_node.start_byte:name_node.end_byte].decode()
            chunk = Chunk(
                id=str(uuid4()),
                content=source_code[node.start_byte : node.end_byte].decode("utf-8"),
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                path=file_path,
                metadata={
                    "function_name": symbol,
                    "file_path": file_path,
                    "start_line": node.start_point[0],
                    "end_line": node.end_point[0],
                    "chunk_type": node.type,
                    "language": "python"
                }
            )
            chunks.append(chunk)
        return chunks   
    
#     _collect_nodes(node, target_types) -> list[Node]:
#         - recursively walks the AST tree to find target node types
#         - target types are: function_definition, class_definition
#         - if current node IS a target type -> add to results, stop recursing (treat as atomic unit)
#         - if current node is NOT a target type -> recurse into its children
#         - stopping at target nodes means we get complete functions/classes, not fragments
#         - returns flat list of all matching nodes found in the tree                
    def _collect_nodes(self, node: Node) -> list[Node]:
        result: list[Node] = []
        if node.type in self.target_types:
            result.append(node)
        else:
            for child in node.children:
                result.extend(self._collect_nodes(child))
        return result
    
#     build_graphs(chunks) -> None:
#         - extracts function calls and imports from each chunk
#         - builds NetworkX directed call graph: caller -> callee edges
#         - builds symbol table: function/class name -> file, lines, type
#         - saves call_graph.json and symbol_table.json to data/
    def build_graphs(self, chunks: list[Chunk]) -> None:
        graph = nx.DiGraph()

        for chunk in chunks:
            caller = chunk.metadata["function_name"]
            graph.add_node(caller)

            # parse the chunk content to find calls
            tree = self.parser.parse(chunk.content.encode())
            calls = self._extract_calls(tree.root_node)
            for callee in calls:
                graph.add_edge(caller, callee)

            # update symbol table
            self.symbol_table[caller] = {
                "file": chunk.path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "type": chunk.metadata["chunk_type"],
            }

        # merge into existing call graph
        self.call_graph = nx.node_link_data(graph)

        # save to disk
        with open(self.call_graph_path, 'w') as f:
            json.dump(self.call_graph, f)

        with open(self.symbol_table_path, 'w') as f:
            json.dump(self.symbol_table, f)

    def _extract_calls(self, node: Node) -> list[str]:
        calls = []
        if node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node:
                calls.append(func_node.text.decode())
        for child in node.children:
            calls.extend(self._extract_calls(child))
        return calls
    
#     _check_index_status(file_path, indexed_size, indexed_mtime, indexed_hash) -> str:
#         - checks if a file has changed since it was last indexed
#         - uses a two-step approach: fast metadata pre-check (size + mtime) first
#         - if metadata differs, falls back to MD5 hash comparison for accuracy
#         - returns "DELETED" if file no longer exists
#         - returns "CHANGED" if file content has changed since last index
#         - returns "UNCHANGED" if file is identical to last indexed version
    def _check_index_status(self, file_path: str, indexed_size: int, indexed_mtime: float, indexed_hash: str):
        try:
            stat_info = os.stat(file_path)
            current_size = stat_info.st_size
            current_mtime = stat_info.st_mtime
        except FileNotFoundError:
            return "DELETED"

        # Step 1: Fast metadata pre-check
        if current_size != indexed_size or current_mtime != indexed_mtime:

            # Step 2: Fallback to hash verification (confirms content alteration)
            current_hash = self._get_file_md5(file_path)
            if current_hash != indexed_hash:
                return "CHANGED"

        return "UNCHANGED"
    
#     _get_file_md5(file_path, chunk_size) -> str:
#         - computes MD5 hash of a file's contents to detect changes
#         - reads file in chunks of 8192 bytes to avoid loading large files into memory
#         - returns hex digest string e.g. "abc123..." representing the file fingerprint
#         - returns empty string "" if file is not found
#         - used by _check_index_status as fallback when metadata pre-check fails
    def _get_file_md5(self, file_path: str, chunk_size: int = 8192) -> str:
        """Computes the MD5 hash of a file by reading it in chunks."""
        md5_hash = hashlib.md5()

        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to efficiently manage memory
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    md5_hash.update(chunk)
            return md5_hash.hexdigest()
        except FileNotFoundError:
            return ""
    
#     index_repository() -> None:
#         - walks all .py files in repo_path
#         - for each file: check hash against stored hash
#         - if changed or new: parse_and_chunk -> build_graphs -> embed -> store in ChromaDB
#         - if unchanged: skip
#         - saves updated hashes
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
                chunks = self.parse_and_chunk(file_path_str)
                self.build_graphs(chunks)
                self._store_chunks(chunks)
                stat_info = os.stat(file_path_str)
                self.file_hashes[file_path_str] = {
                    "size": stat_info.st_size,
                    "mtime": stat_info.st_mtime,
                    "hash": self._get_file_md5(file_path_str)
                }
            elif status == 'DELETED':
                del self.file_hashes[file_path_str]

        # Save updated hashes to disk after processing all files
        with open(self.file_hashes_path, 'w') as f:
            json.dump(self.file_hashes, f)

    
#     _store_chunks(chunks) -> None:
#         - calls embedder.embed_chunks() to get vectors
#         - batch adds to ChromaDB collection with metadata
    def _store_chunks(self, chunks: list[Chunk]) -> None:
        # Use batching
        BATCH_SIZE = 300
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            texts = [chunk.content for chunk in batch]
            self.collection.add(
                ids=[chunk.id for chunk in batch],
                documents=texts,
                embeddings=self.embedder.embed_chunks(texts),
                metadatas=[chunk.metadata for chunk in batch]
            )
