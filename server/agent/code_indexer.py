"""
A service that indexes all code in the repository for self-reflection capabilities.
With improved filtering and proper update logic to avoid duplicate entries.
"""
import os
import logging
import fnmatch
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
import subprocess

from server.agent.rag_manager import rag_manager

logger = logging.getLogger(__name__)

class CodeIndexer:
    """
    Scans the codebase, chunks code files appropriately, and adds them to the RAG system.
    Includes robust filtering and proper updating to prevent duplication.
    """

    def __init__(self, repo_root: Optional[str] = None):
        """Initialize the code indexer with repository root path."""
        self.repo_root = self._find_repo_root() if repo_root is None else repo_root
        # Track indexed files and their content hashes to detect changes
        self.indexed_files_meta: Dict[str, str] = {}  # path -> content_hash
        self.source_to_ids: Dict[str, List[int]] = {}  # source -> list of document IDs

        # Directories to exclude (expanded list)
        self.exclude_dirs = {
            # Python-specific
            '__pycache__', 'venv', 'env', '.venv', '.env', '.pytest_cache',
            # JS/TS-specific
            'node_modules', 'dist', 'build', '.next',
            # Version control
            '.git', '.svn', '.hg',
            # Data directories
            'data', 'datasets', 'chroma_db',
            # Cache and output directories
            '.cache', '.chainlit',
            # OS-specific
            '.DS_Store', 'Thumbs.db'
        }

        # Specific extensions to exclude (expanded)
        self.exclude_extensions = {
            # Python bytecode
            '.pyc', '.pyo', '.pyd',
            # Binaries
            '.so', '.dll', '.exe', '.bin', '.dat',
            # Images
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp', '.svg',
            # Archives
            '.zip', '.tar', '.gz', '.7z', '.rar', '.jar', '.war',
            # Media
            '.mp3', '.mp4', '.avi', '.mov', '.flv', '.wav',
            # Data files
            '.db', '.sqlite', '.mdb', '.ldb', '.npy', '.pkl',
            # Log and cache files
            '.log', '.cache', '.tmp',
            # IDE/editor files
            '.idea', '.vscode', '.vs'
        }

        # Specific files to exclude by pattern
        self.exclude_file_patterns = [
            '.*',          # All hidden files
            '*.lock',      # Lock files
            '*.min.*',     # Minified files
            'package-lock.json',
            'yarn.lock',
            'poetry.lock',
            'Pipfile.lock',
            'requirements*.txt',
            '.env*',
            '.flake8',
            '.gitignore',
            '.prettierrc',
            '.eslintrc',
            'Dockerfile',
            'LICENSE',
            '*.md5',
            '*.sum'
        ]

        # Only include these extensions
        self.include_extensions = {
            # Code files
            '.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.scss',
            # Configuration files
            '.json', '.yaml', '.yml', '.toml',
            # Documentation
            '.md', '.rst', '.txt'
        }

        self.max_file_size_kb = 500  # Reduced to 500KB to avoid very large files
        self._load_indexed_files_meta()

    def _find_repo_root(self) -> str:
        """Find the Git repository root using git command."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--show-toplevel'],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("Could not determine Git repo root. Using current directory.")
            return os.getcwd()

    def _should_index_file(self, file_path: str) -> bool:
        """
        Determine whether a file should be indexed based on multiple filtering criteria.
        Returns True if the file should be indexed, False otherwise.
        """
        # Skip files in excluded directories
        parts = Path(file_path).parts
        for part in parts:
            if part in self.exclude_dirs or any(fnmatch.fnmatch(part, pattern) for pattern in ['.*']):
                return False

        # Get file name and extension
        file_name = os.path.basename(file_path)
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # Skip files matching exclude patterns
        if any(fnmatch.fnmatch(file_name, pattern) for pattern in self.exclude_file_patterns):
            return False

        # Skip files with excluded extensions
        if ext in self.exclude_extensions:
            return False

        # Skip files that aren't in the include list (if specified)
        if self.include_extensions and ext not in self.include_extensions:
            return False

        # Skip files that are too large
        try:
            if os.path.getsize(file_path) > self.max_file_size_kb * 1024:
                logger.info(f"Skipping large file: {file_path}")
                return False
        except OSError:
            return False

        # Simple binary file detection
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try to read the first few kb to see if it's text
                sample = f.read(1024)
                # If sample contains null bytes, it's probably binary
                if '\0' in sample:
                    logger.info(f"Skipping binary file: {file_path}")
                    return False
        except UnicodeDecodeError:
            logger.info(f"Skipping binary file (decode error): {file_path}")
            return False
        except Exception as e:
            logger.warning(f"Error checking file {file_path}: {e}")
            return False

        return True

    def _compute_content_hash(self, content: str) -> str:
        """
        Compute a hash of file content to detect changes.
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _load_indexed_files_meta(self) -> None:
        """
        Load metadata about indexed files from the document store.
        This helps us track what's already indexed and their content hashes.
        """
        if not hasattr(rag_manager, 'doc_store') or not hasattr(rag_manager, 'index'):
            logger.warning("RAG manager not properly initialized, can't load indexed files metadata")
            return

        try:
            # Build source_to_ids mapping from the current document store
            self.source_to_ids = {}
            for doc_id, doc in rag_manager.doc_store.items():
                source = doc.get('source', '')

                # Check if this is a code file source (contains a path)
                if ':' in source:  # Format is typically "path:section"
                    file_path = source.split(':', 1)[0]  # Extract just the file path part

                    if file_path not in self.source_to_ids:
                        self.source_to_ids[file_path] = []
                    self.source_to_ids[file_path].append(doc_id)

            # Update indexed_files_meta
            for file_path in self.source_to_ids.keys():
                abs_path = os.path.join(self.repo_root, file_path)
                if os.path.exists(abs_path):
                    try:
                        with open(abs_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            self.indexed_files_meta[file_path] = self._compute_content_hash(content)
                    except Exception:
                        pass  # Skip files that can't be read

            logger.info(f"Loaded metadata for {len(self.indexed_files_meta)} previously indexed files")

        except Exception as e:
            logger.error(f"Error loading indexed files metadata: {e}", exc_info=True)

    def _chunk_code_file(self, file_path: str, content: str) -> List[Dict[str, str]]:
        """
        Split a code file into logical chunks for better retrieval.

        For Python files, this tries to chunk by class and function definitions.
        For other files, it uses simpler line-based chunking.
        """
        chunks = []
        _, ext = os.path.splitext(file_path)
        rel_path = os.path.relpath(file_path, self.repo_root)

        # If it's a Python file, use more sophisticated chunking
        if ext.lower() == '.py':
            # Simple implementation for Python files - split by classes and functions
            import re

            # Pattern for class and function definitions
            pattern = r'(class\s+\w+\(.*?\)|def\s+\w+\(.*?\))'

            # Get all matches of the pattern
            matches = list(re.finditer(pattern, content))

            if not matches:
                # If no matches, treat the whole file as a chunk
                chunks.append({
                    "text": content,
                    "source": f"{rel_path}:FULL"
                })
                return chunks

            # Process each chunk
            for i, match in enumerate(matches):
                start_pos = match.start()
                # If it's the last match, go to the end of the file
                end_pos = matches[i+1].start() if i < len(matches) - 1 else len(content)

                chunk_content = content[start_pos:end_pos]
                # Get the definition line
                definition_line = match.group(0)

                chunks.append({
                    "text": chunk_content,
                    "source": f"{rel_path}:{definition_line.strip()}"
                })

            # Also add the imports and module-level code at the top
            if matches and matches[0].start() > 0:
                top_content = content[:matches[0].start()]
                chunks.append({
                    "text": top_content,
                    "source": f"{rel_path}:IMPORTS"
                })

            return chunks

        # For other files, use simpler line-based chunking
        lines = content.split('\n')
        chunk_size = 100  # Number of lines per chunk

        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_content = '\n'.join(chunk_lines)

            chunks.append({
                "text": chunk_content,
                "source": f"{rel_path}:{i+1}-{i+len(chunk_lines)}"
            })

        return chunks

    def _remove_previous_chunks(self, rel_path: str) -> None:
        """
        Remove all previously indexed chunks for a specific file.
        """
        # Get all document IDs that need to be removed
        doc_ids_to_remove = []

        # Find all documents with sources starting with this file path
        for source, ids in list(self.source_to_ids.items()):
            if source == rel_path or source.startswith(f"{rel_path}:"):
                doc_ids_to_remove.extend(ids)
                # Remove from the source_to_ids mapping
                del self.source_to_ids[source]

        if not doc_ids_to_remove:
            return

        # We can't directly remove from the FAISS index, so we'll have to rebuild it
        # Instead, we'll mark these documents as deleted in the document store
        for doc_id in doc_ids_to_remove:
            if doc_id in rag_manager.doc_store:
                del rag_manager.doc_store[doc_id]

        logger.info(f"Removed {len(doc_ids_to_remove)} previous chunks for {rel_path}")

    def index_file(self, file_path: str) -> bool:
        """
        Index a single file by chunking it and adding to the RAG system.
        If the file was previously indexed, check if it changed before reindexing.
        """
        if not self._should_index_file(file_path):
            return False

        try:
            # Get relative path for storage
            rel_path = os.path.relpath(file_path, self.repo_root)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Compute content hash to check for changes
            content_hash = self._compute_content_hash(content)

            # Check if file already indexed and unchanged
            if rel_path in self.indexed_files_meta and self.indexed_files_meta[rel_path] == content_hash:
                logger.debug(f"Skipping unchanged file: {rel_path}")
                return False

            # File is new or changed, remove previous chunks if they exist
            if rel_path in self.indexed_files_meta:
                self._remove_previous_chunks(rel_path)

            # Create new chunks and add to RAG
            chunks = self._chunk_code_file(file_path, content)

            # Track the document IDs for each chunk by source
            for chunk in chunks:
                # Add the chunk to RAG manager
                rag_manager.add_knowledge(
                    text=chunk["text"],
                    source=chunk["source"]
                )

                # Get the document ID for this chunk (it's the last one added)
                doc_id = rag_manager.index.ntotal - 1

                # Track the source -> ID mapping
                source = chunk["source"]
                if source not in self.source_to_ids:
                    self.source_to_ids[source] = []
                self.source_to_ids[source].append(doc_id)

            # Update the indexed files metadata
            self.indexed_files_meta[rel_path] = content_hash

            return True

        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            return False

    def index_directory(self, dir_path: str = None) -> int:
        """
        Recursively index all code files in the given directory.
        Returns the number of files indexed.
        """
        if dir_path is None:
            dir_path = self.repo_root

        files_indexed = 0

        try:
            for root, dirs, files in os.walk(dir_path):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if d not in self.exclude_dirs and not d.startswith('.')]

                for file in files:
                    file_path = os.path.join(root, file)
                    if self.index_file(file_path):
                        files_indexed += 1

            return files_indexed

        except Exception as e:
            logger.error(f"Error indexing directory {dir_path}: {e}")
            return files_indexed

    def refresh_index(self) -> int:
        """
        Re-index only the files that have changed since the last indexing.
        Returns the number of files updated.
        """
        files_updated = 0

        try:
            # Get list of files tracked by git
            result = subprocess.run(
                ['git', 'ls-files', '--full-name'],
                capture_output=True, text=True, check=True,
                cwd=self.repo_root
            )
            all_files = result.stdout.strip().split('\n')

            # Get list of modified files
            result = subprocess.run(
                ['git', 'ls-files', '--modified', '--full-name'],
                capture_output=True, text=True, check=True,
                cwd=self.repo_root
            )
            modified_files = result.stdout.strip().split('\n') if result.stdout.strip() else []

            # Index all new and modified files
            for file in all_files:
                if not file:
                    continue

                file_path = os.path.join(self.repo_root, file)

                # If file is not already indexed or has been modified
                is_modified = file in modified_files
                rel_path = os.path.relpath(file_path, self.repo_root)

                if rel_path not in self.indexed_files_meta or is_modified:
                    if self.index_file(file_path):
                        files_updated += 1

            return files_updated

        except Exception as e:
            logger.error(f"Error refreshing index: {e}")
            return files_updated

    def get_indexed_files_count(self) -> int:
        """Return the number of indexed files."""
        return len(self.indexed_files_meta)

    def get_stats(self) -> Dict:
        """Return statistics about the indexed codebase."""
        file_extensions = {}

        for file_path in self.indexed_files_meta.keys():
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            if ext in file_extensions:
                file_extensions[ext] += 1
            else:
                file_extensions[ext] = 1

        return {
            "total_files": len(self.indexed_files_meta),
            "file_extensions": file_extensions,
            "knowledge_entries": rag_manager.index.ntotal if hasattr(rag_manager, 'index') else 0
        }

# Create a global instance for use across the application
code_indexer = CodeIndexer()