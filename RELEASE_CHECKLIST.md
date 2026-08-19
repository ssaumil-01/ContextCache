# ContextCache Release & Versioning Guide

This document describes the semantic versioning strategy, installation patterns, and the step-by-step checklist to prepare and publish the `rag-cachex` package on PyPI.

---

## 1. Semantic Versioning Strategy

ContextCache strictly adheres to [Semantic Versioning (SemVer) 2.0.0](https://semver.org/).
Every release is versioned in the format: `MAJOR.MINOR.PATCH`

### Increment Rules:
- **`MAJOR` (e.g., `1.0.0`)**: Increment when making incompatible/breaking API changes:
  - Modifying signatures of core cache endpoints (e.g. `ContextCache
.run`, `ContextCache
.get_docs`).
  - Removing supported databases/integrations or changing minimum Python version requirement.
- **`MINOR` (e.g., `0.2.0`)**: Increment when adding backward-compatible functionality:
  - Adding a new vector store integration (e.g., ChromaDB, Pinecone).
  - Adding a new key-value store integration (e.g., Memcached).
  - Adding new cache statistics, eviction algorithms, or calibration tools.
- **`PATCH` (e.g., `0.1.1`)**: Increment when executing backward-compatible bug fixes or minor performance improvements:
  - Fixing connection leak bugs, improving prompt calibration scripts, or dependency security bumps.

---

## 2. Installation Guide

Once published, users can install ContextCache under various configurations depending on their backend requirements.

### Core Installation (Minimal Dependencies)
Installs only core schemas (`pydantic`), the Redis integration (`redis`), and Prometheus telemetry (`prometheus-client`). Fits production pipelines using a remote Redis server and custom/external vector stores or remote embeddings (e.g., OpenAI).
```bash
pip install rag-cachex
```

### Installation with Local FAISS Support
Installs core dependencies plus `faiss-cpu` and `numpy` to support local high-performance vector operations.
```bash
pip install "rag-cachex[faiss]"
```

### Installation with Offline Embeddings Support
Installs core dependencies plus `sentence-transformers` to run offline local embeddings natively on the host machine.
```bash
pip install "rag-cachex[embeddings]"
```

### Complete Installation (All Backends)
Installs all of the above optional backends.
```bash
pip install "rag-cachex[all]"
```

### Editable Development Installation
To install the package in editable mode with development dependencies:
```bash
pip install -e .[all]
pip install pytest black isort mypy
```

---

## 3. PyPI Publication Checklist

Follow these steps for every release. Do **NOT** publish directly without going through this checklist.

### Step 1: Pre-Release Verification
1. [ ] **Update version number**: Update the `version` field in `pyproject.toml` (e.g. `version = "0.1.0"`).
2. [ ] **Code Formatting**: Format the codebase:
   ```bash
   black rag_cache/ examples/
   isort rag_cache/ examples/
   ```
3. [ ] **Run Linter / Type Checks**: Run static analysis:
   ```bash
   mypy rag_cache/
   ```
4. [ ] **Run Unit & Integration Tests**: Verify all tests pass:
   ```bash
   pytest
   ```

### Step 2: Build Package Distributions
1. [ ] Ensure build dependencies are up to date:
   ```bash
   python3 -m pip install --upgrade build twine
   ```
2. [ ] Clean old build artifacts:
   ```bash
   rm -rf build/ dist/ *.egg-info
   ```
3. [ ] Build the source archive and binary wheel:
   ```bash
   python3 -m build
   ```
4. [ ] Verify distribution packages structure and metadata:
   ```bash
   python3 -m twine check dist/*
   ```

### Step 3: Publish to TestPyPI (Staging)
1. [ ] Upload the build artifacts to TestPyPI:
   ```bash
   python3 -m twine upload --repository testpypi dist/*
   ```
   *Note: Securely log in using your PyPI API token prefix `pypi-...`.*
2. [ ] Test the installation in a fresh virtual environment:
   ```bash
   python3 -m venv test_env
   source test_env/bin/activate
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ rag-cachex[all]
   ```
3. [ ] Verify imports and basic quickstart works inside `test_env`:
   ```bash
   python3 -c "from rag_cache import ContextCache
; print(ContextCache
)"
   ```

### Step 4: Publish to PyPI (Production)
1. [ ] Upload build distributions to production PyPI:
   ```bash
   python3 -m twine upload dist/*
   ```
2. [ ] Verify publication by installing in a new virtual environment:
   ```bash
   pip install rag-cachex[all]
   ```

### Step 5: Post-Release Tagging
1. [ ] Tag the commit in git with release version:
   ```bash
   git tag -a v0.1.2 -m "Release version 0.1.2"
   git push origin v0.1.2
   ```
2. [ ] Create a GitHub release under the tagged commit, listing key updates and performance enhancements.
