# Browser-Based RAG System

A lightweight, browser-based RAG (Retrieval-Augmented Generation) implementation using Ollama for local LLM inference. This system provides semantic search capabilities and knowledge base management through an intuitive web interface.

[![Demo](https://img.shields.io/badge/Demo-Live-success)](https://steveseguin.github.io/rag/)
![License](https://img.shields.io/badge/license-MIT-blue)

## Features

- 🌐 Fully client-side implementation with browser persistence
- 🔍 Semantic search using local LLM models
- 📚 Built-in knowledge base management
- 💾 Import/export functionality for knowledge bases
- 🔄 Automatic remote knowledge base loading
- 📊 Real-time processing progress indicators
- 🎯 Token-aware text chunking
- 🧠 Recursive query refinement

## Prerequisites

1. Install [Ollama](https://ollama.ai)
2. Configure Ollama for CORS from HTTPS websites
3. Install required models:
```bash
ollama pull granite-embedding:30m
ollama pull llama3.2:latest
```

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/steveseguin/rag.git
cd rag
```

2. Configure Ollama endpoint in `rag.js`:
```javascript
const OLLAMA_ENDPOINT = "http://localhost:11434";
const EMBEDDING_MODEL = "granite-embedding:30m";
const COMPLETION_MODEL = "llama3.2:latest";
```

3. Serve the files using any web server and open in your browser

## Usage

### Pre-trained Knowledge Base
- By default, the system loads a pre-trained VDO.Ninja knowledge base
- Can be disabled by removing the remote loading functionality

### Custom Knowledge Base
1. Click "Process Files" to upload documents
2. Select files (supported formats: .js, .html, .md, .txt)
3. Monitor processing progress
4. Knowledge base persists in browser storage

### Search Interface
- Enter queries in the search box
- View semantic search results with context
- Results include source attribution

### Knowledge Base Management
- Download current knowledge base
- Load existing knowledge base files
- Clear knowledge base
- View storage statistics

## Architecture

The system uses:
- IndexedDB for embeddings storage
- WebRTC for file processing
- Ollama API for:
  - Text embeddings (granite-embedding:30m)
  - Text generation (llama3.2:latest)

## Features Deep Dive

### Token-Aware Chunking
- Intelligent document splitting
- Context preservation
- Overlap handling

### Semantic Search
- Cosine similarity ranking
- Context scoring
- Recursive query refinement

### Storage Management
- Browser-based persistence
- Import/export capabilities
- Statistics tracking

## Contributing

Pull requests welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and development process.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ollama](https://ollama.ai) for local LLM inference
- [VDO.Ninja](https://vdo.ninja) for the sample knowledge base
