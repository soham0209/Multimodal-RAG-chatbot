# Multimodal RAG Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) chatbot that integrates text and image data from PDF documents using Pinecone vector databases and Google's Gemini API.

## 📋 Overview

This project implements a multimodal RAG system that can:
- Extract and process text from PDF documents
- Extract and index figures/images from PDFs with automatic caption detection
- Perform hybrid search (dense + sparse embeddings) over text content
- Retrieve and rank relevant documents using advanced reranking
- Generate context-aware answers using Google Gemini with retrieved evidence
- Maintain conversation history for contextual understanding

## 🏗️ Architecture

### Components

1. **PDF Processing**
   - Text extraction using `pdfminer.six`
   - Figure extraction using PyMuPDF with caption detection
   - Paragraph-level chunking with token-aware splitting

2. **Vector Storage**
   - **Pinecone** for scalable vector database management
   - Multiple index types:
     - Dense text index (llama-text-embed-v2)
     - Sparse text index (pinecone-sparse-english-v0)
     - Image caption index

3. **Retrieval Pipeline**
   - Query reformulation using Gemini
   - Dual retrieval: dense + sparse search
   - BGE reranker for result ranking
   - Multi-modal retrieval (text + images)

4. **Generation**
   - Google Gemini 2.5 Flash for answer generation
   - Evidence-based response with figure references
   - Conversation history awareness

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- API Keys:
  - Pinecone API Key
  - Google Gemini API Key
- PDF document to index

### Installation

1. **Install Dependencies**

```bash
pip install PyMuPDF pdfminer.six tiktoken langchain==1.0 langchain-community \
  langchain-google-genai langchain-huggingface langchain-pinecone pinecone \
  pinecone-text rank_bm25 transformers pillow
```

2. **Set Environment Variables**

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
GEMINI_API_KEY=your_google_gemini_api_key
```

3. **Update Configuration**

In the notebook, update these variables:

```python
TEXT_INDEX = "your-text-index-name"
IMAGE_INDEX = "your-image-index-name"
NAMESPACE = "your-namespace"
```

## 📊 Pipeline Workflow

### 1. Data Processing & Storage

#### Extract Text Documents
```python
docs = extract_paragraph_docs("WHO_document.pdf")
```

Extracts paragraphs from PDF with:
- Automatic line merging and hyphenation handling
- Token-aware chunking (max 1200 tokens per chunk)
- Sliding window overlap (120 tokens)
- Page and paragraph metadata tracking

#### Extract Figures
```python
figures, metadata = extract_figures_from_pdf(pdf_path, "figures_method1")
```

Features:
- Automatic caption detection using regex patterns
- Intelligent cropping around figure regions
- Metadata extraction and storage in JSON format
- Support for multiple extraction methods

**Output Structure:**
```json
{
  "figure_number": "1.2",
  "caption": "Figure caption text",
  "filename": "page_1_figure_caption.png",
  "page_number": 1,
  "extraction_date": "2024-01-01T00:00:00"
}
```

#### Build Vectors
```python
text_vectors = build_text_vectors(docs)
image_caption_vectors = build_image_caption_vectors(images)
```

Creates structured vector payloads with:
- Unique IDs for tracking
- Text content preserved for context
- Metadata for filtering and sourcing

### 2. Vector Indexing with Pinecone

#### Create Indexes
- **Dense Text Index**: Uses `llama-text-embed-v2` for semantic understanding
- **Sparse Text Index**: Uses `pinecone-sparse-english-v0` for keyword matching
- **Image Index**: Stores and retrieves figure captions

#### Upsert Vectors
```python
upsert_text(text_vectors)
upsert_images(image_caption_vectors)
```

### 3. Retrieval System

#### Query Reformulation
Converts user queries into standalone search queries using conversation history:
```python
reform_q = reformulate(query, history)
```

#### Hybrid Search
Combines dense and sparse retrieval:

```python
dense_results = text_idx.search(query_terms, top_k=5)
sparse_results = sp_text_ix.search(query_terms, top_k=5)
```

#### Reranking
Uses BGE reranker v2 to rank combined results:
```python
ranked_results = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query=query,
    documents=merged_docs,
    top_n=3
)
```

#### Multi-Modal Retrieval
```python
context_docs, image_docs = retrieve_context(
    query, 
    history=[],
    top_k_text=10,
    top_k_images=3
)
```

### 4. Answer Generation

#### Evidence Serialization
Formats retrieved documents for LLM consumption:
```python
serialized_text = serialize_top_docs(docs)
```

#### Generation with Context
```python
answer, figure_ref = generate_answer_from_retrieval(
    question,
    serialized_text,
    images,
    image_dir="./figures_method1"
)
```

**System Instructions:**
- Use ONLY retrieved evidence (no hallucination)
- Provide concise answers (1-3 sentences)
- Reference figures appropriately
- Maintain conversation context
- Prioritize high-scoring results

## 📁 Project Structure

```
/Users/soham/RAG/
├── Multimodal_RAG_chatbot.ipynb    # Main notebook with full pipeline
├── rag_og.py                        # Original Python implementation
├── Questions.csv                    # Input questions for processing
├── README.md                        # This file
├── figures_method1/                 # Extracted figures and metadata
│   ├── figures_metadata.json        # Figure metadata
│   ├── page_1_figure_*.png          # Extracted figure images
│   └── ...
└── submission.csv                   # Output with answers and figure refs
```

## 🔧 Key Functions

### Text Processing

#### `extract_paragraph_docs(pdf_path, max_tokens=1200, overlap_tokens=120)`
Extracts paragraphs from PDF with token-aware chunking.

**Returns:**
- List of dicts with `page_content` and `metadata` keys

#### `build_text_vectors(docs)`
Converts documents into vector payloads for Pinecone.

**Returns:**
- List of vector dicts with `id` and `chunk_text`

### Image Processing

#### `extract_figures_from_pdf(pdf_path, output_folder="extracted_figures")`
Extracts figures using text formatting detection.

**Features:**
- Caption pattern matching
- Intelligent region cropping
- 2x quality scaling

**Returns:**
- Tuple of (figure_paths, metadata)

#### `extract_figures_alternative(pdf_path, output_folder="extracted_figures_alt")`
Alternative method using text search (more robust for different encodings).

**Returns:**
- Tuple of (figure_paths, metadata)

#### `build_image_caption_vectors(images)`
Creates vector payloads from image metadata.

**Returns:**
- List of vector dicts with caption text

### Retrieval & Generation

#### `retrieve_context(query, history=None, top_k_text=10, top_k_rerank=5, top_k_images=3, img_threshold=0.5)`
Multi-stage retrieval pipeline.

**Process:**
1. Query reformulation with history
2. Dense + sparse search
3. Document reranking
4. Image retrieval

**Returns:**
- Tuple of (ranked_text_docs, image_docs)

#### `generate_answer_from_retrieval(question, doc_texts, artifacts, image_dir)`
Generates answer using Gemini with retrieved evidence.

**Returns:**
- Tuple of (answer_text, figure_reference)

#### `build_submission(input_csv, output_csv)`
Batch processes questions and generates submission file.

**Features:**
- Per-conversation history tracking
- Rate limiting (20s between queries)
- Error handling and partial result saving
- CSV output with question IDs

## 📊 Data Formats

### Document Format
```python
{
    "page_content": "Extracted paragraph text...",
    "metadata": {
        "page": 1,
        "para": 1,
        "sub": 0  # Optional: sub-chunk index for large paragraphs
    }
}
```

### Vector Format (Pinecone)
```python
{
    "id": "text-0",
    "chunk_text": "Document content...",
    "metadata": {...}  # Optional additional metadata
}
```

### Output Format (submission.csv)
```
id,conversation_id,question_id,answer,figure_references
1,conv1,1,"Answer text","Figure 1.2"
2,conv1,2,"Next answer","0"
...
```

## ⚙️ Configuration

### Pinecone Setup

```python
# Index Configuration
TEXT_INDEX = "who-text-index"
IMAGE_INDEX = "who-image-index"
NAMESPACE = "who-pdf"

# Dense Model
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_MODEL = "openai/clip-vit-base-patch32"
```

### LLM Configuration

```python
LLM_MODEL_NAME = "gemini-2.5-flash"

# System Instruction enforces:
# - Evidence-based answers only
# - No hallucination
# - Concise responses
# - Figure citations
# - Context awareness
```

## 🔍 Advanced Usage

### Custom Query Reformulation

Modify the `REFORM_PROMPT` to customize how queries are reformulated based on conversation history.

### Adjusting Retrieval Parameters

```python
context_docs, images = retrieve_context(
    query,
    history=conv_history,
    top_k_text=15,      # More text results
    top_k_rerank=7,     # Rerank more docs
    top_k_images=5,     # Retrieve more images
    img_threshold=0.6   # Higher confidence for images
)
```

### Custom Answer Prompting

Edit `ANSWER_PROMPT` and `SYSTEM_INSTRUCTION` in the generation section to customize:
- Response style and tone
- Citation format
- Evidence prioritization
- Handling of ambiguous queries

## 🐛 Troubleshooting

### Pinecone Connection Issues

```python
# Verify API key
from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
print(pc.list_indexes())
```

### Figure Extraction Not Working

- Verify PDF has extractable text (not scanned images)
- Try alternative extraction method with `extract_figures_alternative()`
- Check caption pattern matches "Figure X.X" or "Figure X"

### Low Quality Answers

- Increase `top_k_text` in retrieval for more context
- Adjust reranker `top_n` parameter
- Check Gemini API key validity
- Verify sufficient retrieved evidence in `retrieved_block`

### Memory Issues with Large PDFs

- Reduce `max_tokens` in `extract_paragraph_docs()` to chunk more aggressively
- Decrease `top_k_text` and `top_k_images` parameters
- Process PDFs in batches rather than all at once

## 📝 Example Usage

### Single Question Answering

```python
# Set up
question = "What were the top causes of death globally in 2021?"
history = []

# Retrieve context
docs, images = retrieve_context(question, history)

# Generate answer
serialized = serialize_top_docs(docs)
answer, fig_ref = generate_answer_from_retrieval(
    question,
    serialized,
    images,
    image_dir="./figures_method1"
)

print(f"Answer: {answer}")
print(f"Figure Reference: {fig_ref}")
```

### Batch Processing

```python
# Process CSV of questions
build_submission(
    input_csv="Questions.csv",
    output_csv="submission.csv"
)
```

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyMuPDF | Latest | PDF text/image extraction |
| pdfminer.six | Latest | Advanced PDF parsing |
| tiktoken | Latest | Token counting |
| langchain | 1.0 | LLM orchestration |
| langchain-google-genai | Latest | Gemini integration |
| langchain-pinecone | Latest | Pinecone integration |
| pinecone | Latest | Vector DB client |
| transformers | Latest | Embedding models |
| Pillow | Latest | Image processing |

## 🎯 Performance Metrics

- **Retrieval Speed**: ~500ms for hybrid search + reranking
- **Generation Speed**: ~2-5s depending on answer length
- **Batch Processing**: ~20s per question (with rate limiting)
- **Storage**: ~1 vector per paragraph + 1 per figure

## 📄 License

This project is provided as-is for research and educational purposes.

## 🤝 Contributing

Feel free to extend this project with:
- Additional embedding models
- Multi-document support
- Real-time indexing
- Web UI interface
- Additional LLM providers

## 📧 Support

For issues or questions, refer to:
- Pinecone Documentation: https://docs.pinecone.io/
- LangChain Documentation: https://python.langchain.com/
- Google Gemini API: https://ai.google.dev/
