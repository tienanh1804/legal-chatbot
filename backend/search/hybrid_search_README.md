# Hybrid Search Implementation

## Overview

Hybrid Search combines vector search and BM25 lexical search to improve retrieval quality in the Vietnamese legal document Q&A system. This approach leverages both semantic understanding and keyword matching for optimal results, addressing the limitations of each method when used alone.

## Core Components

### 1. Vector Search

Vector search uses dense embeddings to capture semantic relationships between documents and queries:

- **Embedding Generation**: Documents and queries are converted into high-dimensional vectors (embeddings) using a fine-tuned SentenceTransformer model specialized for Vietnamese legal text.

- **Semantic Matching**: The system calculates cosine similarity between query embeddings and document embeddings to find semantically similar content, even when exact keywords don't match.

- **Advantages**: Captures semantic relationships and contextual meaning, handles synonyms and related concepts effectively.

### 2. BM25 Search

BM25 is a probabilistic retrieval model that excels at keyword matching:

- **Term Frequency-Inverse Document Frequency**: BM25 extends the TF-IDF concept with better handling of document length and term saturation.

- **Parameter Optimization**: The implementation uses optimized parameters (k1 and b) specifically tuned for Vietnamese legal documents, where k1 controls term frequency saturation and b controls document length normalization.

- **Advantages**: Excellent at exact keyword matching, handles specialized terminology well, and gives higher scores to documents with multiple query term occurrences.

### 3. Dynamic Weighting

The system dynamically adjusts the balance between vector and BM25 scores:

- **Query Analysis**: Analyzes the query to determine if it contains legal terminology or specific patterns.

- **Adaptive Alpha**: Adjusts the alpha parameter (weight for vector search) based on query characteristics. Legal queries with specific terminology receive a lower alpha value to prioritize BM25 matching, while more conceptual queries receive a higher alpha to prioritize semantic matching.

- **Legal Term Boosting**: Applies additional score boosts for documents that match specific legal terms in the query.

### 4. Query Expansion

Query expansion enhances the original query to improve recall:

- **Term Extraction**: Analyzes top-ranked documents from an initial search to identify relevant terms.

- **Synonym Addition**: Adds legal synonyms and related terms to the query.

- **Bigram Generation**: Creates word pairs from the query to capture legal phrases that should be treated as single units.

## Search Process in Detail

### 1. Text Preprocessing

The system applies specialized preprocessing for Vietnamese legal text:

- **Vietnamese Tokenization**: Uses pyvi library to properly tokenize Vietnamese text, preserving word boundaries and compound words.

- **Stopword Removal**: Removes common Vietnamese stopwords while preserving legal terminology that might otherwise be considered stopwords.

- **N-gram Generation**: Creates bigrams and trigrams for important legal phrases to ensure they're treated as single units during search.

- **Case Normalization**: Converts text to lowercase while preserving special characters important in legal documents.

### 2. Parallel Search Execution

The system executes vector and BM25 searches in parallel:

- **Vector Search**: Calculates cosine similarity between the query embedding and all document embeddings.

- **BM25 Search**: Processes the tokenized (and possibly expanded) query through the BM25 algorithm to get keyword-based relevance scores.

- **Legal Term Matching**: Separately counts matches of specific legal terminology to provide additional scoring signals.

### 3. Score Combination

The scores from different methods are combined using a sophisticated weighting approach:

- **Score Normalization**: Both vector and BM25 scores are normalized to a 0-1 range to ensure fair comparison.

- **Weighted Combination**: The normalized scores are combined using the formula: `hybrid_score = (alpha * vector_score) + ((1 - alpha) * bm25_score) + (legal_term_boost)`.

- **Dynamic Adjustment**: The alpha parameter is adjusted based on query characteristics, typically ranging from 0.2-0.4 for legal queries (favoring BM25) and 0.4-0.6 for more conceptual queries (favoring vector search).

### 4. Result Reranking

After initial scoring, results undergo reranking to improve precision:

- **Document Structure Analysis**: Examines document structure to prioritize well-formatted legal documents.

- **Content Density Evaluation**: Calculates the density of query terms in each document relative to document length.

- **Quality Scoring**: Applies additional scoring based on document metadata like source authority and publication date.

- **Threshold Filtering**: Filters results by keeping only documents with scores above a relative threshold (typically 50-70% of the top score).

## Caching System

The implementation uses a comprehensive caching system to optimize performance:

### 1. Document Cache

- **Purpose**: Stores preprocessed document content to avoid repeated file I/O and text processing.
- **Implementation**: Uses pickle serialization with document IDs as keys and preprocessed content as values.
- **Benefits**: Reduces document loading time from seconds to milliseconds.

### 2. Embeddings Cache

- **Purpose**: Stores pre-computed document embeddings to avoid repeated computation of these resource-intensive vectors.
- **Implementation**: Serializes embedding vectors with document IDs as keys.
- **Benefits**: Reduces embedding generation time from minutes to milliseconds for the entire corpus.

### 3. BM25 Index Cache

- **Purpose**: Stores the BM25 model, corpus, and document ID mappings.
- **Implementation**: Serializes the complete BM25 index for instant loading.
- **Benefits**: Avoids rebuilding the BM25 index on each application start, reducing startup time significantly.

### 4. Metadata Cache

- **Purpose**: Stores document metadata like titles, sources, and publication dates.
- **Implementation**: Serializes metadata dictionaries for quick access during result formatting.
- **Benefits**: Enables rich result presentation without additional database queries.

### Cache Invalidation and Rebuilding

The system includes mechanisms to detect when caches need to be rebuilt:

- **Version Checking**: Caches store version information to detect incompatible changes.
- **Document Set Validation**: Verifies that cached document IDs match the current document set.
- **Forced Rebuilding**: Provides a command-line option to force cache rebuilding when needed.

## Integration with RAG Pipeline

The hybrid search is seamlessly integrated into the Retrieval-Augmented Generation pipeline:

### 1. Query Classification

- The system first determines if a query is legal-related using a trained classifier.
- Legal queries trigger the hybrid search pipeline, while general queries bypass retrieval.

### 2. Document Retrieval

- For legal queries, the system retrieves relevant documents using hybrid search.
- The retrieved documents are formatted with their content and metadata.

### 3. Context Formation

- The content from retrieved documents is combined to form a comprehensive context.
- This context is structured to highlight the most relevant information first.

### 4. Answer Generation

- The query, context, and conversation history are passed to the Gemini API.
- A specialized prompt instructs the model to generate answers based solely on the provided context.

### 5. Source Attribution

- The system formats source information from the retrieved documents.
- These sources are presented alongside the answer for transparency and verification.

## Performance Optimization Techniques

### 1. Result Caching

- **Mechanism**: Stores the last query and its results in memory.
- **Benefit**: Enables instant response for repeated queries without reprocessing.
- **Implementation**: Uses a global variable with query string as key and full result set as value.

### 2. Dynamic Alpha Adjustment

- **Mechanism**: Analyzes query characteristics to determine optimal weighting between vector and BM25 scores.
- **Benefit**: Improves precision by favoring the most appropriate search method for each query type.
- **Implementation**: Adjusts alpha based on presence of legal terminology, query length, and query structure.

### 3. Legal Term Boosting

- **Mechanism**: Identifies legal terminology in queries and gives additional weight to documents containing these terms.
- **Benefit**: Improves precision for queries containing specific legal references.
- **Implementation**: Maintains a list of legal terms and applies score boosts when matches are found.

### 4. Threshold Filtering

- **Mechanism**: Filters results by keeping only documents with scores above a relative threshold.
- **Benefit**: Removes marginally relevant documents that might dilute the context quality.
- **Implementation**: Calculates a threshold as a percentage of the top document score (typically 50%).

### 5. Conversation Context Integration

- **Mechanism**: Incorporates previous questions and answers from the same conversation.
- **Benefit**: Enables the system to maintain context across multiple interactions.
- **Implementation**: Retrieves conversation history from the database and includes it in the prompt.

## Dependencies and Their Roles

- **rank-bm25 (0.2.2)**: Provides the core BM25Okapi algorithm implementation for keyword-based retrieval.

- **scikit-learn (1.6.1)**: Used for vector operations, cosine similarity calculations, and normalization functions.

- **sentence-transformers (3.3.1)**: Provides the embedding models for converting text to vector representations.

- **pyvi**: Vietnamese text processing library used for tokenization.
