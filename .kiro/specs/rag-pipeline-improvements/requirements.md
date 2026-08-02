# Requirements Document

## Introduction

This document specifies improvements to the RAG experimental pipeline for the Information Retrieval course project. The pipeline consists of 10 experiment notebooks (A0–A8 + final evaluation) that progressively enhance a Retrieval-Augmented Generation system evaluated against a 30-question ground-truth set over a Climate Change corpus. The improvements address source ID alignment, evaluation granularity, configuration correctness, metadata propagation, compression robustness, reranker efficiency, classification logic, router quality, evaluation metrics, and final evaluation completeness.

## Glossary

- **Pipeline**: The complete RAG experimental system comprising `report_common/` utilities, `config/experiment.yaml`, and `report_demo/` notebooks
- **Source_ID**: The identifier string extracted from chunk metadata used to match retrieved documents against ground-truth relevant documents
- **Ground_Truth_ID**: The identifier string in `questions.json` field `relevant_documents` or `relevant_chunks` used as reference for evaluation
- **Chunk**: A segment of text produced by splitting source documents; the unit of retrieval in the vector store
- **Page_ID**: A composite identifier in the format `{filename}::page_{N}` that identifies a specific page within a PDF document
- **Chunk_ID**: A composite identifier in the format `{filename}::page_{N}::chunk_{M}` that identifies a specific chunk within a page
- **Evaluation_Module**: The file `report_common/evaluation.py` containing retrieval and generation metric functions
- **Config**: The file `config/experiment.yaml` containing experiment parameters
- **Semantic_Chunker**: The `SemanticChunker` from `langchain_experimental` used in notebook 03
- **Compression_Module**: The LLM-based extractive compression logic in notebook 07
- **Cross_Encoder**: A neural reranker model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) that scores (query, document) pairs directly
- **Router**: The query classification component in notebook 09 that routes queries to appropriate retrieval strategies
- **Sufficiency_Evaluator**: The LLM-based component in notebook 09 that classifies retrieved context as SUFFICIENT or INSUFFICIENT
- **Final_Evaluation_Notebook**: The notebook `report_demo/10_final_evaluation.ipynb` that aggregates and compares all experiment results
- **Faithfulness**: A generation metric measuring whether the answer is grounded in the retrieved context
- **Relevancy**: A generation metric measuring whether the answer addresses the question asked
- **Hallucination_Rate**: A generation metric measuring the fraction of answer claims not supported by the retrieved context

## Requirements

### Requirement 1: Standardize Source IDs to Match Ground-Truth IDs

**User Story:** As a researcher, I want retrieved document IDs to match the format used in ground-truth evaluation data, so that retrieval metrics (hit_rate, precision, recall, MRR, nDCG) are computed correctly.

#### Acceptance Criteria

1. WHEN a PDF page is loaded by PyPDFLoader, THE Pipeline SHALL assign each page a Source_ID in the format `{filename}::page_{N}` where `{filename}` is the PDF file name without path prefix and `{N}` is the zero-indexed page number
2. WHEN a chunk is produced by a text splitter, THE Pipeline SHALL assign each chunk a Source_ID in the format `{filename}::page_{N}::chunk_{M}` where `{M}` is the zero-indexed sequential chunk index within that page
3. THE Ground_Truth_ID values in `questions.json` field `relevant_chunks` SHALL use the same `{filename}::page_{N}::chunk_{M}` format as the chunk Source_IDs
4. WHEN `compute_retrieval_metrics` is called, THE Evaluation_Module SHALL compare retrieved chunk Source_IDs against Ground_Truth_IDs using case-sensitive exact string matching on the standardized format
5. IF a question has `relevant_chunks` as an empty list but `relevant_documents` is populated, THEN THE Evaluation_Module SHALL fall back to page-level matching by extracting the `{filename}::page_{N}` prefix from chunk Source_IDs and matching against Ground_Truth_IDs in `{filename}::page_{N}` format
6. IF a retrieved Source_ID does not conform to the expected `{filename}::page_{N}::chunk_{M}` pattern, THEN THE Evaluation_Module SHALL treat that Source_ID as non-matching for that question

### Requirement 2: Move Evaluation Granularity from File-Level to Page/Chunk-Level

**User Story:** As a researcher, I want evaluation to operate at page or chunk granularity, so that retrieval metrics reflect whether the correct specific section was retrieved rather than just the correct file.

#### Acceptance Criteria

1. THE `questions.json` evaluation set SHALL include a `relevant_chunks` field for each question containing zero or more Chunk_IDs in the format `{filename}::page_{N}::chunk_{M}` identifying the specific chunks that contain the answer
2. WHEN `relevant_chunks` is non-empty for a question and granularity is `"chunk"` or `"auto"`, THE Evaluation_Module SHALL compute retrieval metrics by performing exact string matching of retrieved Source_IDs against the entries in the `relevant_chunks` field
3. WHEN `relevant_chunks` is empty and `relevant_documents` is non-empty for a question and granularity is `"page"` or `"auto"`, THE Evaluation_Module SHALL compute retrieval metrics by extracting the `{filename}::page_{N}` prefix from each retrieved Source_ID and matching it against `relevant_documents` entries expanded to their page-level identifiers in the same `{filename}::page_{N}` format
4. THE Evaluation_Module SHALL expose a parameter `granularity` with allowed values `"chunk"`, `"page"`, or `"auto"`, defaulting to `"auto"`, to control matching behavior
5. IF granularity is set to `"chunk"` and `relevant_chunks` is empty for a question, THEN THE Evaluation_Module SHALL fall back to page-level matching for that question and include a warning indicator in the per-question result record
6. IF a retrieved Source_ID does not contain a recognized `{filename}::page_{N}` prefix during page-level matching, THEN THE Evaluation_Module SHALL treat that Source_ID as non-matching for that question

### Requirement 3: Correct Chunk Unit Configuration from Token to Character

**User Story:** As a researcher, I want the configuration to accurately reflect that `RecursiveCharacterTextSplitter` with `length_function=len` measures chunk size in characters, so that experiment documentation and reproducibility are correct.

#### Acceptance Criteria

1. THE Config field `baseline.chunk_unit` in `config/experiment.yaml` SHALL have value `"character"` instead of `"token"`
2. WHEN `print_config_summary` displays the chunk size, THE Config module SHALL print the unit as `"character"` in the format `Chunk size   : <value> character`
3. THE `report_demo/` notebooks that instantiate `RecursiveCharacterTextSplitter` with `length_function=len` SHALL include a code comment within the same cell as the splitter instantiation stating that the chunk size and overlap values are measured in characters

### Requirement 4: Attach Metadata to Semantic Chunks

**User Story:** As a researcher, I want semantic chunks to retain source metadata (filename, page number), so that retrieval evaluation and citation tracking work correctly for the semantic chunking experiment.

#### Acceptance Criteria

1. WHEN the Semantic_Chunker produces chunks from combined page text, THE Pipeline SHALL determine each chunk's source page by tracking cumulative character offsets of each page in the concatenated text and assigning the chunk to the page whose offset range contains the chunk's starting character position
2. THE semantic chunk metadata SHALL contain at minimum the fields `source` (PDF filename without path prefix), `page` (zero-indexed page number matching Requirement 1), and `chunk_index` (zero-indexed sequential integer across all semantic chunks produced from the document)
3. WHEN a semantic chunk spans a page boundary (its text content maps to character ranges in two or more pages), THE Pipeline SHALL assign the metadata of the page where the chunk's first character originates
4. THE semantic chunk Source_ID SHALL follow the standardized format `{filename}::page_{N}::chunk_{M}` as defined in Requirement 1, where `{M}` is the global zero-indexed sequential chunk index across all semantic chunks
5. IF the Semantic_Chunker produces a chunk whose `page_content` is empty or contains only whitespace, THEN THE Pipeline SHALL discard that chunk and not assign it a chunk_index (subsequent chunks continue sequential numbering without gaps)
6. WHEN building the concatenated text for semantic splitting, THE Pipeline SHALL insert a page separator (double newline `\n\n`) between consecutive pages and record the cumulative character offset at which each page's content begins in the concatenated string

### Requirement 5: Fix Compression with Fallback to Full Context

**User Story:** As a researcher, I want the compression module to fall back to full context when compression produces empty or minimal output, so that the generation step always has usable context.

#### Acceptance Criteria

1. WHEN the Compression_Module returns empty text, whitespace-only text, or text shorter than 50 characters for a document, THE Pipeline SHALL treat that document's compression as failed and use the document's full uncompressed text in place of its compressed output
2. WHEN the LLM returns `"NO_RELEVANT_CONTENT"` for a document, THE Pipeline SHALL exclude that document from compressed context but retain other documents that produced valid compressed output of 50 characters or more
3. IF all documents return `"NO_RELEVANT_CONTENT"` or all documents produce compressed text shorter than 50 characters, THEN THE Pipeline SHALL use the full uncompressed context of the first document in rerank-score order as fallback
4. THE Pipeline SHALL record whether fallback was triggered in the result record under a boolean field `compression_fallback` set to `true` when any per-document or full fallback was applied, and `false` otherwise
5. WHEN fallback is triggered, THE Pipeline SHALL ensure the final context passed to the generation step contains at least 50 characters of text

### Requirement 6: Replace LLM Reranker with Cross-Encoder

**User Story:** As a researcher, I want to use a cross-encoder neural reranker instead of the LLM-based scoring reranker, so that reranking is faster, cheaper, and more consistent.

#### Acceptance Criteria

1. THE Pipeline SHALL use a Cross_Encoder model (from the `sentence-transformers` library) to score (query, document) pairs for reranking
2. WHEN reranking is performed, THE Pipeline SHALL pass all candidate (query, document) pairs to the Cross_Encoder in a single batch `predict` call, replacing all individual LLM-based relevance scoring calls
3. THE Config SHALL include a `reranker` section with fields `model_name` (cross-encoder model identifier, default: `cross-encoder/ms-marco-MiniLM-L-6-v2`) and `top_k` (number of documents to return after reranking, default: value of `retrieval.final_top_k`)
4. WHEN the Cross_Encoder model is not available locally, THE Pipeline SHALL download the model on first use and cache it to the default `sentence-transformers` cache directory
5. IF the Cross_Encoder model fails to load or download, THEN THE Pipeline SHALL raise an error indicating the model name and the failure reason, and SHALL NOT fall back to LLM-based reranking
6. THE Pipeline SHALL record the reranking latency per query in seconds (as a float under the field `rerank_seconds` in the result record) for comparison with the previous LLM-based approach
7. WHEN scoring candidates, THE Pipeline SHALL pass the full chunk text (up to the Cross_Encoder model's maximum input length of 512 tokens) as the document input to the Cross_Encoder

### Requirement 7: Fix SUFFICIENT/INSUFFICIENT Classification Logic

**User Story:** As a researcher, I want the sufficiency evaluator to correctly detect when context is insufficient and to properly handle edge cases, so that the corrective retrieval loop triggers appropriately.

#### Acceptance Criteria

1. WHEN the Sufficiency_Evaluator parses the LLM response, THE Sufficiency_Evaluator SHALL check for the substring "INSUFFICIENT" before checking for "SUFFICIENT", and if "INSUFFICIENT" is found, classify the response as INSUFFICIENT regardless of whether "SUFFICIENT" also appears as a substring
2. WHEN the Sufficiency_Evaluator is called with retrieved documents, THE Sufficiency_Evaluator SHALL concatenate the full page_content of the top-3 ranked documents and pass up to 1500 characters of the concatenated text to the evaluation prompt, without per-document truncation below that total limit
3. IF the LLM response does not contain either the substring "SUFFICIENT" or the substring "INSUFFICIENT" (case-insensitive comparison after stripping whitespace), THEN THE Sufficiency_Evaluator SHALL default to INSUFFICIENT
4. THE Sufficiency_Evaluator prompt SHALL instruct the LLM to determine whether the context contains specific facts, data points, or direct statements that answer the question, and to respond INSUFFICIENT when the context is only topically related but lacks details needed to produce a complete answer
5. WHEN the Sufficiency_Evaluator receives an empty document list (zero documents retrieved), THE Sufficiency_Evaluator SHALL return INSUFFICIENT without invoking the LLM

### Requirement 8: Improve Adaptive CRAG Router

**User Story:** As a researcher, I want the query router to make more accurate routing decisions and handle edge cases, so that each query type uses the optimal retrieval strategy.

#### Acceptance Criteria

1. THE Router prompt SHALL include 2-3 few-shot examples per category (IDENTIFIER, SEMANTIC, MULTI_HOP, CONFLICTING, OUT_OF_SCOPE), where each example consists of a sample question paired with the expected category label
2. WHEN the Router classifies a query as OUT_OF_SCOPE, THE Pipeline SHALL perform a top-1 dense retrieval and proceed with grounded generation only if the retrieved chunk's similarity score meets or exceeds the relevance threshold used by the sufficiency evaluator; otherwise THE Pipeline SHALL abstain
3. WHEN the Router LLM response does not match any known category (IDENTIFIER, SEMANTIC, MULTI_HOP, CONFLICTING, OUT_OF_SCOPE), THE Pipeline SHALL default to SEMANTIC retrieval
4. THE Router SHALL include a CONFLICTING category for questions that contain premises contradicting the corpus, routing them to retrieval followed by a generation step that identifies the false premise and responds that the stated claim is not supported by the available sources
5. THE result records SHALL include the `route` classification field for each question, containing the category string assigned by the Router
6. WHEN the Router classifies a query as CONFLICTING, THE Pipeline SHALL set `predicted_abstain` to true and produce an answer indicating that the question's premise is not supported by the available sources

### Requirement 9: Add Faithfulness, Relevancy, and Hallucination Rate Metrics

**User Story:** As a researcher, I want to evaluate generation quality using Faithfulness, Relevancy, and Hallucination Rate metrics, so that I can measure not just retrieval quality but also answer quality.

#### Acceptance Criteria

1. THE Evaluation_Module SHALL provide a function `compute_faithfulness(answer, context, llm)` that uses the provided LLM to decompose the answer into individual atomic claims, determines how many of those claims are supported by the context, and returns a score equal to (number of supported claims / total number of claims), ranging from 0.0 to 1.0
2. THE Evaluation_Module SHALL provide a function `compute_relevancy(answer, question, llm)` that uses the provided LLM to judge whether the answer addresses the question, and returns a score between 0.0 and 1.0 where 1.0 means the answer fully addresses the question and 0.0 means the answer is completely irrelevant
3. THE Evaluation_Module SHALL provide a function `compute_hallucination_rate(answer, context, llm)` that uses the provided LLM to decompose the answer into individual atomic claims, determines how many of those claims are NOT supported by the context, and returns a score equal to (number of unsupported claims / total number of claims), ranging from 0.0 to 1.0
4. WHEN `compute_faithfulness` and `compute_hallucination_rate` are both called on the same (answer, context) pair, THE Evaluation_Module SHALL ensure that `faithfulness + hallucination_rate` equals approximately 1.0 (within floating point tolerance of 0.01)
5. THE generation metrics functions SHALL accept an optional `llm` parameter; IF `llm` is not provided, THEN THE Evaluation_Module SHALL construct an LLM client using the `llm` section of experiment.yaml (base_url, api_key, model, temperature, max_tokens)
6. IF the `answer` parameter is an empty string, THEN THE Evaluation_Module SHALL return a score of 0.0 without making an LLM call
7. IF the LLM call fails or returns an unparseable response, THEN THE Evaluation_Module SHALL raise an exception with an error message indicating the failure reason, without returning a numeric score
8. IF the `context` parameter is an empty string when calling `compute_faithfulness` or `compute_hallucination_rate`, THEN THE Evaluation_Module SHALL return 0.0 for faithfulness and 1.0 for hallucination_rate without making an LLM call

### Requirement 10: Fix Final Evaluation Notebook to Load All Experiments A0–A8

**User Story:** As a researcher, I want the final evaluation notebook to automatically discover and load results from all experiments A0 through A8, so that the comparison table is complete without manual file name specification.

#### Acceptance Criteria

1. WHEN the Final_Evaluation_Notebook scans for result files, THE Pipeline SHALL discover all JSONL files matching the pattern `A*.jsonl` in the `report_results` directory and sort them alphanumerically by filename stem
2. THE Final_Evaluation_Notebook SHALL produce a comparison table with one row per discovered experiment, including columns for: experiment name, number of questions, mean values for each numeric metric found in the records' `metrics` dictionary, Faithfulness, Relevancy, Hallucination_Rate, and latency statistics (mean and p95 in seconds)
3. THE Final_Evaluation_Notebook SHALL produce a per-category breakdown table for every discovered experiment (not only a single hardcoded experiment), showing metric averages grouped by the `category` field from `report_data/evaluation/questions.json`
4. IF fewer than 9 experiment files (A0 through A8) are found in the results directory, THEN THE Final_Evaluation_Notebook SHALL print a warning message to the notebook output cell listing each missing experiment identifier and continue processing with the available files
5. THE Final_Evaluation_Notebook title markdown cell SHALL read "Final Evaluation – A0 through A8" and the comparison section SHALL dynamically list all discovered experiment names rather than referencing a hardcoded subset
6. WHEN more than one experiment is loaded, THE Final_Evaluation_Notebook SHALL produce a pairwise delta comparison section that dynamically compares all loaded experiments against the baseline (A0) rather than hardcoding a comparison between only two specific experiments
