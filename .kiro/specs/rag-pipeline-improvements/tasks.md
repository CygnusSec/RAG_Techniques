# Implementation Plan: RAG Pipeline Improvements

## Overview

This plan implements 10 improvements to the RAG experimental pipeline: source ID standardization, evaluation granularity, config correction, semantic chunk metadata, compression fallback, cross-encoder reranker, sufficiency classification fix, router improvements, generation metrics, and final evaluation completeness. All shared logic goes into `report_common/`, configuration into `experiment.yaml`, and notebooks consume the shared utilities.

## Tasks

- [x] 1. Set up dependencies and config changes
  - [x] 1.1 Add new dependencies to requirements-report.txt
    - Add `sentence-transformers>=2.2.0`, `hypothesis>=6.82.0`, `pytest>=7.4.0`, `pytest-mock>=3.11.0` to `requirements-report.txt`
    - _Requirements: 6.1, 6.4_

  - [x] 1.2 Fix chunk_unit in config/experiment.yaml and add reranker section
    - Change `baseline.chunk_unit` from `"token"` to `"character"`
    - Add `reranker` section with `model_name: cross-encoder/ms-marco-MiniLM-L-6-v2` and `top_k: 5`
    - _Requirements: 3.1, 6.3_

  - [x] 1.3 Update print_config_summary in report_common/config.py
    - Ensure the summary prints the corrected chunk_unit value as `"character"`
    - Verify output format matches: `Chunk size   : <value> character`
    - _Requirements: 3.2_

- [x] 2. Implement Source ID utility functions
  - [x] 2.1 Add source ID functions to report_common/evaluation.py
    - Add `SOURCE_ID_PATTERN` and `PAGE_ID_PATTERN` regex constants
    - Implement `format_page_id(filename, page_number)` returning `{filename}::page_{N}`
    - Implement `format_chunk_id(filename, page_number, chunk_index)` returning `{filename}::page_{N}::chunk_{M}`
    - Implement `extract_page_id(chunk_id)` that parses chunk IDs and returns the page prefix or None
    - _Requirements: 1.1, 1.2, 1.5, 1.6_

  - [ ]* 2.2 Write property test for Source ID Round-Trip
    - **Property 1: Source ID Round-Trip**
    - **Validates: Requirements 1.1, 1.2, 1.5**
    - Create `tests/test_source_ids.py` with Hypothesis test: for any valid filename (no `::`), non-negative page number, and non-negative chunk index, `extract_page_id(format_chunk_id(...))` returns the expected page prefix

  - [x] 2.3 Enhance compute_retrieval_metrics with granularity parameter
    - Add `granularity` parameter (`"chunk"`, `"page"`, `"auto"`) defaulting to `"auto"`
    - Add `relevant_chunks` parameter (optional list of chunk-level ground truth)
    - Implement auto logic: use chunk matching if `relevant_chunks` non-empty, else extract page prefixes and match against `relevant_ids`
    - Handle fallback from chunk to page when `relevant_chunks` is empty
    - _Requirements: 1.4, 1.5, 1.6, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 2.4 Write property test for Retrieval Metrics Bounds
    - **Property 2: Retrieval Metrics Bounds**
    - **Validates: Requirements 1.4, 2.2**
    - In `tests/test_retrieval_metrics.py`: for any lists of retrieved/relevant IDs with k > 0, all metrics are within [0, 1] and hit_rate is 1.0 iff at least one retrieved ID matches

  - [ ]* 2.5 Write property test for Page-Level Fallback Matching
    - **Property 3: Page-Level Fallback Matching**
    - **Validates: Requirements 1.5, 2.3, 2.5**
    - In `tests/test_source_ids.py`: when relevant_chunks is empty and granularity is "auto", hit_rate equals manual page-prefix comparison

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement semantic chunk metadata utilities
  - [x] 4.1 Add page offset tracking functions to report_common/evaluation.py
    - Implement `compute_page_offsets(pages, separator)` returning `(full_text, offsets)`
    - Implement `find_page_for_position(position, offsets, page_lengths)` using binary search
    - _Requirements: 4.1, 4.6_

  - [x] 4.2 Implement attach_metadata_to_semantic_chunks
    - Add `attach_metadata_to_semantic_chunks(pages, semantic_chunks, filename, separator)` to `report_common/evaluation.py`
    - Track character offsets to find which page each chunk starts in
    - Discard empty/whitespace chunks, assign sequential chunk_index, set source_id in standardized format
    - _Requirements: 4.2, 4.3, 4.4, 4.5_

  - [ ]* 4.3 Write property test for Character Offset Page Assignment
    - **Property 4: Character Offset Page Assignment**
    - **Validates: Requirements 4.1, 4.3, 4.6**
    - In `tests/test_offset_tracking.py`: for any list of non-empty page texts, `find_page_for_position` returns the correct page index for positions within page content

  - [ ]* 4.4 Write property test for Semantic Chunk Metadata Invariants
    - **Property 5: Semantic Chunk Metadata Invariants**
    - **Validates: Requirements 4.2, 4.4, 4.5**
    - In `tests/test_offset_tracking.py`: output chunks all have required metadata fields, valid source_id format, no empty content, and contiguous chunk_index starting at 0

- [x] 5. Implement compression fallback module
  - [x] 5.1 Create report_common/compression.py with compress_with_fallback
    - Create new module `report_common/compression.py`
    - Implement `compress_with_fallback(query, documents, compress_fn, min_length=50)` with the full fallback logic
    - Handle: empty/short output → use full text; "NO_RELEVANT_CONTENT" → exclude doc; all fail → use first doc
    - Return `(context_text, fallback_triggered)` tuple
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 5.2 Write property test for Compression Fallback Guarantees
    - **Property 6: Compression Fallback Guarantees**
    - **Validates: Requirements 5.1, 5.4, 5.5**
    - In `tests/test_compression.py`: for any non-empty document list (page_content >= 50 chars), output is always >= 50 chars and fallback flag is correctly set

- [x] 6. Implement cross-encoder reranker
  - [x] 6.1 Add build_cross_encoder and cross_encoder_rerank to report_common/models.py
    - Import `CrossEncoder` from `sentence_transformers`
    - Implement `build_cross_encoder(config)` reading from `config['reranker']['model_name']`
    - Implement `cross_encoder_rerank(query, documents, cross_encoder, top_k)` with batch predict and sorting
    - Handle empty documents list, truncate page_content to 512 chars
    - Raise `RuntimeError` on model load failure
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.7_

  - [ ]* 6.2 Write unit tests for cross-encoder reranker
    - In `tests/integration/test_cross_encoder.py`: test batch predict call, empty list handling, RuntimeError on load failure
    - _Requirements: 6.2, 6.5_

- [x] 7. Implement router and sufficiency parsing module
  - [x] 7.1 Create report_common/router.py with parsing functions
    - Create new module `report_common/router.py`
    - Define `CATEGORIES` list: IDENTIFIER, SEMANTIC, MULTI_HOP, CONFLICTING, OUT_OF_SCOPE
    - Implement `parse_route_response(response)` with category extraction and SEMANTIC default
    - Implement `parse_sufficiency_response(response)` checking INSUFFICIENT before SUFFICIENT
    - Implement `prepare_sufficiency_context(documents, max_chars=1500)` concatenating top-3 docs
    - _Requirements: 7.1, 7.2, 7.3, 7.5, 8.3, 8.4, 8.5_

  - [ ]* 7.2 Write property test for Sufficiency Parsing Priority
    - **Property 7: Sufficiency Parsing Priority**
    - **Validates: Requirements 7.1, 7.3**
    - In `tests/test_sufficiency_parser.py`: any string containing "INSUFFICIENT" returns "INSUFFICIENT"; strings with neither keyword default to "INSUFFICIENT"

  - [ ]* 7.3 Write property test for Router Default Fallback
    - **Property 8: Router Default Fallback**
    - **Validates: Requirements 8.3**
    - In `tests/test_router_parser.py`: any string not containing any category keyword returns "SEMANTIC"

- [x] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement generation metrics
  - [x] 9.1 Add compute_faithfulness, compute_relevancy, compute_hallucination_rate to report_common/evaluation.py
    - Implement `compute_faithfulness(answer, context, llm=None)` with claim decomposition
    - Implement `compute_relevancy(answer, question, llm=None)` with relevance scoring
    - Implement `compute_hallucination_rate(answer, context, llm=None)` as complement of faithfulness
    - Handle edge cases: empty answer → 0.0, empty context → 0.0/1.0, LLM failure → raise exception
    - Ensure faithfulness + hallucination_rate ≈ 1.0 (tolerance 0.01)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_

  - [ ]* 9.2 Write property test for Faithfulness and Hallucination Complementarity
    - **Property 9: Faithfulness and Hallucination Complementarity**
    - **Validates: Requirements 9.1, 9.3, 9.4**
    - In `tests/test_generation_metrics.py`: for any (supported_claims, total_claims) with total > 0, faithfulness + hallucination_rate == 1.0 within tolerance 0.01

  - [ ]* 9.3 Write unit tests for generation metrics edge cases
    - Test empty answer returns 0.0, empty context returns 0.0/1.0, LLM failure raises exception
    - _Requirements: 9.6, 9.7, 9.8_

- [x] 10. Implement final evaluation utilities and notebook updates
  - [x] 10.1 Add experiment discovery and comparison functions to report_common/evaluation.py or io.py
    - Implement `discover_experiments(results_dir)` using glob `A*.jsonl` with alphanumeric sort
    - Implement `build_comparison_table(experiments)` producing one row per experiment with all metrics
    - Implement `build_category_breakdown(experiment_records, questions)` for per-category averages
    - Implement `build_pairwise_deltas(experiments, baseline_name="A0")` for delta comparison
    - _Requirements: 10.1, 10.2, 10.3, 10.6_

  - [ ]* 10.2 Write property test for Evaluation Summary Completeness
    - **Property 10: Evaluation Summary Completeness**
    - **Validates: Requirements 10.1, 10.2, 10.6**
    - In `tests/test_final_evaluation.py`: for any set of experiment result dicts, summary has exactly one row per experiment and contains mean of every numeric metric

- [ ] 11. Update notebooks to use new shared utilities
  - [ ] 11.1 Update notebook 01 (01_naive_rag.ipynb) with source ID assignment
    - Import `format_page_id`, `format_chunk_id` from `report_common.evaluation`
    - Assign standardized source_id to each chunk after splitting
    - Add code comment that chunk_size and overlap are measured in characters
    - _Requirements: 1.1, 1.2, 3.3_

  - [ ] 11.2 Update notebook 03 (03_semantic_chunking.ipynb) with metadata attachment
    - Import `attach_metadata_to_semantic_chunks` from `report_common.evaluation`
    - Replace existing semantic chunking logic with call to shared utility
    - Ensure all semantic chunks have proper source, page, chunk_index metadata and source_id
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [ ] 11.3 Update notebook 06 (06_reranking.ipynb) with cross-encoder reranker
    - Import `build_cross_encoder`, `cross_encoder_rerank` from `report_common.models`
    - Replace LLM-based reranking with cross-encoder batch scoring
    - Record `rerank_seconds` in the result record latency field
    - _Requirements: 6.1, 6.2, 6.6_

  - [ ] 11.4 Update notebook 07 (07_compression.ipynb) with fallback logic
    - Import `compress_with_fallback` from `report_common.compression`
    - Replace inline compression logic with call to shared utility
    - Record `compression_fallback` boolean in result records
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ] 11.5 Update notebook 09 (09_adaptive_crag.ipynb) with router and sufficiency fixes
    - Import `parse_route_response`, `parse_sufficiency_response`, `prepare_sufficiency_context` from `report_common.router`
    - Add few-shot examples to router prompt (2-3 per category including CONFLICTING)
    - Fix sufficiency parsing to check INSUFFICIENT before SUFFICIENT
    - Handle CONFLICTING and OUT_OF_SCOPE routes per design
    - Record `route` field in result records
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [ ] 11.6 Update notebook 10 (10_final_evaluation.ipynb) with dynamic discovery and generation metrics
    - Import `discover_experiments`, `build_comparison_table`, `build_category_breakdown`, `build_pairwise_deltas` from shared utilities
    - Replace hardcoded file references with dynamic discovery
    - Add faithfulness, relevancy, hallucination_rate columns to comparison table
    - Add per-category breakdown for all experiments
    - Add pairwise delta comparison against A0 baseline
    - Update title to "Final Evaluation – A0 through A8"
    - Print warning if fewer than 9 experiment files found
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

- [ ] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The project uses Python with LangChain, FAISS, and sentence-transformers
- All shared logic lives in `report_common/`; notebooks import from there
- Tests use Hypothesis for property-based testing and pytest as the test runner

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1", "5.1", "7.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "4.1", "5.2", "6.1", "7.2", "7.3"] },
    { "id": 3, "tasks": ["2.4", "2.5", "4.2", "6.2", "9.1"] },
    { "id": 4, "tasks": ["4.3", "4.4", "9.2", "9.3", "10.1"] },
    { "id": 5, "tasks": ["10.2", "11.1", "11.2", "11.3", "11.4", "11.5"] },
    { "id": 6, "tasks": ["11.6"] }
  ]
}
```
