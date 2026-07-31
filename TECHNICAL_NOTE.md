# Technical note — Team <name>

## 1. System

<!-- Complete this paragraph and table after the Level-1/2 integration is final. -->

| Stage | What you did | Changed from the scaffold? |
|---|---|---|
| Extraction | | |
| Chunking | | |
| Embeddings / index | | |
| Retrieval | | |
| Answer + citation | | |

## 2. Level 2 — conversational memory

<!-- Replace this example with the real q5 rewrite produced by the final system. -->

```text
q5 as asked:     "Why does that happen?"
rewritten query: "..."
```

## 3. Level 3 — whole-document reasoning

The three levels were developed in parallel. When Level-3 work started, the improved PDF
extraction and chunking from the other workstreams were not yet available, so we initially
worked against the scaffold's page-level index. We first focused on a minimal conditional
multi-hop method: retrieve five results, ask the LLM whether evidence is sufficient, and run
at most one targeted `SEARCH:` query when it is not. Contrary to our initial expectation,
page-level two-hop retrieval added latency without improving evidence recall. The model could
often write a plausible answer from partial evidence and stop, showing that answer sufficiency
is not the same as complete evidence coverage.

We therefore added isolated Level-3 chunking experiments rather than waiting for the shared
pipeline. Fixed overlapping chunks performed worse because nearby chunks from the same page
occupied several Top-K positions. Following the semantic-chunking idea discussed in the morning
session, we embedded adjacent sentences with multilingual E5 and created a boundary below 0.78
cosine similarity, subject to 250--1,000 character limits. The semantic chunks were stored in a
separate Qdrant collection, so the parallel Level-1/2 work was unaffected. Our final method uses
semantic Top-5 retrieval, at most one missing-evidence query, and exact duplicate removal before
answer generation.

## 4. Measurement

All Level-3 variants used the same PDF, Q7--Q9, embedding model, Ollama chat model and manually
prepared gold evidence. The three questions make these diagnostic results rather than stable
estimates.

| Method | Gold-page recall | BERTScore F1 | Latency |
|---|---:|---:|---:|
| Page baseline | 0.439 | 0.867 | 25.54 s |
| Page + two-hop | 0.439 | 0.865 | 36.71 s |
| Fixed chunks + two-hop | 0.328 | 0.863 | 30.26 s |
| Semantic chunks | 0.567 | 0.862 | 18.73 s |
| **Semantic chunks + two-hop** | **0.633** | **0.865** | **26.56 s** |

Multi-hop was useful only after improving the retrieval units. Semantic chunking produced the
first clear recall gain; its bounded second hop raised recall from 0.567 to 0.633. The selected
method improved recall over the page baseline while keeping BERTScore close (0.865 vs. 0.867)
and latency within approximately one second (26.56 vs. 25.54 seconds).

## 5. What broke

Our initial assumption was that another retrieval step would be the main Level-3 improvement.
Instead, page-level two-hop stopped early, while fixed chunks caused duplicate-page crowding.
These failures redirected the work from adding more hops to improving how the document was
represented. They also explain why a second query cannot reliably recover evidence from a weak
index.

## 6. Limitations and next steps

The retrieval algorithm remains dense Top-K search without hybrid lexical retrieval, reranking,
maximal-marginal-relevance selection or an explicit page-diversity constraint. Query generation
with the local Ollama model is stochastic, and short semantic chunks can omit individual table
values. Page recall and BERTScore therefore do not measure complete factual correctness.

The final Level-3 method was also not re-evaluated after integration with the Level-1 extraction
and Level-2 conversational work. Because the levels ran concurrently and we also had to produce
all nine grounded answers, we prioritised a working isolated Level-3 implementation over a final
cross-level ablation. With another iteration, we would integrate the shared pipeline and compare
semantic two-hop with hybrid retrieval, diversity-aware reranking and table-aware extraction
across repeated runs.

---

**Repository**: `https://github.com/v-hirak/essir2026-aim-hackathon-participants`  
**Provider / models**: `Level 1/2: <team to complete>; Level 3: Ollama, gemma4:latest, intfloat/multilingual-e5-large`  
**Team**: `<names>`
