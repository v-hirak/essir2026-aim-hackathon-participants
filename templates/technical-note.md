# Technical note — Team SHLV

**Two pages.** Evidence-dense beats long. Commit it to your fork as `TECHNICAL_NOTE.md`
and delete this guidance before submitting.

It supports your Friday presentation and lets us check that your `results/` came from a
system you actually built. Graded on: claim-to-evidence, technical accuracy, solution
design, **insight and measurement**, AI-slop (inverse). See
[`../docs/05_evaluation.md`](../docs/05_evaluation.md).

---

## 1. System

*One paragraph and a pipeline sketch. What happens to a question from `POST /query` to the
returned answer? Name the models, the embedding model, the chunking, the retrieval. A reader
should be able to redraw your `app/rag/` from this.*

| Stage | What you did | Changed from the scaffold? |
|---|---|---|
| Extraction | One major issue with the pypdf reader was that it removed all structural information and split the text as per the line. After experimenting with several other pdf readers, we chose pymupdf because it offered the cleanest text. Furthermore, pymupdf also has a functionality of splitting the pdf text into "blocks", i.e., paragraphs already. So we extracted the text as per blocks and stored the page number as part of the block's metadata which were then passed to the chunking function.
| Chunking | For the chunking, we mostly experimented with different chunk size and overlap combinations until a set of random examples reflected a sensible split of the text.
| Embeddings / index | | |
| Retrieval | | |
| Answer + citation | | |

## 2. Level 2 — conversational memory

*How does a follow-up become answerable? Show a real example from the Level-2 questions: the
raw follow-up, and the standalone query your system produced from the history.*

```
q5 as asked:     "Why does that happen?"
rewritten query: "..."
```

## 3. Level 3 — whole-document reasoning

*What did you build beyond single-shot retrieval — multi-hop, agentic retrieval, a second
index, table handling, a graph? What worked, what didn't?*

## 4. Measurement

*The heaviest-weighted section. Numbers you produced on your own system.*

- How do your answers do per level? Where does it break down — Level 3?
- At least one **ablation**: something you changed and its effect. "Adding query rewriting
  fixed 2 of 3 Level-2 questions" with the before/after beats any unmeasured claim.
- Cost: what does a query cost you (tokens, latency), from your `diagnostics`?

## 5. What broke

*One failure you diagnosed — which stage, how you found it, what you did or why you left it.
An honest negative result scores above an unexamined success.*

## 6. Limitations and next steps

*What your system does not do, and what you would build with another day.*

---

**Repository**: `<your fork URL>`
**Provider / models**: `<provider, chat model, embedding model>`
**Team**: `<names>`
