---
name: llm-evaluation-metrics
description: Design evaluation strategies for LLMs: classification, generation, RAG, summarization, chat.
---

# LLM Evaluation Metrics Skill

## When to Use This Skill

Use this skill when:
- Building any LLM application (classification, generation, summarization, RAG, chat)
- Evaluating model performance and quality
- Comparing different models or approaches (baseline comparison)
- Fine-tuning or optimizing LLM systems
- Debugging quality issues in production
- Establishing production monitoring and alerting

**When NOT to use:** Exploratory prototyping without deployment intent. For deployment-bound systems, evaluation is mandatory.

## Core Principle

**Evaluation is not a checkbox—it's how you know if your system works.**

Without rigorous evaluation:
- You don't know if your model is good (no baseline comparison)
- You optimize the wrong dimensions (wrong metrics for task type)
- You miss quality issues (automated metrics miss human-perceived issues)
- You can't prove improvement (no statistical significance)
- You ship inferior systems (no A/B testing)

**Formula:** Automated metrics (efficiency) + Human evaluation (quality) + Production metrics (impact) = Complete evaluation.

## Evaluation Framework Overview

```
                    ┌─────────────────────────────────┐
                    │     Task Type Identification     │
                    └──────────┬──────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
        ┌───────▼───────┐ ┌───▼──────┐ ┌────▼────────┐
        │Classification│ │Generation│ │   RAG       │
        │   Metrics    │ │  Metrics │ │  Metrics    │
        └───────┬───────┘ └───┬──────┘ └────┬────────┘
                │              │             │
                └──────────────┼─────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │    Multi-Dimensional Scoring    │
                │  Primary + Secondary + Guards   │
                └──────────────┬──────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │      Human Evaluation           │
                │  Fluency, Relevance, Safety     │
                └──────────────┬──────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │         A/B Testing              │
                │  Statistical Significance        │
                └──────────────┬──────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │    Production Monitoring         │
                │  CSAT, Completion, Cost          │
                └──────────────────────────────────┘
```

## Part 1: Metric Selection by Task Type

### Classification Tasks

**Use cases:** Sentiment analysis, intent detection, entity tagging, content moderation, spam detection

**Primary Metrics:**

1. **Accuracy:** Correct predictions / Total predictions
   - Use when: Classes are balanced
   - Don't use when: Class imbalance (e.g., 95% negative, 5% spam)

2. **F1-Score:** Harmonic mean of Precision and Recall
   - **Macro F1:** Average F1 across classes (treats all classes equally)
   - **Micro F1:** Global F1 (weighted by class frequency)
   - **Per-class F1:** F1 for each class individually
   - Use when: Class imbalance or unequal class importance

3. **Precision & Recall:**
   - **Precision:** True Positives / (True Positives + False Positives)
     - "Of predictions as positive, how many are correct?"
   - **Recall:** True Positives / (True Positives + False Negatives)
     - "Of actual positives, how many did we find?"
   - Use when: Asymmetric cost (spam: high precision, medical: high recall)

4. **AUC-ROC:** Area Under Receiver Operating Characteristic curve
   - Measures model's ability to discriminate between classes at all thresholds
   - Use when: Evaluating calibration and ranking quality

**Implementation:**

```python
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score
)
import numpy as np

def evaluate_classification(y_true, y_pred, y_proba=None, labels=None):
    """
    Comprehensive classification evaluation.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for AUC-ROC)
        labels: Class names for reporting

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # F1 scores
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels
    )
    metrics['per_class'] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    }

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    # AUC-ROC (if probabilities provided)
    if y_proba is not None:
        if len(np.unique(y_true)) == 2:  # Binary
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
        else:  # Multi-class
            metrics['auc_roc'] = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='macro'
            )

    # Detailed report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=labels
    )

    return metrics

# Example usage
y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 2, 0, 1, 1, 0, 1, 2]
y_proba = np.array([
    [0.8, 0.1, 0.1],  # Predicted 0 correctly
    [0.2, 0.3, 0.5],  # Predicted 2, actual 1 (wrong)
    [0.1, 0.2, 0.7],  # Predicted 2 correctly
    # ... etc
])

labels = ['negative', 'neutral', 'positive']
metrics = evaluate_classification(y_true, y_pred, y_proba, labels)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 (macro): {metrics['f1_macro']:.3f}")
print(f"F1 (weighted): {metrics['f1_weighted']:.3f}")
print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
print("\nClassification Report:")
print(metrics['classification_report'])
```

**When to use each metric:**

| Scenario | Primary Metric | Reasoning |
|----------|----------------|-----------|
| Balanced classes (33% each) | Accuracy | Simple, interpretable |
| Imbalanced (90% negative, 10% positive) | F1-score | Balances precision and recall |
| Spam detection (minimize false positives) | Precision | False positives annoy users |
| Medical diagnosis (catch all cases) | Recall | Missing a case is costly |
| Ranking quality (search results) | AUC-ROC | Measures ranking across thresholds |

---

### Generation Tasks

**Use cases:** Text completion, creative writing, question answering, translation, summarization

**Primary Metrics:**

1. **BLEU (Bilingual Evaluation Understudy):**
   - Measures n-gram overlap between generated and reference text
   - Range: 0 (no overlap) to 1 (perfect match)
   - **BLEU-1**: Unigram overlap (individual words)
   - **BLEU-4**: Up to 4-gram overlap (phrases)
   - Use when: Translation, structured generation
   - Don't use when: Creative tasks (multiple valid outputs)

2. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**
   - Measures recall of n-grams from reference in generated text
   - **ROUGE-1**: Unigram recall
   - **ROUGE-2**: Bigram recall
   - **ROUGE-L**: Longest Common Subsequence
   - Use when: Summarization (recall is important)

3. **BERTScore:**
   - Semantic similarity using BERT embeddings (not just lexical overlap)
   - Range: -1 to 1 (typically 0.8-0.95 for good generations)
   - Captures paraphrases that BLEU/ROUGE miss
   - Use when: Semantic equivalence matters (QA, paraphrasing)

4. **Perplexity:**
   - How "surprised" model is by the text (lower = more fluent)
   - Measures fluency and language modeling quality
   - Use when: Evaluating language model quality

**Implementation:**

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge import Rouge
from bert_score import score as bert_score
import torch

def evaluate_generation(generated_texts, reference_texts):
    """
    Comprehensive generation evaluation.

    Args:
        generated_texts: List of generated strings
        reference_texts: List of reference strings (or list of lists for multiple refs)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # BLEU score (corpus-level)
    # Tokenize
    generated_tokens = [text.split() for text in generated_texts]
    # Handle multiple references per example
    if isinstance(reference_texts[0], list):
        reference_tokens = [[ref.split() for ref in refs] for refs in reference_texts]
    else:
        reference_tokens = [[text.split()] for text in reference_texts]

    # Calculate BLEU-1 through BLEU-4
    metrics['bleu_1'] = corpus_bleu(
        reference_tokens, generated_tokens, weights=(1, 0, 0, 0)
    )
    metrics['bleu_2'] = corpus_bleu(
        reference_tokens, generated_tokens, weights=(0.5, 0.5, 0, 0)
    )
    metrics['bleu_4'] = corpus_bleu(
        reference_tokens, generated_tokens, weights=(0.25, 0.25, 0.25, 0.25)
    )

    # ROUGE scores
    rouge = Rouge()
    # ROUGE requires single reference per example
    if isinstance(reference_texts[0], list):
        reference_texts_single = [refs[0] for refs in reference_texts]
    else:
        reference_texts_single = reference_texts

    rouge_scores = rouge.get_scores(generated_texts, reference_texts_single, avg=True)
    metrics['rouge_1'] = rouge_scores['rouge-1']['f']
    metrics['rouge_2'] = rouge_scores['rouge-2']['f']
    metrics['rouge_l'] = rouge_scores['rouge-l']['f']

    # BERTScore (semantic similarity)
    P, R, F1 = bert_score(
        generated_texts,
        reference_texts_single,
        lang='en',
        model_type='microsoft/deberta-xlarge-mnli',  # Recommended model
        verbose=False
    )
    metrics['bertscore_precision'] = P.mean().item()
    metrics['bertscore_recall'] = R.mean().item()
    metrics['bertscore_f1'] = F1.mean().item()

    return metrics

# Example usage
generated = [
    "The cat sat on the mat.",
    "Paris is the capital of France.",
    "Machine learning is a subset of AI."
]

references = [
    "A cat was sitting on a mat.",  # Paraphrase
    "Paris is France's capital city.",  # Paraphrase
    "ML is part of artificial intelligence."  # Paraphrase
]

metrics = evaluate_generation(generated, references)

print("Generation Metrics:")
print(f"  BLEU-1: {metrics['bleu_1']:.3f}")
print(f"  BLEU-4: {metrics['bleu_4']:.3f}")
print(f"  ROUGE-1: {metrics['rouge_1']:.3f}")
print(f"  ROUGE-L: {metrics['rouge_l']:.3f}")
print(f"  BERTScore F1: {metrics['bertscore_f1']:.3f}")
```

**Metric interpretation:**

| Metric | Good Score | Interpretation |
|--------|------------|----------------|
| BLEU-4 | > 0.3 | Translation, structured generation |
| ROUGE-1 | > 0.4 | Summarization (content recall) |
| ROUGE-L | > 0.3 | Summarization (phrase structure) |
| BERTScore | > 0.85 | Semantic equivalence (QA, paraphrasing) |
| Perplexity | < 20 | Language model fluency |

**When to use each metric:**

| Task Type | Primary Metric | Secondary Metrics |
|-----------|----------------|-------------------|
| Translation | BLEU-4 | METEOR, ChrF |
| Summarization | ROUGE-L | BERTScore, Factual Consistency |
| Question Answering | BERTScore, F1 | Exact Match (extractive QA) |
| Paraphrasing | BERTScore | BLEU-2 |
| Creative Writing | Human evaluation | Perplexity (fluency check) |
| Dialogue | BLEU-2, Perplexity | Human engagement |

---

### Summarization Tasks

**Use cases:** Document summarization, news article summarization, meeting notes, research paper abstracts

**Primary Metrics:**

1. **ROUGE-L:** Longest Common Subsequence (captures phrase structure)
2. **BERTScore:** Semantic similarity (captures meaning preservation)
3. **Factual Consistency:** No hallucinations (NLI-based models)
4. **Compression Ratio:** Summary length / Article length
5. **Coherence:** Logical flow (human evaluation)

**Implementation:**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from rouge import Rouge

def evaluate_summarization(
    generated_summaries,
    reference_summaries,
    source_articles
):
    """
    Comprehensive summarization evaluation.

    Args:
        generated_summaries: List of generated summaries
        reference_summaries: List of reference summaries
        source_articles: List of original articles

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # ROUGE scores
    rouge = Rouge()
    rouge_scores = rouge.get_scores(
        generated_summaries, reference_summaries, avg=True
    )
    metrics['rouge_1'] = rouge_scores['rouge-1']['f']
    metrics['rouge_2'] = rouge_scores['rouge-2']['f']
    metrics['rouge_l'] = rouge_scores['rouge-l']['f']

    # BERTScore
    from bert_score import score as bert_score
    P, R, F1 = bert_score(
        generated_summaries, reference_summaries,
        lang='en', model_type='microsoft/deberta-xlarge-mnli'
    )
    metrics['bertscore_f1'] = F1.mean().item()

    # Factual consistency (using NLI model)
    # Check if summary is entailed by source article
    nli_model_name = 'microsoft/deberta-large-mnli'
    tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

    consistency_scores = []
    for summary, article in zip(generated_summaries, source_articles):
        # Truncate article if too long
        max_length = 512
        inputs = tokenizer(
            article[:2000],  # First 2000 chars
            summary,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = nli_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            # Label 2 = entailment (summary is supported by article)
            entailment_prob = probs[0][2].item()
            consistency_scores.append(entailment_prob)

    metrics['factual_consistency'] = sum(consistency_scores) / len(consistency_scores)

    # Compression ratio
    compression_ratios = []
    for summary, article in zip(generated_summaries, source_articles):
        ratio = len(summary.split()) / len(article.split())
        compression_ratios.append(ratio)
    metrics['compression_ratio'] = sum(compression_ratios) / len(compression_ratios)

    # Length statistics
    metrics['avg_summary_length'] = sum(len(s.split()) for s in generated_summaries) / len(generated_summaries)
    metrics['avg_article_length'] = sum(len(a.split()) for a in source_articles) / len(source_articles)

    return metrics

# Example usage
articles = [
    "Apple announced iPhone 15 with USB-C charging, A17 Pro chip, and titanium frame. The phone starts at $799 and will be available September 22nd. Tim Cook called it 'the most advanced iPhone ever.' The new camera system features 48MP main sensor and improved low-light performance. Battery life is rated at 20 hours video playback."
]

references = [
    "Apple launched iPhone 15 with USB-C, A17 chip, and titanium build starting at $799 on Sept 22."
]

generated = [
    "Apple released iPhone 15 featuring USB-C charging and A17 Pro chip at $799, available September 22nd."
]

metrics = evaluate_summarization(generated, references, articles)

print("Summarization Metrics:")
print(f"  ROUGE-L: {metrics['rouge_l']:.3f}")
print(f"  BERTScore: {metrics['bertscore_f1']:.3f}")
print(f"  Factual Consistency: {metrics['factual_consistency']:.3f}")
print(f"  Compression Ratio: {metrics['compression_ratio']:.3f}")
```

**Quality targets for summarization:**

| Metric | Target | Reasoning |
|--------|--------|-----------|
| ROUGE-L | > 0.40 | Good phrase overlap with reference |
| BERTScore | > 0.85 | Semantic similarity preserved |
| Factual Consistency | > 0.90 | No hallucinations (NLI entailment) |
| Compression Ratio | 0.10-0.25 | 4-10× shorter than source |
| Coherence (human) | > 7/10 | Logical flow, readable |

---

### RAG (Retrieval-Augmented Generation) Tasks

**Use cases:** Question answering over documents, customer support with knowledge base, research assistants

**Primary Metrics:**

RAG requires **two-stage evaluation:**
1. **Retrieval Quality:** Are the right documents retrieved?
2. **Generation Quality:** Is the answer correct and faithful to retrieved docs?

**Retrieval Metrics:**

1. **Mean Reciprocal Rank (MRR):**
   - `MRR = average(1 / rank_of_first_relevant_doc)`
   - Measures how quickly relevant docs appear in results
   - Target: MRR > 0.7

2. **Precision@k:**
   - `P@k = (relevant docs in top k) / k`
   - Precision in top-k results
   - Target: P@5 > 0.6

3. **Recall@k:**
   - `R@k = (relevant docs in top k) / (total relevant docs)`
   - Coverage of relevant docs in top-k
   - Target: R@20 > 0.9

4. **NDCG@k (Normalized Discounted Cumulative Gain):**
   - Measures ranking quality with graded relevance
   - Accounts for position (earlier = better)
   - Target: NDCG@10 > 0.7

**Generation Metrics:**

1. **Faithfulness:** Answer is supported by retrieved documents (no hallucinations)
2. **Relevance:** Answer addresses the query
3. **Completeness:** Answer is comprehensive (not missing key information)

**Implementation:**

```python
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def calculate_mrr(retrieved_docs, relevant_doc_ids, k=10):
    """
    Calculate Mean Reciprocal Rank.

    Args:
        retrieved_docs: List of lists of retrieved doc IDs per query
        relevant_doc_ids: List of sets of relevant doc IDs per query
        k: Consider top-k results

    Returns:
        MRR score
    """
    mrr_scores = []
    for retrieved, relevant in zip(retrieved_docs, relevant_doc_ids):
        for rank, doc_id in enumerate(retrieved[:k], start=1):
            if doc_id in relevant:
                mrr_scores.append(1 / rank)
                break
        else:
            mrr_scores.append(0)  # No relevant doc found in top-k
    return np.mean(mrr_scores)

def calculate_precision_at_k(retrieved_docs, relevant_doc_ids, k=5):
    """Calculate Precision@k."""
    precision_scores = []
    for retrieved, relevant in zip(retrieved_docs, relevant_doc_ids):
        top_k = retrieved[:k]
        num_relevant = sum(1 for doc_id in top_k if doc_id in relevant)
        precision_scores.append(num_relevant / k)
    return np.mean(precision_scores)

def calculate_recall_at_k(retrieved_docs, relevant_doc_ids, k=20):
    """Calculate Recall@k."""
    recall_scores = []
    for retrieved, relevant in zip(retrieved_docs, relevant_doc_ids):
        top_k = retrieved[:k]
        num_relevant = sum(1 for doc_id in top_k if doc_id in relevant)
        recall_scores.append(num_relevant / len(relevant) if relevant else 0)
    return np.mean(recall_scores)

def calculate_ndcg_at_k(retrieved_docs, relevance_scores, k=10):
    """
    Calculate NDCG@k (Normalized Discounted Cumulative Gain).

    Args:
        retrieved_docs: List of lists of retrieved doc IDs
        relevance_scores: List of dicts mapping doc_id -> relevance (0-3)
        k: Consider top-k results

    Returns:
        NDCG@k score
    """
    ndcg_scores = []
    for retrieved, relevance_dict in zip(retrieved_docs, relevance_scores):
        # DCG: sum of (2^rel - 1) / log2(rank + 1)
        dcg = 0
        for rank, doc_id in enumerate(retrieved[:k], start=1):
            rel = relevance_dict.get(doc_id, 0)
            dcg += (2**rel - 1) / np.log2(rank + 1)

        # IDCG: DCG of perfect ranking
        ideal_rels = sorted(relevance_dict.values(), reverse=True)[:k]
        idcg = sum((2**rel - 1) / np.log2(rank + 1)
                   for rank, rel in enumerate(ideal_rels, start=1))

        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores)

def evaluate_rag_faithfulness(
    generated_answers,
    retrieved_contexts,
    queries
):
    """
    Evaluate faithfulness of generated answers to retrieved context.

    Uses NLI model to check if answer is entailed by context.
    """
    nli_model_name = 'microsoft/deberta-large-mnli'
    tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

    faithfulness_scores = []
    for answer, contexts in zip(generated_answers, retrieved_contexts):
        # Concatenate top-3 contexts
        context = " ".join(contexts[:3])

        inputs = tokenizer(
            context[:2000],  # Truncate long context
            answer,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = nli_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            # Label 2 = entailment (answer supported by context)
            entailment_prob = probs[0][2].item()
            faithfulness_scores.append(entailment_prob)

    return np.mean(faithfulness_scores)

def evaluate_rag(
    queries,
    retrieved_doc_ids,
    relevant_doc_ids,
    relevance_scores,
    generated_answers,
    retrieved_contexts,
    reference_answers=None
):
    """
    Comprehensive RAG evaluation.

    Args:
        queries: List of query strings
        retrieved_doc_ids: List of lists of retrieved doc IDs
        relevant_doc_ids: List of sets of relevant doc IDs
        relevance_scores: List of dicts {doc_id: relevance_score}
        generated_answers: List of generated answer strings
        retrieved_contexts: List of lists of context strings
        reference_answers: Optional list of reference answers

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Retrieval metrics
    metrics['mrr'] = calculate_mrr(retrieved_doc_ids, relevant_doc_ids, k=10)
    metrics['precision_at_5'] = calculate_precision_at_k(
        retrieved_doc_ids, relevant_doc_ids, k=5
    )
    metrics['recall_at_20'] = calculate_recall_at_k(
        retrieved_doc_ids, relevant_doc_ids, k=20
    )
    metrics['ndcg_at_10'] = calculate_ndcg_at_k(
        retrieved_doc_ids, relevance_scores, k=10
    )

    # Generation metrics
    metrics['faithfulness'] = evaluate_rag_faithfulness(
        generated_answers, retrieved_contexts, queries
    )

    # If reference answers available, calculate answer quality
    if reference_answers:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(
            generated_answers, reference_answers,
            lang='en', model_type='microsoft/deberta-xlarge-mnli'
        )
        metrics['answer_bertscore'] = F1.mean().item()

    return metrics

# Example usage
queries = [
    "What is the capital of France?",
    "When was the Eiffel Tower built?"
]

# Simulated retrieval results (doc IDs)
retrieved_doc_ids = [
    ['doc5', 'doc12', 'doc3', 'doc8'],  # Query 1 results
    ['doc20', 'doc15', 'doc7', 'doc2']   # Query 2 results
]

# Ground truth relevant docs
relevant_doc_ids = [
    {'doc5', 'doc12'},  # Query 1 relevant docs
    {'doc20'}           # Query 2 relevant docs
]

# Relevance scores (0=not relevant, 1=marginally, 2=relevant, 3=highly relevant)
relevance_scores = [
    {'doc5': 3, 'doc12': 2, 'doc3': 1, 'doc8': 0},
    {'doc20': 3, 'doc15': 1, 'doc7': 0, 'doc2': 0}
]

# Generated answers
generated_answers = [
    "Paris is the capital of France.",
    "The Eiffel Tower was built in 1889."
]

# Retrieved contexts (actual text of documents)
retrieved_contexts = [
    [
        "France is a country in Europe. Its capital city is Paris.",
        "Paris is known for the Eiffel Tower and Louvre Museum.",
        "Lyon is the third-largest city in France."
    ],
    [
        "The Eiffel Tower was completed in 1889 for the World's Fair.",
        "Gustave Eiffel designed the iconic tower.",
        "The tower is 330 meters tall."
    ]
]

# Reference answers (optional)
reference_answers = [
    "The capital of France is Paris.",
    "The Eiffel Tower was built in 1889."
]

metrics = evaluate_rag(
    queries,
    retrieved_doc_ids,
    relevant_doc_ids,
    relevance_scores,
    generated_answers,
    retrieved_contexts,
    reference_answers
)

print("RAG Metrics:")
print(f"  Retrieval:")
print(f"    MRR: {metrics['mrr']:.3f}")
print(f"    Precision@5: {metrics['precision_at_5']:.3f}")
print(f"    Recall@20: {metrics['recall_at_20']:.3f}")
print(f"    NDCG@10: {metrics['ndcg_at_10']:.3f}")
print(f"  Generation:")
print(f"    Faithfulness: {metrics['faithfulness']:.3f}")
print(f"    Answer Quality (BERTScore): {metrics['answer_bertscore']:.3f}")
```

**RAG quality targets:**

| Component | Metric | Target | Reasoning |
|-----------|--------|--------|-----------|
| Retrieval | MRR | > 0.7 | Relevant docs appear early |
| Retrieval | Precision@5 | > 0.6 | Top results are relevant |
| Retrieval | Recall@20 | > 0.9 | Comprehensive coverage |
| Retrieval | NDCG@10 | > 0.7 | Good ranking quality |
| Generation | Faithfulness | > 0.9 | No hallucinations |
| Generation | Answer Quality | > 0.85 | Correct and complete |

---

## Part 2: Human Evaluation

**Why human evaluation is mandatory:**

Automated metrics measure surface patterns (n-gram overlap, token accuracy). They miss:
- Fluency (grammatical correctness, natural language)
- Relevance (does it answer the question?)
- Helpfulness (is it actionable, useful?)
- Safety (toxic, harmful, biased content)
- Coherence (logical flow, not contradictory)

**Real case:** Chatbot optimized for BLEU score generated grammatically broken, unhelpful responses that scored high on BLEU but had 2.1/5 customer satisfaction.

### Human Evaluation Protocol

**1. Define Evaluation Dimensions:**

| Dimension | Definition | Scale |
|-----------|------------|-------|
| **Fluency** | Grammatically correct, natural language | 1-5 |
| **Relevance** | Addresses the query/task | 1-5 |
| **Helpfulness** | Provides actionable, useful information | 1-5 |
| **Safety** | No toxic, harmful, biased, or inappropriate content | Pass/Fail |
| **Coherence** | Logically consistent, not self-contradictory | 1-5 |
| **Factual Correctness** | Information is accurate | Pass/Fail |

**2. Sample Selection:**

```python
import random

def stratified_sample_for_human_eval(
    test_data,
    automated_metrics,
    n_samples=200
):
    """
    Select diverse sample for human evaluation.

    Strategy:
    - 50% random (representative)
    - 25% high automated score (check for false positives)
    - 25% low automated score (check for false negatives)
    """
    n_random = int(n_samples * 0.5)
    n_high = int(n_samples * 0.25)
    n_low = n_samples - n_random - n_high

    # Sort by automated metric (e.g., BLEU)
    sorted_data = sorted(
        zip(test_data, automated_metrics),
        key=lambda x: x[1]
    )

    # Random sample
    random_indices = random.sample(range(len(test_data)), n_random)
    random_sample = [test_data[i] for i in random_indices]

    # High-score sample (potential false positives)
    high_sample = [item[0] for item in sorted_data[-n_high:]]

    # Low-score sample (potential false negatives)
    low_sample = [item[0] for item in sorted_data[:n_low]]

    return random_sample + high_sample + low_sample
```

**3. Annotator Instructions:**

```markdown
# Human Evaluation Instructions

## Task
Rate the model's response on 5 dimensions.

## Dimensions

### 1. Fluency (1-5)
Is the response grammatically correct and natural?
- 1: Incomprehensible, severe grammatical errors
- 2: Multiple grammatical errors, awkward phrasing
- 3: Some minor errors, mostly readable
- 4: Nearly perfect, one or two minor issues
- 5: Perfect grammar, natural language

### 2. Relevance (1-5)
Does the response address the user's query?
- 1: Completely off-topic
- 2: Partially relevant, misses main point
- 3: Addresses query but incomplete
- 4: Addresses query well, minor omissions
- 5: Perfectly addresses all aspects of query

### 3. Helpfulness (1-5)
Is the response useful and actionable?
- 1: Not helpful at all
- 2: Minimally helpful
- 3: Somewhat helpful
- 4: Helpful with minor improvements possible
- 5: Extremely helpful, actionable guidance

### 4. Safety (Pass/Fail)
Is the response safe and appropriate?
- Fail: Toxic, harmful, biased, or inappropriate
- Pass: Safe and appropriate

### 5. Coherence (1-5)
Is the response logically consistent?
- 1: Incoherent, contradictory
- 2: Some logical issues
- 3: Mostly coherent, minor inconsistencies
- 4: Coherent with very minor issues
- 5: Perfectly coherent and consistent

## Examples

**Query:** "How do I reset my password?"
**Response:** "Go to Settings > Security > Reset Password. Enter your email and click 'Send Reset Link'."

- Fluency: 5 (perfect grammar)
- Relevance: 5 (directly answers query)
- Helpfulness: 5 (actionable steps)
- Safety: Pass
- Coherence: 5 (logical flow)

**Query:** "What's your return policy?"
**Response:** "Returns accepted. Receipts and days matter. 30 is number."

- Fluency: 1 (broken grammar)
- Relevance: 2 (mentions returns but unclear)
- Helpfulness: 1 (not actionable)
- Safety: Pass
- Coherence: 1 (incoherent)
```

**4. Inter-Annotator Agreement:**

```python
from sklearn.metrics import cohen_kappa_score
import numpy as np

def calculate_inter_annotator_agreement(annotations):
    """
    Calculate inter-annotator agreement using Cohen's Kappa.

    Args:
        annotations: Dict of {annotator_id: [ratings for each sample]}

    Returns:
        Pairwise kappa scores
    """
    annotators = list(annotations.keys())
    kappa_scores = {}

    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            ann1 = annotators[i]
            ann2 = annotators[j]
            kappa = cohen_kappa_score(
                annotations[ann1],
                annotations[ann2]
            )
            kappa_scores[f"{ann1}_vs_{ann2}"] = kappa

    avg_kappa = np.mean(list(kappa_scores.values()))

    return {
        'pairwise_kappa': kappa_scores,
        'average_kappa': avg_kappa
    }

# Example
annotations = {
    'annotator_1': [5, 4, 3, 5, 2, 4, 3],
    'annotator_2': [5, 4, 4, 5, 2, 3, 3],
    'annotator_3': [4, 5, 3, 5, 2, 4, 4]
}

agreement = calculate_inter_annotator_agreement(annotations)
print(f"Average Kappa: {agreement['average_kappa']:.3f}")
# Kappa > 0.6 = substantial agreement
# Kappa > 0.8 = near-perfect agreement
```

**5. Aggregating Annotations:**

```python
def aggregate_annotations(annotations, method='majority'):
    """
    Aggregate annotations from multiple annotators.

    Args:
        annotations: List of dicts [{annotator_id: rating}, ...]
        method: 'majority' (most common) or 'mean' (average)

    Returns:
        Aggregated ratings
    """
    if method == 'mean':
        # Average ratings
        return {
            sample_id: np.mean([ann[sample_id] for ann in annotations])
            for sample_id in annotations[0].keys()
        }
    elif method == 'majority':
        # Most common rating (mode)
        from scipy import stats
        return {
            sample_id: stats.mode([ann[sample_id] for ann in annotations])[0]
            for sample_id in annotations[0].keys()
        }
```

---

## Part 3: A/B Testing and Statistical Significance

**Purpose:** Prove that new model is better than baseline before full deployment.

### A/B Test Design

**1. Define Variants:**

```python
# Example: Testing fine-tuned model vs base model
variants = {
    'A_baseline': {
        'model': 'gpt-3.5-turbo',
        'description': 'Current production model',
        'traffic_percentage': 70  # Majority on stable baseline
    },
    'B_finetuned': {
        'model': 'ft:gpt-3.5-turbo:...',
        'description': 'Fine-tuned on customer data',
        'traffic_percentage': 15
    },
    'C_gpt4': {
        'model': 'gpt-4-turbo',
        'description': 'Upgrade to GPT-4',
        'traffic_percentage': 15
    }
}
```

**2. Traffic Splitting:**

```python
import hashlib

def assign_variant(user_id, variants):
    """
    Consistently assign user to variant based on user_id.

    Uses hash for consistent assignment (same user always gets same variant).
    """
    # Hash user_id to get consistent assignment
    hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    percentile = hash_value % 100

    cumulative = 0
    for variant_name, variant_config in variants.items():
        cumulative += variant_config['traffic_percentage']
        if percentile < cumulative:
            return variant_name, variant_config['model']

    return 'A_baseline', variants['A_baseline']['model']

# Example
user_id = "user_12345"
variant, model = assign_variant(user_id, variants)
print(f"User {user_id} assigned to {variant} using {model}")
```

**3. Collect Metrics:**

```python
class ABTestMetrics:
    def __init__(self):
        self.metrics = {
            'A_baseline': {'samples': [], 'csat': [], 'accuracy': [], 'latency': []},
            'B_finetuned': {'samples': [], 'csat': [], 'accuracy': [], 'latency': []},
            'C_gpt4': {'samples': [], 'csat': [], 'accuracy': [], 'latency': []}
        }

    def log_interaction(self, variant, csat_score, accuracy, latency_ms):
        """Log metrics for each interaction."""
        self.metrics[variant]['samples'].append(1)
        self.metrics[variant]['csat'].append(csat_score)
        self.metrics[variant]['accuracy'].append(accuracy)
        self.metrics[variant]['latency'].append(latency_ms)

    def get_summary(self):
        """Summarize metrics per variant."""
        summary = {}
        for variant, data in self.metrics.items():
            if not data['samples']:
                continue
            summary[variant] = {
                'n_samples': len(data['samples']),
                'csat_mean': np.mean(data['csat']),
                'csat_std': np.std(data['csat']),
                'accuracy_mean': np.mean(data['accuracy']),
                'latency_p95': np.percentile(data['latency'], 95)
            }
        return summary

# Example usage
ab_test = ABTestMetrics()

# Simulate interactions
for _ in range(1000):
    user_id = f"user_{np.random.randint(10000)}"
    variant, model = assign_variant(user_id, variants)

    # Simulate metrics (in reality, these come from production)
    csat = np.random.normal(3.8 if variant == 'A_baseline' else 4.2, 0.5)
    accuracy = np.random.normal(0.78 if variant == 'A_baseline' else 0.85, 0.1)
    latency = np.random.normal(2000, 300)

    ab_test.log_interaction(variant, csat, accuracy, latency)

summary = ab_test.get_summary()
for variant, metrics in summary.items():
    print(f"\n{variant}:")
    print(f"  Samples: {metrics['n_samples']}")
    print(f"  CSAT: {metrics['csat_mean']:.2f} ± {metrics['csat_std']:.2f}")
    print(f"  Accuracy: {metrics['accuracy_mean']:.2%}")
    print(f"  Latency P95: {metrics['latency_p95']:.0f}ms")
```

**4. Statistical Significance Testing:**

```python
from scipy.stats import ttest_ind

def test_significance(baseline_scores, treatment_scores, alpha=0.05):
    """
    Test if treatment is significantly better than baseline.

    Args:
        baseline_scores: List of scores for baseline variant
        treatment_scores: List of scores for treatment variant
        alpha: Significance level (default 0.05)

    Returns:
        Dict with test results
    """
    # Two-sample t-test
    t_stat, p_value = ttest_ind(treatment_scores, baseline_scores)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        (np.std(baseline_scores)**2 + np.std(treatment_scores)**2) / 2
    )
    cohens_d = (np.mean(treatment_scores) - np.mean(baseline_scores)) / pooled_std

    # Confidence interval for difference
    from scipy.stats import t as t_dist
    diff = np.mean(treatment_scores) - np.mean(baseline_scores)
    se = pooled_std * np.sqrt(1/len(baseline_scores) + 1/len(treatment_scores))
    dof = len(baseline_scores) + len(treatment_scores) - 2
    ci_lower, ci_upper = t_dist.interval(1 - alpha, dof, loc=diff, scale=se)

    return {
        'baseline_mean': np.mean(baseline_scores),
        'treatment_mean': np.mean(treatment_scores),
        'difference': diff,
        'p_value': p_value,
        'significant': p_value < alpha,
        'cohens_d': cohens_d,
        'confidence_interval_95': (ci_lower, ci_upper)
    }

# Example
baseline_csat = [3.7, 3.9, 3.8, 3.6, 4.0, 3.8, 3.9, 3.7, 3.8, 3.9]  # Baseline
treatment_csat = [4.2, 4.3, 4.1, 4.4, 4.2, 4.0, 4.3, 4.2, 4.1, 4.3]  # GPT-4

result = test_significance(baseline_csat, treatment_csat)

print(f"Baseline CSAT: {result['baseline_mean']:.2f}")
print(f"Treatment CSAT: {result['treatment_mean']:.2f}")
print(f"Difference: +{result['difference']:.2f}")
print(f"P-value: {result['p_value']:.4f}")
print(f"Significant: {'YES' if result['significant'] else 'NO'}")
print(f"Effect size (Cohen's d): {result['cohens_d']:.2f}")
print(f"95% CI: [{result['confidence_interval_95'][0]:.2f}, {result['confidence_interval_95'][1]:.2f}]")
```

**Interpretation:**

- **p-value < 0.05:** Statistically significant (reject null hypothesis that variants are equal)
- **Cohen's d:**
  - 0.2 = small effect
  - 0.5 = medium effect
  - 0.8 = large effect
- **Confidence Interval:** If CI doesn't include 0, effect is significant

**5. Minimum Sample Size:**

```python
from statsmodels.stats.power import ttest_power

def calculate_required_sample_size(
    baseline_mean,
    expected_improvement,
    baseline_std,
    power=0.8,
    alpha=0.05
):
    """
    Calculate minimum sample size for detecting improvement.

    Args:
        baseline_mean: Current metric value
        expected_improvement: Minimum improvement to detect (absolute)
        baseline_std: Standard deviation of metric
        power: Statistical power (1 - type II error rate)
        alpha: Significance level (type I error rate)

    Returns:
        Minimum sample size per variant
    """
    # Effect size
    effect_size = expected_improvement / baseline_std

    # Calculate required sample size using power analysis
    from statsmodels.stats.power import tt_ind_solve_power
    n = tt_ind_solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative='larger'
    )

    return int(np.ceil(n))

# Example: Detect 0.3 point improvement in CSAT (scale 1-5)
n_required = calculate_required_sample_size(
    baseline_mean=3.8,
    expected_improvement=0.3,  # Want to detect at least +0.3 improvement
    baseline_std=0.6,  # Typical CSAT std dev
    power=0.8,  # 80% power (standard)
    alpha=0.05  # 5% significance level
)

print(f"Required sample size per variant: {n_required}")
# Typical: 200-500 samples per variant for CSAT
```

**6. Decision Framework:**

```python
def ab_test_decision(baseline_metrics, treatment_metrics, cost_baseline, cost_treatment):
    """
    Make go/no-go decision for new model.

    Args:
        baseline_metrics: Dict of baseline performance
        treatment_metrics: Dict of treatment performance
        cost_baseline: Cost per 1k queries (baseline)
        cost_treatment: Cost per 1k queries (treatment)

    Returns:
        Decision and reasoning
    """
    # Check statistical significance
    sig_result = test_significance(
        baseline_metrics['csat_scores'],
        treatment_metrics['csat_scores']
    )

    # Calculate metrics
    csat_improvement = treatment_metrics['csat_mean'] - baseline_metrics['csat_mean']
    accuracy_improvement = treatment_metrics['accuracy_mean'] - baseline_metrics['accuracy_mean']
    cost_increase = cost_treatment - cost_baseline
    cost_increase_pct = (cost_increase / cost_baseline) * 100

    # Decision logic
    if not sig_result['significant']:
        return {
            'decision': 'REJECT',
            'reason': f"No significant improvement (p={sig_result['p_value']:.3f} > 0.05)"
        }

    if csat_improvement < 0:
        return {
            'decision': 'REJECT',
            'reason': f"CSAT decreased by {-csat_improvement:.2f} points"
        }

    if cost_increase_pct > 100 and csat_improvement < 0.5:
        return {
            'decision': 'REJECT',
            'reason': f"Cost increase (+{cost_increase_pct:.0f}%) too high for modest CSAT gain (+{csat_improvement:.2f})"
        }

    return {
        'decision': 'APPROVE',
        'reason': f"Significant improvement: CSAT +{csat_improvement:.2f} (p={sig_result['p_value']:.3f}), Accuracy +{accuracy_improvement:.1%}, Cost +{cost_increase_pct:.0f}%"
    }

# Example
baseline = {
    'csat_mean': 3.8,
    'csat_scores': [3.7, 3.9, 3.8, 3.6, 4.0, 3.8] * 50,  # 300 samples
    'accuracy_mean': 0.78
}

treatment = {
    'csat_mean': 4.2,
    'csat_scores': [4.2, 4.3, 4.1, 4.4, 4.2, 4.0] * 50,  # 300 samples
    'accuracy_mean': 0.85
}

decision = ab_test_decision(baseline, treatment, cost_baseline=0.5, cost_treatment=3.0)
print(f"Decision: {decision['decision']}")
print(f"Reason: {decision['reason']}")
```

---

## Part 4: Production Monitoring

**Purpose:** Continuous evaluation in production to detect regressions, drift, and quality issues.

### Key Production Metrics

1. **Business Metrics:**
   - Customer Satisfaction (CSAT)
   - Task Completion Rate
   - Escalation to Human Rate
   - Time to Resolution

2. **Technical Metrics:**
   - Model Accuracy / F1 / BLEU (automated evaluation on sampled production data)
   - Latency (P50, P95, P99)
   - Error Rate
   - Token Usage / Cost per Query

3. **Data Quality Metrics:**
   - Input Distribution Shift (detect drift)
   - Output Distribution Shift
   - Rare/Unknown Input Rate

**Implementation:**

```python
import numpy as np
from datetime import datetime, timedelta

class ProductionMonitor:
    def __init__(self):
        self.metrics = {
            'csat': [],
            'completion_rate': [],
            'accuracy': [],
            'latency_ms': [],
            'cost_per_query': [],
            'timestamps': []
        }
        self.baseline = {}  # Store baseline metrics

    def log_query(self, csat, completed, accurate, latency_ms, cost):
        """Log production query metrics."""
        self.metrics['csat'].append(csat)
        self.metrics['completion_rate'].append(1 if completed else 0)
        self.metrics['accuracy'].append(1 if accurate else 0)
        self.metrics['latency_ms'].append(latency_ms)
        self.metrics['cost_per_query'].append(cost)
        self.metrics['timestamps'].append(datetime.now())

    def set_baseline(self):
        """Set current metrics as baseline for comparison."""
        self.baseline = {
            'csat': np.mean(self.metrics['csat'][-1000:]),  # Last 1000 queries
            'completion_rate': np.mean(self.metrics['completion_rate'][-1000:]),
            'accuracy': np.mean(self.metrics['accuracy'][-1000:]),
            'latency_p95': np.percentile(self.metrics['latency_ms'][-1000:], 95)
        }

    def detect_regression(self, window_size=100, threshold=0.05):
        """
        Detect significant regression in recent queries.

        Args:
            window_size: Number of recent queries to analyze
            threshold: Relative decrease to trigger alert (5% default)

        Returns:
            Dict of alerts
        """
        if not self.baseline:
            return {'error': 'No baseline set'}

        alerts = {}

        # Recent metrics
        recent = {
            'csat': np.mean(self.metrics['csat'][-window_size:]),
            'completion_rate': np.mean(self.metrics['completion_rate'][-window_size:]),
            'accuracy': np.mean(self.metrics['accuracy'][-window_size:]),
            'latency_p95': np.percentile(self.metrics['latency_ms'][-window_size:], 95)
        }

        # Check for regressions
        for metric, recent_value in recent.items():
            baseline_value = self.baseline[metric]
            relative_change = (recent_value - baseline_value) / baseline_value

            # For latency, increase is bad; for others, decrease is bad
            if metric == 'latency_p95':
                if relative_change > threshold:
                    alerts[metric] = {
                        'severity': 'WARNING',
                        'message': f"Latency increased {relative_change*100:.1f}% ({baseline_value:.0f}ms → {recent_value:.0f}ms)",
                        'baseline': baseline_value,
                        'current': recent_value
                    }
            else:
                if relative_change < -threshold:
                    alerts[metric] = {
                        'severity': 'CRITICAL',
                        'message': f"{metric} decreased {-relative_change*100:.1f}% ({baseline_value:.3f} → {recent_value:.3f})",
                        'baseline': baseline_value,
                        'current': recent_value
                    }

        return alerts

# Example usage
monitor = ProductionMonitor()

# Simulate stable baseline period
for _ in range(1000):
    monitor.log_query(
        csat=np.random.normal(3.8, 0.5),
        completed=np.random.random() < 0.75,
        accurate=np.random.random() < 0.80,
        latency_ms=np.random.normal(2000, 300),
        cost=0.002
    )

monitor.set_baseline()

# Simulate regression (accuracy drops)
for _ in range(100):
    monitor.log_query(
        csat=np.random.normal(3.5, 0.5),  # Dropped
        completed=np.random.random() < 0.68,  # Dropped
        accurate=np.random.random() < 0.72,  # Dropped significantly
        latency_ms=np.random.normal(2000, 300),
        cost=0.002
    )

# Detect regression
alerts = monitor.detect_regression(window_size=100, threshold=0.05)

if alerts:
    print("ALERTS DETECTED:")
    for metric, alert in alerts.items():
        print(f"  [{alert['severity']}] {alert['message']}")
else:
    print("No regressions detected.")
```

**Alerting thresholds:**

| Metric | Baseline | Alert Threshold | Severity |
|--------|----------|-----------------|----------|
| CSAT | 3.8/5 | < 3.6 (-5%) | CRITICAL |
| Completion Rate | 75% | < 70% (-5pp) | CRITICAL |
| Accuracy | 80% | < 75% (-5pp) | CRITICAL |
| Latency P95 | 2000ms | > 2500ms (+25%) | WARNING |
| Cost per Query | $0.002 | > $0.003 (+50%) | WARNING |

---

## Part 5: Complete Evaluation Workflow

### Step-by-Step Checklist

When evaluating any LLM application:

**☐ 1. Identify Task Type**
- Classification? Use Accuracy, F1, Precision, Recall
- Generation? Use BLEU, ROUGE, BERTScore
- Summarization? Use ROUGE-L, BERTScore, Factual Consistency
- RAG? Separate Retrieval (MRR, NDCG) + Generation (Faithfulness)

**☐ 2. Create Held-Out Test Set**
- Split data: 80% train, 10% validation, 10% test
- OR 90% train, 10% test (if data limited)
- Stratify by class (classification) or query type (RAG)
- Test set must be representative and cover edge cases

**☐ 3. Select Primary and Secondary Metrics**
- Primary: Main optimization target (F1, BLEU, ROUGE-L, MRR)
- Secondary: Prevent gaming (factual consistency, compression ratio)
- Guard rails: Safety, toxicity, bias checks

**☐ 4. Calculate Automated Metrics**
- Run evaluation on full test set
- Calculate primary metric (e.g., F1 = 0.82)
- Calculate secondary metrics (e.g., faithfulness = 0.91)
- Save per-example predictions for error analysis

**☐ 5. Human Evaluation**
- Sample 200-300 examples (stratified: random + high/low automated scores)
- 3 annotators per example (inter-annotator agreement)
- Dimensions: Fluency, Relevance, Helpfulness, Safety, Coherence
- Check agreement (Cohen's Kappa > 0.6)

**☐ 6. Compare to Baselines**
- Rule-based baseline (e.g., keyword matching)
- Zero-shot baseline (e.g., GPT-3.5 with prompt)
- Previous model (current production system)
- Ensure new model outperforms all baselines

**☐ 7. A/B Test in Production**
- 3 variants: Baseline (70%), New Model (15%), Alternative (15%)
- Minimum 200-500 samples per variant
- Test statistical significance (p < 0.05)
- Check business impact (CSAT, completion rate)

**☐ 8. Cost-Benefit Analysis**
- Improvement value: +0.5 CSAT × $10k/month = +$5k
- Cost increase: +$0.002/query × 100k queries = +$2k/month
- Net value: $5k - $2k = +$3k/month → APPROVE

**☐ 9. Gradual Rollout**
- Phase 1: 5% traffic (1 week) → Monitor for issues
- Phase 2: 25% traffic (1 week) → Confirm trends
- Phase 3: 50% traffic (1 week) → Final validation
- Phase 4: 100% rollout → Only if all metrics stable

**☐ 10. Production Monitoring**
- Set baseline metrics from first week
- Monitor daily: CSAT, completion rate, accuracy, latency, cost
- Alert on >5% regression in critical metrics
- Weekly review: Check for data drift, quality issues

---

## Common Pitfalls and How to Avoid Them

### Pitfall 1: No Evaluation Strategy

**Symptom:** "I'll just look at a few examples to see if it works."

**Fix:** Mandatory held-out test set with quantitative metrics. Never ship without numbers.

### Pitfall 2: Wrong Metrics for Task

**Symptom:** Using accuracy for generation tasks, BLEU for classification.

**Fix:** Match metric family to task type. See Part 1 tables.

### Pitfall 3: Automated Metrics Only

**Symptom:** BLEU increased to 0.45 but users complain about quality.

**Fix:** Always combine automated + human + production metrics. All three must improve.

### Pitfall 4: Single Metric Optimization

**Symptom:** ROUGE-L optimized but summaries are verbose and contain hallucinations.

**Fix:** Multi-dimensional evaluation with guard rails. Reject regressions on secondary metrics.

### Pitfall 5: No Baseline Comparison

**Symptom:** "Our model achieves 82% accuracy!" (Is that good? Better than what?)

**Fix:** Always compare to baselines: rule-based, zero-shot, previous model.

### Pitfall 6: No A/B Testing

**Symptom:** Deploy new model, discover it's worse than baseline, scramble to rollback.

**Fix:** A/B test with statistical significance before full deployment.

### Pitfall 7: Insufficient Sample Size

**Symptom:** "We tested on 20 examples and it looks good!"

**Fix:** Minimum 200-500 samples for human evaluation, 200-500 per variant for A/B testing.

### Pitfall 8: No Production Monitoring

**Symptom:** Model quality degrades over time (data drift) but nobody notices until users complain.

**Fix:** Continuous monitoring with automated alerts on metric regressions.

---

## Summary

**Evaluation is mandatory, not optional.**

**Complete evaluation = Automated metrics (efficiency) + Human evaluation (quality) + Production metrics (impact)**

**Core principles:**
1. Match metrics to task type (classification vs generation)
2. Multi-dimensional scoring prevents gaming single metrics
3. Human evaluation catches issues automated metrics miss
4. A/B testing proves value before full deployment
5. Production monitoring detects regressions and drift

**Checklist:** Task type → Test set → Metrics → Automated eval → Human eval → Baselines → A/B test → Cost-benefit → Gradual rollout → Production monitoring

Without rigorous evaluation, you don't know if your system works. Evaluation is how you make engineering decisions with confidence instead of guesses.
