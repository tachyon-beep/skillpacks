# LLM and AI Security

## Overview

Threat modeling and architectural controls for systems that incorporate large
language models, embeddings, retrieval-augmented generation, agentic tools, or
machine-learning pipelines. This skill covers **threats and controls at the
architecture level**, not LLM application internals (prompt engineering,
fine-tuning, RAG correctness — see `yzmir-llm-specialist`).

**Core Principle**: An LLM is a **second untrusted parser** sitting inside your
application. Anything it ingests (prompts, retrieved documents, tool outputs,
training data) is attacker-controllable input. Anything it emits (text, tool
calls, structured output) is attacker-controllable output. Treat its inputs and
outputs the way you would treat raw HTTP requests and untrusted SQL — never
trusted, always validated at boundaries.

## When to Use

Load this skill when:

- Designing or reviewing a system that calls an LLM (chat, completion,
  embeddings, function calling, MCP, agentic tool use)
- Threat modeling RAG pipelines, vector databases, or document ingestion paths
- Reviewing model-registry usage (HuggingFace, internal model repositories)
- Building agents that execute tools, browse the web, or access internal APIs
- User mentions: "LLM", "prompt injection", "jailbreak", "RAG", "agent",
  "MCP", "model poisoning", "AI safety" (security flavor, not alignment)

**Don't use for:**

- Prompt engineering / fine-tuning quality (`yzmir-llm-specialist`)
- LLM evaluation correctness, hallucination detection, RAG retrieval tuning
  (`yzmir-llm-specialist`)
- General application security with no AI component (use `threat-modeling`)
- Training-pipeline ML engineering (`yzmir-training-optimization`)

**Boundary with `yzmir-llm-specialist`**: This skill answers *"how does an
attacker abuse this system, and what controls block them?"*. The Yzmir skill
answers *"how do I make this LLM application work well?"*.

## Reference Frameworks

| Framework | Version | Scope | Reference |
|-----------|---------|-------|-----------|
| **OWASP Top 10 for LLM Applications** | 2025 | Risks specific to LLM apps | genai.owasp.org |
| **MITRE ATLAS** | Living | Adversarial tactics/techniques for AI | atlas.mitre.org |
| **NIST AI RMF** | 1.0 (Jan 2023) + GenAI Profile (July 2024) | AI risk management | nist.gov/itl/ai-risk-management-framework |
| **EU AI Act** | Regulation (EU) 2024/1689 (in force Aug 2024) | EU-wide AI regulation | eur-lex.europa.eu |
| **ISO/IEC 42001** | 2023 | AI management systems | iso.org |

**Always check current versions** — these documents update on roughly annual
cycles. Cite the version you used in any threat model artifact.

---

## OWASP Top 10 for LLM Applications (2025)

Apply each item as a STRIDE-style enumeration prompt during threat modeling.
Every LLM-using component should be checked against all ten.

### LLM01:2025 — Prompt Injection

**Definition**: Attacker-controlled text causes the model to deviate from its
intended instructions, leak data, or perform unauthorized actions.

**Two variants:**

- **Direct prompt injection**: Attacker is the user; inputs malicious prompt
  directly (`"Ignore previous instructions and..."`).
- **Indirect prompt injection**: Attacker poisons content the model retrieves
  or reads — web pages, emails, PDFs, vector store entries, tool outputs.
  The model treats those as instructions.

**CWE**: CWE-1426 (Improper Validation of Generative AI Output), CWE-77
(Command Injection — conceptual analog), CWE-20 (Improper Input Validation).

**MITRE ATLAS**: AML.T0051 (LLM Prompt Injection), AML.T0054 (LLM Jailbreak).

**Architectural controls** (in order of effectiveness):

1. **Don't give the LLM trust**. Treat its outputs as untrusted user input on
   the way back into your system. Never `eval()` or shell-exec model output.
2. **Separate instructions from data** at the API level. Use system prompts +
   structured tool calls; never concatenate user content into instruction
   text.
3. **Privilege isolation**. The LLM call runs with the privileges of the
   *user*, never the application. An agent acting on behalf of user A must
   not be able to read user B's data even if the prompt says so.
4. **Output filtering / parsers**. Validate model output against a schema
   before passing downstream (e.g., reject SQL keywords, validate URLs
   against an allow-list, schema-validate JSON tool args).
5. **Human-in-the-loop for high-impact actions**. Sending email, executing
   code, transferring money → require explicit confirmation outside the LLM
   conversation channel.
6. **Indirect-injection mitigations**: provenance tagging on retrieved
   content, sandboxed retrieval (no executable content), and "untrusted
   content" delimiters that downstream prompts treat as data.

**Anti-pattern**: "Just tell the model not to follow injected instructions."
This does not work. Models are statistical and adversaries iterate.

### LLM02:2025 — Sensitive Information Disclosure

**Definition**: Model emits secrets, PII, proprietary data, or system prompt
contents.

**Sources of disclosure:**

- Training data memorization (model regurgitates verbatim training samples)
- System prompt leakage (LLM07, treated separately below)
- Context-window leakage between conversations or tenants
- Tool/RAG outputs that include data the user shouldn't see

**CWE**: CWE-200 (Exposure of Sensitive Information), CWE-359 (Privacy
Violation).

**Controls**:

1. **Don't put secrets in prompts**. API keys, internal URLs, customer PII
   used as context — treat the prompt as a log line that may be exfiltrated.
2. **Tenant isolation at retrieval**. RAG queries must be filtered by tenant
   *before* the LLM sees results, not after.
3. **Output-side DLP**. Run model output through PII/secret scanners before
   returning to client (regex + entropy + classifier).
4. **Differential privacy or RLHF for memorization** if training on
   sensitive data (deeper: see `yzmir-llm-specialist`).

### LLM03:2025 — Supply Chain

**Definition**: Compromise via third-party model artifacts, datasets, or
inference dependencies.

**Specific risks:**

- **Pickle exploits**: HuggingFace `.bin`/`.pt`/`.pkl` files execute
  arbitrary code on `torch.load()`. Only load `.safetensors` from untrusted
  sources.
- **Model registry typosquatting**: `transformer-bert-base` vs
  `transformers-bert-base`.
- **Backdoored weights**: Model produces correct output on benchmarks but
  triggers attacker behavior on a specific input pattern (see
  BadNets-style attacks in academic literature).
- **Compromised inference dependencies**: `transformers`, `vllm`, `langchain`
  pulled from PyPI — same supply-chain risks as any Python dep (see
  `supply-chain-security.md`).

**Controls** (cross-link to `supply-chain-security.md`):

1. **Pin model versions by hash**, not by tag. HuggingFace tags are mutable.
2. **Prefer `safetensors`** over pickle-based formats.
3. **SBOM models**: model name + version + SHA + license + provenance.
4. **Sandbox model loading** — no network, no filesystem write, no
   capabilities the model genuinely doesn't need.
5. **Provenance verification**: signed model cards (Sigstore + in-toto are
   emerging here).

### LLM04:2025 — Data and Model Poisoning

**Definition**: Attacker corrupts training data, fine-tuning data, or
embedding store contents to influence model behavior.

**Vectors:**

- Public training data (Common Crawl, Wikipedia) — attacker plants content.
- Fine-tuning datasets sourced from user submissions, support tickets,
  customer feedback.
- RAG vector stores fed from internal documents that an insider can edit.

**MITRE ATLAS**: AML.T0019 (Publish Poisoned Datasets), AML.T0020 (Poison
Training Data).

**Controls**:

1. **Provenance for training data**. Cryptographic hashes, signed manifests,
   chain-of-custody for any dataset used for fine-tune.
2. **Anomaly detection on RAG ingest**. Monitor for sudden additions of
   adversarial-looking text (long instruction-like strings, base64 blobs,
   prompt-injection patterns).
3. **Approval workflow for RAG sources** when content authority is
   ambiguous (user-uploaded files, support tickets).
4. **Canary inputs**. Maintain a held-out test set of known-good queries;
   alert when output drifts.

### LLM05:2025 — Improper Output Handling

**Definition**: Application trusts LLM output and uses it in security-
sensitive contexts (rendering, eval, SQL, shell, navigation) without
validation.

**Examples:**

- Rendering model output as HTML → XSS (CWE-79).
- Passing model-generated SQL to a database → SQL injection (CWE-89).
- `eval()` on model-generated code → RCE (CWE-94).
- Model-generated URLs for navigation → SSRF (CWE-918) or open redirect.

**Controls**:

1. **Treat model output exactly like user input on the way back in.** Same
   sanitization, same parameterization, same allow-list validation.
2. **Structured outputs**: Use JSON schema, regex-constrained generation,
   or function-calling APIs rather than free text.
3. **Output context-aware encoding**: HTML-escape for browser, parameterize
   for SQL, allow-list URL hosts before fetch.

### LLM06:2025 — Excessive Agency

**Definition**: Agent has more capability, permission, or autonomy than its
task requires.

**Three sub-failures:**

- **Excessive functionality**: Tool exposes 20 operations when the agent
  only needs read.
- **Excessive permissions**: Agent's API key has admin scope.
- **Excessive autonomy**: No human approval gates between high-impact
  actions.

**MITRE ATLAS**: AML.T0053 (LLM Plugin Compromise).

**Controls**:

1. **Least privilege per tool**. Each tool exposes the minimum surface
   needed; agent's credentials scope to tool's surface.
2. **Action ladders**. Cheap reads autonomous; writes require schema
   validation; high-impact writes require human approval; irreversible
   actions (delete, transfer, send) require multi-party authorization.
3. **Rate limiting per agent identity**, separate from per-user limits.
4. **Audit log every tool call** with prompt, args, result, decision
   chain — even when the agent is "just looking around".
5. **Capability tokens / authenticated tool use**: tool invocation includes
   user authorization context, not just agent context.

### LLM07:2025 — System Prompt Leakage

**Definition**: Application secrets, business logic, or other proprietary
content placed in the system prompt are extractable by the user.

**Reality check**: System prompts are *recoverable*. Treat them as public.

**Controls**:

1. **Don't put secrets in system prompts**. API keys, internal URLs,
   credentials → env vars or secret manager, not prompt strings.
2. **Don't rely on system prompts for authorization**. "You must not reveal
   X to user Y" is unenforceable. Enforce in code, not text.
3. **Accept leakage**: design the system assuming the prompt is in the
   user's hands.

### LLM08:2025 — Vector and Embedding Weaknesses

**Definition**: Attacks against embedding generation, vector storage, or
retrieval mechanisms.

**Specific risks:**

- **Embedding inversion**: Attacker reconstructs original text from
  embeddings if they leak (treat embeddings as PII when source was PII).
- **RAG poisoning**: Adversarial document crafted to be retrieved for many
  queries (semantic squatting).
- **Cross-tenant retrieval bleed**: Vector DB query lacks tenant filter;
  agent retrieves another tenant's documents.

**Controls**:

1. **Tenant-scoped indices** OR mandatory tenant filter at query time
   (verified server-side, not client-supplied).
2. **Treat embeddings of sensitive content as sensitive**. They reverse.
3. **Retrieval allow-lists** for high-trust agents.
4. **Anomaly monitoring** on retrieval scores and source distribution.

### LLM09:2025 — Misinformation

**Definition**: Model produces confident, incorrect output that downstream
systems or users act on.

**Security framing**: This is a **safety / quality** issue with security
implications when the misinformation drives security-relevant decisions
(e.g., agent is asked "is this URL safe?" and confidently misanswers).

**Controls** (security-relevant only — see `yzmir-llm-specialist` for
quality):

1. **Don't use the LLM for security decisions** that have a deterministic
   answer. Use the LLM to *explain* a deterministic check, not replace it.
2. **Cite-and-verify** for retrieved facts in agentic contexts.

### LLM10:2025 — Unbounded Consumption

**Definition**: Resource exhaustion via prompt floods, long contexts,
recursion, or expensive operations.

**Vectors:**

- Token cost inflation (giant prompts).
- Recursive agent loops (agent calls itself indefinitely).
- Context-window blowup (RAG returns huge documents).
- Embedding-generation flood.

**CWE**: CWE-400 (Uncontrolled Resource Consumption), CWE-770 (Allocation
Without Limits).

**Controls**:

1. **Per-user, per-agent token budgets** (daily and per-request).
2. **Max iteration count** for agentic loops; hard timeout.
3. **Context-window bounds** with retrieval truncation.
4. **Cost monitoring + alerts** on cost-per-request anomalies.

---

## Threat Modeling LLM Systems

Use STRIDE as the spine, but add LLM-specific decomposition.

### Decomposition for an LLM System

Identify each of:

1. **Prompt sources**: Who/what supplies prompt text? (user, system, RAG,
   tool output, prior turn)
2. **Trust labels per source**: Which are attacker-controllable, fully or
   partially?
3. **Tools / actuators**: What can the model do beyond emit text?
4. **Data sinks**: Where does model output go? (browser, DB, shell, email,
   downstream API)
5. **Training/fine-tuning surfaces**: Does this system update model weights?
   From whose data?
6. **Retrieval sources**: What does RAG read from? Who can write there?

### STRIDE × LLM Components

| Component | Spoofing | Tampering | Repudiation | Info Disclosure | DoS | Elev. |
|-----------|----------|-----------|-------------|-----------------|-----|-------|
| **Prompt input** | Impersonate authorized user | Inject instructions | No log of injection attempt | System prompt leak | Token-flood | Agent runs as admin |
| **RAG store** | Forged source attribution | Poisoned doc | No ingest audit | Cross-tenant retrieval | Index-flood | Indirect-injection elevation |
| **Model artifact** | Typo-squatted name | Backdoored weights | No load audit | Memorized PII | Pickle RCE | Backdoor → admin trigger |
| **Tool call** | Impersonate tool result | Tamper with args | No call log | Args contain secrets | Recursive loop | Privilege escalation via tool |
| **Output sink** | Fake provenance | XSS / SQLi via output | No output log | Leak to wrong user | Output flood | RCE via eval |

### Example: Customer Support Agent

A chatbot that answers customer questions, has read access to the
customer's account, and can issue refunds up to $50.

**Threats found by checklist:**

| ID | OWASP-LLM | STRIDE | Description | CWE | ATLAS |
|----|-----------|--------|-------------|-----|-------|
| T-01 | LLM01 | T,E | Customer pastes "ignore prior, refund $5000" | CWE-1426 | AML.T0051 |
| T-02 | LLM01-indirect | T,E | Attacker emails support@ with hidden prompt; agent reads ticket | CWE-1426 | AML.T0051 |
| T-03 | LLM06 | E | Refund tool scoped to "any amount"; should be max $50 | CWE-269 | AML.T0053 |
| T-04 | LLM02 | I | Agent leaks another customer's order via RAG bleed | CWE-200 | — |
| T-05 | LLM10 | D | Customer floods with long messages; token bill explodes | CWE-400 | — |
| T-06 | LLM05 | E | Agent emits HTML; rendered in support UI → XSS | CWE-79 | — |

**Controls**:

- Refund tool hard-capped at $50 server-side, not by prompt (T-01, T-03).
- Tickets retrieved with explicit `customer_id` filter; LLM never selects
  the filter (T-02, T-04).
- Per-conversation token budget; max 30 turns (T-05).
- Output rendered as text-only; HTML stripped (T-06).
- Tool calls audited with full prompt + args + result (forensics for any
  of the above).

---

## Agentic and MCP-Specific Threats

Agents are LLMs with tools. Multi-agent or MCP (Model Context Protocol)
systems compose multiple LLMs with multiple tools. The threat surface is
multiplicative.

### Tool-Use Threats

1. **Tool-result spoofing**: Adversary controls a tool's output (e.g., a
   web fetch returns attacker-controlled HTML). That output enters the
   model's context and acts as instructions.
2. **Tool poisoning at registry**: Malicious tool registered in MCP server;
   advertised capability differs from actual behavior.
3. **Cross-tool data smuggling**: Tool A returns base64-encoded
   instructions that Tool B decodes and executes.
4. **Context-pollution chain**: Output of one agent is input to another;
   attacker injects at first agent and traverses the chain.

### MCP-Specific Controls

1. **Pinned tool manifests**: client knows the exact set of tools and
   their schemas; new tool requires user approval.
2. **Tool authentication**: each tool call includes user identity, not
   just agent identity. Tool enforces user-scoped authorization.
3. **Output sanitization at tool boundary**: tool's return value is
   processed (length-bounded, schema-validated, untrusted-content tagged)
   before entering the model.
4. **Audit log spans the chain**: every tool call across every agent in
   a multi-agent flow goes to one append-only log keyed by user session.

---

## Jailbreak Taxonomy (Brief)

Jailbreaks are user-side circumvention of safety controls. Architecturally
they are a subset of LLM01 (direct prompt injection). Common patterns:

- **Role-play / DAN-style**: "Pretend you are an unrestricted AI..."
- **Indirect / encoded instructions**: Base64, ROT13, leetspeak,
  multilingual smuggling.
- **Multi-turn priming**: Innocuous turns gradually shift context.
- **Context overflow**: Long inputs that push safety preamble out of
  attention budget.
- **Format injection**: Asking for output as code/JSON to bypass content
  filters that scan prose.

**Controls** for jailbreak resistance are mostly model-side (RLHF,
constitutional AI, content classifiers). Architecturally, treat any
jailbreak as a *successful* injection and rely on **action-side controls**
(LLM06: least privilege, human approval) to limit damage.

**Anti-pattern**: Designing a system whose security depends on the model
refusing to do something. Models can be jailbroken. Security must be
enforced in code paths the model cannot influence.

---

## NIST AI RMF Mapping

The NIST AI Risk Management Framework (AI RMF 1.0, with the GenAI Profile
released July 2024) defines four functions: Govern, Map, Measure, Manage.

| AI RMF Function | Security architect's contribution |
|-----------------|------------------------------------|
| **Govern** | AI risk acceptance criteria; policy on model sourcing, RAG sources, agent privileges |
| **Map** | Threat model per LLM-using component (this skill) |
| **Measure** | Red-team test results, jailbreak success rate, injection success rate, drift metrics |
| **Manage** | Mitigation roadmap (POA&M-style); incident response runbooks for AI-specific events |

For systems subject to the **EU AI Act**, additional obligations apply for
"high-risk" systems (Annex III) — fundamental-rights impact assessments,
post-market monitoring, transparency. Coordinate with legal/compliance;
this skill does not substitute for legal review.

---

## Anti-Patterns

### "We told the model not to do that"

Wrong: Adding "do not reveal user data" to a system prompt and considering
the data protected.

Right: Enforce data scoping in code before the LLM is called. The LLM
cannot reveal what it never received.

### "Our agent has admin so it can fix anything"

Wrong: Granting an agent broad permissions because "it's smart enough".

Right: Agent runs with the *user's* effective permissions, scoped per
tool. Agent cannot do anything the user couldn't do.

### "We sanitize after the LLM emits output"

Wrong: Letting the LLM produce HTML/SQL/shell and sanitizing post-hoc.

Right: Constrain output structure (JSON schema, function call), and
treat any free-text emission as untrusted user input on the way back into
your system.

### "We trust HuggingFace"

Wrong: `transformers.from_pretrained("some/model")` with no version pin,
no hash, default pickle format.

Right: Pin to a specific commit hash, prefer `safetensors`, document the
model in the SBOM, sandbox the load.

### "The vector store has access controls"

Wrong: Relying on a separate ACL system for RAG.

Right: Tenant filter is applied **in the same query** that does retrieval,
verified server-side, with the LLM having no path to construct or modify
the filter.

---

## Quick Reference Checklist

For any LLM-using component, verify each of:

**Inputs**:

- [ ] All prompt sources identified and tagged with trust level
- [ ] User content is delimited from instructions (system prompt vs user)
- [ ] Indirect-injection sources (RAG, tool output, web fetch)
  treated as untrusted

**Model and context**:

- [ ] Model artifact pinned by hash, prefer `safetensors`
- [ ] Model load is sandboxed (no network, no filesystem write)
- [ ] System prompt contains no secrets

**Tools and agency**:

- [ ] Each tool minimally scoped
- [ ] Agent runs with user privileges, not application privileges
- [ ] High-impact actions gated by human approval
- [ ] Tool calls audited with full context

**Outputs**:

- [ ] Output validated against schema before downstream use
- [ ] HTML/SQL/shell paths use parameterization, not concatenation
- [ ] PII/secret scanning on outbound text
- [ ] Per-user / per-agent rate and token limits

**RAG and data**:

- [ ] Tenant filter on every retrieval (server-side enforced)
- [ ] RAG source provenance and approval workflow
- [ ] Embeddings of sensitive content treated as sensitive
- [ ] Anomaly monitoring on retrieval and ingest

**Supply chain**:

- [ ] Models, datasets, and inference deps in SBOM
- [ ] Pickle exploits mitigated (safetensors or sandbox)
- [ ] (See `supply-chain-security.md` for full controls)

---

## Cross-References

**Use WITH this skill**:

- `threat-modeling.md` — STRIDE/attack-tree spine; this skill plugs in
  LLM-specific decomposition
- `security-controls-design.md` — designing the action-side controls that
  remain effective when the model is jailbroken
- `supply-chain-security.md` — model-registry, dataset, and inference-dep
  controls

**Use AFTER this skill**:

- `documenting-threats-and-controls.md` — recording the LLM threat model
- `security-architecture-review.md` — review checklist application

**Cross-faction**:

- `yzmir-llm-specialist:safety` — model-side safety techniques
  (RLHF, constitutional AI, content classifiers)
- `yzmir-llm-specialist:rag-audit` — RAG correctness (paired with the
  RAG threats above)

---

## Summary

**LLM/AI security is application security with two extra rules:**

1. **The model is an untrusted parser.** Its inputs are attacker-influenced;
   its outputs are attacker-influenced. Validate at both boundaries.
2. **Security cannot live in the prompt.** Models can be jailbroken,
   prompts can be extracted. Enforce in code paths the model cannot
   influence.

**OWASP Top 10 for LLMs (2025)** is the current enumeration. **MITRE ATLAS**
provides the technique IDs for cross-referencing with classic ATT&CK.
**NIST AI RMF + GenAI Profile (July 2024)** is the governance overlay.
**EU AI Act** adds regulatory obligations for high-risk systems in EU
markets.

When in doubt: assume the prompt is leaked, assume the model is jailbroken,
and check whether anything bad happens. If yes, design action-side controls
(LLM06 — least agency) until the answer is no.
