
# Prompt Engineering Patterns

## Context

You're writing prompts for an LLM and getting inconsistent or incorrect outputs. Common issues:
- **Vague instructions**: Model guesses intent (inconsistent results)
- **No examples**: Model infers task from description alone (ambiguous)
- **No output format**: Model defaults to prose (unparsable)
- **No reasoning scaffolding**: Model jumps to answer (errors in complex tasks)
- **System message misuse**: Task instructions in system message (inflexible)

**This skill provides effective prompt engineering patterns for chat / instruct models: specificity, few-shot examples, format specification, chain-of-thought, and proper message structure.**

---

## Reasoning Models vs Chat / Instruct Models — Read This First

**The patterns in this sheet target chat / instruct models** (the conventional autoregressive assistants: GPT-4-class chat, Claude chat, Gemini Pro chat, Llama-Instruct, Mistral-Instruct, Qwen-Chat, etc.).

**If you're targeting a reasoning model**, prompt rules differ — sometimes drastically. Reasoning models include:
- OpenAI o-series (o1, o3, o4-mini and successors)
- Anthropic Claude with extended thinking
- DeepSeek-R1 and distilled R1 variants
- Google Gemini "thinking" / Deep Think modes
- Qwen QwQ and reasoning-tuned successors

For these models, explicit chain-of-thought prompting often **hurts** (the model already reasons internally) and verbose few-shot examples can degrade performance. **See [reasoning-models.md](reasoning-models.md) for reasoning-model prompting guidance.**

### Decision tree (run before writing any prompt)

```
Q1: Is the target a reasoning model (o-series / Claude thinking / R1 / Gemini thinking / QwQ)?
    YES → Stop. Use reasoning-models.md instead.
    NO  → Continue.

Q2: Is the model a frontier-general or fast-cheap chat / instruct model?
    YES → This sheet applies.
    NO  → on-device / specialty (code, embedding) → this sheet's principles still apply,
          but few-shot quality matters more on smaller models.
```

> **Capability tiers** (used throughout this pack instead of hardcoded model IDs):
> `frontier-reasoning` (o-series-class), `frontier-general` (flagship chat / instruct), `fast-cheap` (mid-tier), `on-device` (local / quantized). Verify current model IDs in provider docs.

---

## Core Principle: Be Specific

**Vague prompts → Inconsistent outputs**

**Bad:**
```
Analyze this review: "Product was okay."
```

**Why bad:**
- "Analyze" is ambiguous (sentiment? quality? topics?)
- No scale specified (1-5? positive/negative?)
- No output format (text? JSON? number?)

**Good:**
```
Rate this review's sentiment on a scale of 1-5:
1 = Very negative
2 = Negative
3 = Neutral
4 = Positive
5 = Very positive

Review: "Product was okay."

Output ONLY the number (1-5):
```

**Result:** Consistent "3" every time

### Specificity Checklist:

- Define the task clearly (classify, extract, generate, summarize)
- Specify the scale (1-5, 1-10, percentage, positive/negative/neutral)
- Define edge cases (null values, ambiguous inputs, relative dates)
- Specify output format (JSON, CSV, number only, yes/no)
- Set constraints (max length, required fields, allowed values)


## Prompt Structure

### Message Roles:

**1. System Message:**
```python
system = """
You are an expert Python programmer with 10 years of experience.
You write clean, efficient, well-documented code.
You always follow PEP 8 style guidelines.
"""
```

**Purpose:**
- Sets role/persona (expert, assistant, teacher)
- Defines global behavior (concise, detailed, technical)
- Applies to entire conversation

**Best practices:**
- Keep it short (< 200 words)
- Define WHO the model is, not WHAT to do
- Set tone and constraints

**2. User Message:**
```python
user = """
Write a Python function that calculates the Fibonacci sequence up to n terms.

Requirements:
- Use recursion with memoization
- Include docstring
- Handle edge cases (n <= 0)
- Return list of integers

Output only the code, no explanations.
"""
```

**Purpose:**
- Specific task instructions (per-request)
- Input data
- Output format requirements

**Best practices:**
- Be specific about requirements
- Include examples if ambiguous
- Specify output format explicitly

**3. Assistant Message (in conversation):**
```python
messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": "Calculate 2+2"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "Now multiply that by 3"},
]
```

**Purpose:**
- Conversation history
- Shows model previous responses
- Enables multi-turn conversations


## Few-Shot Learning

**Show, don't tell.** Examples teach better than instructions.

### 0-Shot (No Examples):

```
Extract the person, company, and location from this text:

Text: "Tim Cook presented the new iPhone at Apple's Cupertino campus."
```

**Issues:**
- Model guesses format (JSON? Key-value? List?)
- Edge cases unclear (What if no person? Multiple companies?)

### 1-Shot (One Example):

```
Extract entities as JSON.

Example:
Text: "Satya Nadella spoke at Microsoft in Seattle."
Output: {"person": "Satya Nadella", "company": "Microsoft", "location": "Seattle"}

Now extract from:
Text: "Tim Cook presented the new iPhone at Apple's Cupertino campus."
Output:
```

**Better!** Model sees format and structure.

### Few-Shot (3-5 Examples - BEST):

```
Extract entities as JSON.

Example 1:
Text: "Satya Nadella spoke at Microsoft in Seattle."
Output: {"person": "Satya Nadella", "company": "Microsoft", "location": "Seattle"}

Example 2:
Text: "Google announced Gemini in Mountain View."
Output: {"person": null, "company": "Google", "location": "Mountain View"}

Example 3:
Text: "The event took place online with no speakers."
Output: {"person": null, "company": null, "location": "online"}

Now extract from:
Text: "Tim Cook presented the new iPhone at Apple's Cupertino campus."
Output:
```

**Why 3-5 examples?**
- 1 example: Shows format
- 2-3 examples: Shows variation and edge cases
- 4-5 examples: Shows complex patterns
- > 5 examples: Diminishing returns (uses more tokens)

### Few-Shot Best Practices:

1. **Cover edge cases:**
   - Null values (missing entities)
   - Multiple values (list of people)
   - Ambiguous cases (nickname vs full name)

2. **Show desired format consistently:**
   - All examples use same structure
   - Same field names
   - Same data types

3. **Order matters:**
   - Put most representative example first
   - Put edge cases later
   - Model learns from all examples

4. **Balance examples:**
   - Show positive and negative cases
   - Show simple and complex cases
   - Avoid bias (don't show only easy examples)


## Chain-of-Thought (CoT) Prompting

CoT can mean two different things in 2026, and they have nearly opposite implications:

### A. CoT as a prompt technique (chat / instruct models)

On a non-reasoning chat / instruct model, asking the model to "think step by step" or showing few-shot examples that include intermediate steps reliably improves accuracy on multi-step tasks. This is the original sense from Wei et al. 2022, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (arxiv 2201.11903), and the zero-shot variant from Kojima et al. 2022, "Large Language Models are Zero-Shot Reasoners" (arxiv 2205.11916), which established that the trigger phrase "Let's think step by step" alone yields large gains on arithmetic and commonsense reasoning benchmarks for instruct models.

**Use CoT prompting on chat / instruct models for:**
- Math word problems
- Logic puzzles
- Multi-step reasoning
- Complex decision-making
- Ambiguous questions

**Skip CoT for:**
- Simple classification (sentiment)
- Direct lookups (capital of France)
- Pattern matching (regex, entity extraction)
- **Reasoning models (see B)**

#### Without CoT (Direct):

```
Q: A farmer has 17 sheep. All but 9 die. How many sheep are left?
A:
```

**Output:** "8 sheep" (wrong — misreads "all but 9")

#### With CoT:

```
Q: A farmer has 17 sheep. All but 9 die. How many sheep are left?

Think step-by-step:
1. Start with how many sheep
2. Understand what "all but 9 die" means
3. Calculate remaining sheep
4. State the answer

A:
```

**Output:**
```
1. The farmer starts with 17 sheep
2. "All but 9 die" means all sheep except 9 die
3. So 9 sheep remain alive
4. Answer: 9 sheep
```

#### CoT Variants:

**1. Explicit steps:**
```
Solve step-by-step:
1. Identify what we know
2. Identify what we need to find
3. Set up the equation
4. Solve
5. Verify the answer
```

**2. Zero-shot trigger (Kojima et al. 2022):**
```
Q: [question]
A: Let's think step by step.
```

**3. "Explain your reasoning":**
```
Q: [question]
A: I'll explain my reasoning:
```

### B. CoT as a model property (reasoning models)

Reasoning models (o-series, Claude extended thinking, DeepSeek-R1, Gemini thinking, Qwen QwQ) generate an internal chain of thought *as part of how they were trained to answer*. You don't ask them to think step by step; they already do. Adding explicit CoT instructions on top:

- **Often hurts accuracy** — the model's trained reasoning policy is overridden by your scaffolding.
- **Wastes thinking tokens** — your "think step by step" duplicates work the model is already doing.
- **Distorts the answer format** — these models expect terse problem statements, not reasoning recipes.

**Rule of thumb on reasoning models:** state the problem clearly, specify the desired output format, and stop. **See [reasoning-models.md](reasoning-models.md) for full guidance** including how to choose reasoning effort, when to use developer messages, and how to budget thinking tokens.


## Output Formatting

**Specify format explicitly. Don't assume model knows what you want.**

### JSON Output:

**Bad (no format specified):**
```
Extract the name, age, and occupation from: "John is 30 years old and works as an engineer."
```

**Output:** "The person's name is John, who is 30 years old and works as an engineer."

**Good (format specified):**
```
Extract information as JSON:

Text: "John is 30 years old and works as an engineer."

Output in this format:
{
  "name": "<string>",
  "age": <number>,
  "occupation": "<string>"
}

JSON:
```

**Output:**
```json
{
  "name": "John",
  "age": 30,
  "occupation": "engineer"
}
```

### CSV Output:

```
Convert this data to CSV format with columns: name, age, city.

Data: John is 30 and lives in NYC. Mary is 25 and lives in LA.

CSV (with header):
```

**Output:**
```csv
name,age,city
John,30,NYC
Mary,25,LA
```

### Structured Text:

```
Summarize this article in bullet points (max 5 points):

Article: [text]

Summary:
-
```

### XML/HTML:

```
Format this data as HTML table:

Data: [data]

HTML:
```

### Reliable structured output: don't parse prose, constrain generation

For machine-consumed JSON, prompt-only formatting is a fallback, not the primary tool. Use the providers' structured-output features:

- **OpenAI Structured Outputs / `response_format={"type": "json_schema", ...}`** — guarantees the response matches a JSON schema.
- **Anthropic tool use** — define a tool whose `input_schema` is your target schema and force the model to call it.
- **Google Gemini `responseSchema`** — Gemini's structured output mode.
- **[Outlines](https://github.com/dottxt-ai/outlines)** (dottxt-ai) — open-source constrained-decoding library; works with vLLM, Ollama, transformers. Pass a Pydantic model or regex as the output type.
- **[Instructor](https://github.com/jxnl/instructor)** — wrapper that "give me a Pydantic model and I'll get the LLM to fit it" via function calling, with retries on schema-validation failure.

**Order of preference:** provider-native structured output → Instructor (proprietary APIs) → Outlines (self-hosted) → prompt-only JSON formatting.

For full agentic / tool-use patterns (multi-tool orchestration, MCP servers, error recovery, observability), see **[agentic-patterns-and-mcp.md](agentic-patterns-and-mcp.md)**.

### Format Best Practices:

1. **Show the schema:**
   ```json
   {
     "field1": "<type>",
     "field2": <type>,
     ...
   }
   ```

2. **Specify data types:** `<string>`, `<number>`, `<boolean>`, `<array>`

3. **Show example output:** Full example of expected output

4. **Request validation:** "Output valid JSON" or "Ensure CSV is parsable"


## Temperature and Sampling

**Temperature controls randomness. Adjust based on task.**

> The examples below use the `openai>=1.0` client (`client.chat.completions.create`). All providers expose equivalent parameters; check provider docs.

### Temperature = 0 (Deterministic):

```python
from openai import OpenAI
client = OpenAI()

# use a frontier-general model; check provider docs for current IDs
response = client.chat.completions.create(
    model="<frontier-general>",  # provider's flagship chat / instruct tier
    messages=[...],
    temperature=0,  # Deterministic, near-identical outputs
)
```

> Note: even at `temperature=0`, most hosted APIs are not bit-exact deterministic across calls (batching, kernel non-determinism). Use a fixed `seed` parameter where the provider supports it for stronger reproducibility.

**Use for:**
- Classification (sentiment, category)
- Extraction (entities, data fields)
- Structured output (JSON, CSV)
- Factual queries (capital of X, date of Y)

**Why:** Need consistency and correctness, not creativity

### Temperature = 0.7-1.0 (Creative):

```python
# use a frontier-general model; check provider docs for current IDs
response = client.chat.completions.create(
    model="<frontier-general>",
    messages=[...],
    temperature=0.8,  # Creative, varied outputs
)
```

**Use for:**
- Creative writing (stories, poems)
- Brainstorming (ideas, alternatives)
- Conversational chat (natural dialogue)
- Content generation (marketing copy)

**Why:** Want variety and creativity, not determinism

### Temperature = 1.5-2.0 (Very Random):

```python
# use a frontier-general model; check provider docs for current IDs
response = client.chat.completions.create(
    model="<frontier-general>",
    messages=[...],
    temperature=1.8,  # Very random, surprising outputs
)
```

**Use for:**
- Experimental generation
- Highly creative tasks

**Warning:** May produce nonsensical outputs (use carefully)

### Top-p (Nucleus Sampling):

```python
# use a frontier-general model; check provider docs for current IDs
response = client.chat.completions.create(
    model="<frontier-general>",
    messages=[...],
    temperature=0.7,
    top_p=0.9,  # Consider top 90% probability mass
)
```

**Alternative to temperature:**
- top_p = 1.0: Consider all tokens (default)
- top_p = 0.9: Consider top 90% (filters low-probability tokens)
- top_p = 0.5: Consider top 50% (more focused)

**Best practice:** Use temperature OR top_p, not both. Reasoning models often ignore or override these — see [reasoning-models.md](reasoning-models.md).


## Common Task Patterns

### 1. Classification:

```
Classify the sentiment of this review as 'positive', 'negative', or 'neutral'.
Output ONLY the label.

Review: "The product works great but shipping was slow."

Sentiment:
```

**Key elements:**
- Clear categories ('positive', 'negative', 'neutral')
- Output constraint ("ONLY the label")
- Prompt ends with field name ("Sentiment:")

### 2. Extraction:

```
Extract all dates from this text. Output as JSON array.

Text: "Meeting on March 15, 2024. Follow-up on March 22."

Format:
["YYYY-MM-DD", "YYYY-MM-DD"]

Output:
```

**Key elements:**
- Specific format (JSON array)
- Date format specified (YYYY-MM-DD)
- Shows example structure

### 3. Summarization:

```
Summarize this article in 50 words or less. Focus on the main conclusion and key findings.

Article: [long text]

Summary (max 50 words):
```

**Key elements:**
- Length constraint (50 words)
- Focus instruction (main conclusion, key findings)
- Clear output label

### 4. Generation:

```
Write a product description for a wireless mouse with these features:
- Ergonomic design
- 1600 DPI sensor
- 6-month battery life
- Bluetooth 5.0

Style: Professional, concise (50-100 words)

Product Description:
```

**Key elements:**
- Input data (features list)
- Style guide (professional, concise)
- Length constraint (50-100 words)

### 5. Transformation:

```
Convert this SQL query to Python (using pandas):

SQL:
SELECT name, age FROM users WHERE age > 30 ORDER BY age DESC

Python (pandas):
```

**Key elements:**
- Clear source and target formats
- Shows example input
- Labels expected output

### 6. Question Answering:

```
Answer this question based ONLY on the provided context. If the answer is not in the context, say "I don't know."

Context: [document]

Question: What is the return policy?

Answer:
```

**Key elements:**
- Constraint ("based ONLY on context")
- Fallback instruction ("I don't know")
- Prevents hallucination


## Advanced Techniques

### 1. Self-Consistency (Wang et al. 2022):

**Generate multiple outputs, take majority vote.** From Wang et al. 2022, "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (arxiv 2203.11171).

```python
answers = []
for _ in range(5):
    response = llm.generate(prompt, temperature=0.7)
    answers.append(response)

# Take majority vote
final_answer = Counter(answers).most_common(1)[0][0]
```

**Use for:**
- Complex reasoning (math, logic) on chat / instruct models
- When single answer might be wrong
- Accuracy > cost

**Trade-off:** N× cost for accuracy improvement (Wang reports +6 to +18 points on reasoning benchmarks).

**Note:** With reasoning models, self-consistency-style sampling is usually built into the model's own search. Don't reinvent it externally — see [reasoning-models.md](reasoning-models.md).

### 2. Tree-of-Thoughts (Yao et al. 2023):

**Explore multiple reasoning paths, evaluate, pick best.** From Yao et al. 2023, "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (arxiv 2305.10601). On Game of 24, ToT raised GPT-4 from 4% to 74%.

```
Problem: [complex problem]

Let's consider 3 different approaches:

Approach 1: [reasoning path 1]
Approach 2: [reasoning path 2]
Approach 3: [reasoning path 3]

Evaluate each (sound? complete? efficient?):
[evaluation]

Best approach: [selection]

Now solve using the best approach:
[solution]
```

**Use for:**
- Complex planning
- Strategic decision-making
- Multiple valid solutions

### 3. ReAct (Yao et al. 2022):

**Interleave reasoning with actions (tool use).** From Yao et al. 2022, "ReAct: Synergizing Reasoning and Acting in Language Models" (arxiv 2210.03629). The classic agent loop pattern: thought → action → observation, repeat.

```
Task: What's the weather in the city where the Eiffel Tower is located?

Thought: I need to find where the Eiffel Tower is located.
Action: Search "Eiffel Tower location"
Observation: The Eiffel Tower is in Paris, France.

Thought: Now I need the weather in Paris.
Action: Weather API call for Paris
Observation: 15°C, partly cloudy

Answer: It's 15°C and partly cloudy in Paris.
```

**Use for:**
- Multi-step tasks with tool use
- Search + reasoning
- API interactions

This sheet shows the prompt shape only. **For full agentic patterns** — multi-tool orchestration, MCP (Model Context Protocol) servers, error recovery, planner / executor splits, agent observability — see **[agentic-patterns-and-mcp.md](agentic-patterns-and-mcp.md)**.

### 4. Instruction Following:

**Separate instructions from data.**

```
Instructions:
- Extract all email addresses
- Validate format (user@domain.com)
- Remove duplicates
- Sort alphabetically

Data:
[text with emails]

Output (JSON array):
```

**Best practice:** Clearly separate "Instructions" from "Data". When data may contain instructions (untrusted user input), this separation is also a basic prompt-injection defense — see [llm-safety-alignment.md](llm-safety-alignment.md) and `ordis-security-architect`.


## Debugging Prompts

**If output is wrong, diagnose systematically.**

### Problem 1: Inconsistent outputs

**Diagnosis:**
- Instructions too vague?
- No examples?
- Temperature too high?

**Fix:**
- Add specificity
- Add 3-5 examples
- Set temperature=0 (and `seed` if supported)

### Problem 2: Wrong format

**Diagnosis:**
- Format not specified?
- Example format missing?

**Fix:**
- Use provider structured-output features (response_format / tool_use / Outlines / Instructor)
- Specify format explicitly
- Show example output structure
- End prompt with format label ("JSON:", "CSV:")

### Problem 3: Factual errors

**Diagnosis:**
- Hallucination (model making up facts)?
- No chain-of-thought (chat / instruct only)?

**Fix:**
- Add "based only on provided context"
- Request "cite your sources"
- Add "if unsure, say 'I don't know'"
- For chat / instruct models: add CoT scaffolding
- For reasoning models: increase reasoning effort instead — see [reasoning-models.md](reasoning-models.md)

### Problem 4: Too verbose

**Diagnosis:**
- No length constraint?
- No "output only" instruction?

**Fix:**
- Add word/character limit
- Add "output ONLY the [X], no explanations"
- Show concise examples

### Problem 5: Misses edge cases

**Diagnosis:**
- Edge cases not in examples?
- Instructions don't cover edge cases?

**Fix:**
- Add edge case examples (null, empty, ambiguous)
- Explicitly mention edge case handling


## Prompt Testing

**Test prompts systematically before production.**

### 1. Create test cases:

```python
test_cases = [
    # Normal cases
    {"input": "...", "expected": "..."},
    {"input": "...", "expected": "..."},

    # Edge cases
    {"input": "", "expected": "null"},  # Empty input
    {"input": "...", "expected": "null"},  # Missing data

    # Ambiguous cases
    {"input": "...", "expected": "..."},
]
```

### 2. Run tests:

```python
for case in test_cases:
    output = llm.generate(prompt.format(input=case["input"]))
    assert output == case["expected"], f"Failed on {case['input']}"
```

### 3. Measure metrics:

```python
# Accuracy
correct = sum(1 for case in test_cases if output == case["expected"])
accuracy = correct / len(test_cases)

# Consistency (run same input 10 times)
outputs = [llm.generate(prompt) for _ in range(10)]
consistency = len(set(outputs)) == 1  # All outputs identical?

# Latency
import time
start = time.time()
output = llm.generate(prompt)
latency = time.time() - start
```

For full evaluation methodology (test-set design, contamination checks, eval suites like Inspect AI / OLMES / lm-evaluation-harness, LLM-as-judge bias), see **[llm-evaluation-metrics.md](llm-evaluation-metrics.md)**.


## Prompt Optimization Workflow

**Iterative improvement process:**

### Step 1: Baseline prompt (simple)

```
Classify sentiment: [text]
```

### Step 2: Test and measure

```python
accuracy = 65%  # Too low!
consistency = 40%  # Very inconsistent
```

### Step 3: Add specificity

```
Classify sentiment as 'positive', 'negative', or 'neutral'.
Output ONLY the label.

Text: [text]
Sentiment:
```

**Result:** accuracy = 75%, consistency = 80%

### Step 4: Add few-shot examples

```
Classify sentiment as 'positive', 'negative', or 'neutral'.

Examples:
[3 examples]

Text: [text]
Sentiment:
```

**Result:** accuracy = 88%, consistency = 95%

### Step 5: Add edge case handling

```
[Include edge case examples in few-shot]
```

**Result:** accuracy = 92%, consistency = 98%

### Step 6: Optimize for cost/latency

```python
# Reduce examples from 5 to 3 (latency 400ms → 300ms)
# Accuracy still 92%
```

For repeated prompts with stable prefixes (system message + few-shot block), use **prompt caching** to reduce cost and latency dramatically. See **[context-engineering-and-prompt-caching.md](context-engineering-and-prompt-caching.md)**.

**Final:** accuracy = 92%, consistency = 98%, latency = 300ms


## Prompt Libraries and Templates

**Reusable templates for common tasks.**

### Template 1: Classification

```
Classify {item} as one of: {categories}.

{optional: 3-5 examples}

Output ONLY the category label.

{item}: {input}

Category:
```

### Template 2: Extraction

```
Extract {fields} from the text. Output as JSON.

{optional: 3-5 examples showing format and edge cases}

Text: {input}

JSON:
```

### Template 3: Summarization

```
Summarize this {content_type} in {length} words or less.
Focus on {aspects}.

{content_type}: {input}

Summary ({length} words max):
```

### Template 4: Generation

```
Write {output_type} with these characteristics:
{characteristics}

Style: {style}
Length: {length}

{output_type}:
```

### Template 5: Chain-of-Thought (chat / instruct only)

```
{question}

Think step-by-step:
1. {step_1_prompt}
2. {step_2_prompt}
3. {step_3_prompt}

Answer:
```

**Usage:**
```python
prompt = CLASSIFICATION_TEMPLATE.format(
    item="review",
    categories="'positive', 'negative', 'neutral'",
    input=review_text
)
```


## Anti-Patterns

### Anti-pattern 1: "The model is stupid"

**Wrong:** "The model doesn't understand. I need a better model."

**Right:** "My prompt is ambiguous. Let me add examples and specificity."

**Principle:** Most quality issues are prompt issues, not model issues. (The exception: targeting the wrong tier — putting a fast-cheap model on a frontier-reasoning task.)

### Anti-pattern 2: "Just run it multiple times"

**Wrong:** "Run 10 times and take the average/majority."

**Right:** "Fix the prompt so it's consistent (temperature=0, specific instructions)."

**Principle:** Consistency should come from the prompt, not multiple runs. Self-consistency is for *reasoning accuracy*, not for papering over a bad prompt.

### Anti-pattern 3: "Parse the prose output"

**Wrong:** "I'll extract JSON from the prose with regex."

**Right:** Use provider structured-output features, Outlines, or Instructor.

**Principle:** Constrain generation to the schema, don't parse prose after the fact.

### Anti-pattern 4: "System message for everything"

**Wrong:** Put task instructions in system message.

**Right:** System = role/behavior, User = task/instructions.

**Principle:** System message is global (all requests), user message is per-request.

### Anti-pattern 5: "More tokens = better"

**Wrong:** "I'll write a 1000-word prompt with every detail."

**Right:** "I'll write a concise prompt with 3-5 examples."

**Principle:** Concise + examples > verbose instructions.

### Anti-pattern 6: "Same prompt for chat and reasoning models"

**Wrong:** Take a heavily scaffolded CoT prompt and run it on o-series or R1.

**Right:** Strip the scaffolding for reasoning models; keep it for chat / instruct. See [reasoning-models.md](reasoning-models.md).

**Principle:** Reasoning is a property of the model, not always a property of the prompt.

### Anti-pattern 7: "Hardcoded model IDs"

**Wrong:** `model="gpt-4"` baked into production code.

**Right:** Use a config indirection (`model=settings.frontier_general_model`) and pin via configuration, not source.

**Principle:** Model lineups change quarterly. Capability tiers (frontier-reasoning / frontier-general / fast-cheap / on-device) outlive specific IDs.


## Summary

**Core principles:**

1. **Pick the right tier first**: chat / instruct vs reasoning model — they want different prompts.
2. **Be specific**: Define scale, edge cases, constraints, output format.
3. **Use few-shot**: 3-5 examples teach better than instructions (on chat / instruct).
4. **Specify format via the API, not just the prompt**: structured outputs, tool use, Outlines, Instructor.
5. **Request reasoning explicitly only on chat / instruct**: CoT, self-consistency, ToT.
6. **Correct message structure**: System = role, User = task.

**Temperature:**
- 0 (+ `seed` if available): Classification, extraction, structured output (deterministic).
- 0.7-1.0: Creative writing, brainstorming (varied).
- Reasoning models often ignore these — see [reasoning-models.md](reasoning-models.md).

**Common patterns:**
- Classification: Specify categories, output constraint
- Extraction: Format + examples + edge cases
- Summarization: Length + focus areas
- Generation: Features + style + length

**Advanced (chat / instruct):**
- Self-consistency (Wang 2022): Multiple runs + majority vote
- Tree-of-thoughts (Yao 2023): Multiple reasoning paths
- ReAct (Yao 2022): Reasoning + action; full agent patterns in [agentic-patterns-and-mcp.md](agentic-patterns-and-mcp.md)

**Debugging:**
- Inconsistent → Add specificity, examples, temperature=0
- Wrong format → Use structured-output APIs
- Factual errors → Add context constraints, CoT (chat) or higher reasoning effort (reasoning)
- Too verbose → Add length limits, "output only"

**Key insight:** Prompts are code. Treat them like code: test, iterate, optimize, version control.

**See also:**
- [reasoning-models.md](reasoning-models.md) — prompting o-series, Claude thinking, R1, Gemini thinking, QwQ
- [agentic-patterns-and-mcp.md](agentic-patterns-and-mcp.md) — tool use, MCP, agent loops, structured-output integration
- [context-engineering-and-prompt-caching.md](context-engineering-and-prompt-caching.md) — prompt caching, long-context layout
- [llm-evaluation-metrics.md](llm-evaluation-metrics.md) — measuring prompt quality
- [llm-safety-alignment.md](llm-safety-alignment.md) and `ordis-security-architect` — prompt injection, adversarial robustness

---

*Model lineup current as of 2026-05; revisit quarterly.*
