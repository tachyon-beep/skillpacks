---
description: Review document style against a target writing register (technical, policy, government, public-facing, executive, academic)
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[document_path] [register: technical|policy|government|public-facing|executive|academic]"
---

# Review Document Style

You are reviewing a document's writing register — the conventions of tone, authority, structure, and vocabulary it follows.

## Available Registers

| Register | Key Characteristics |
|----------|-------------------|
| **technical** | Precision, domain vocabulary, code examples, assumes expertise |
| **policy** | Normative "shall/must/may", compliance-oriented, auditable requirements |
| **government** | Institutional voice, statutory authority, plain language mandates |
| **public-facing** | Accessible, no assumed expertise, trust-building, action-oriented |
| **executive** | Outcome-focused, concise, ROI/impact, numbers-first |
| **academic** | Hedged claims, citations, methodology, evidence-grounded |

## Process

### Step 1: Read the Document

Read the target document. Understand its content, structure, and apparent audience.

### Step 2: Determine Mode

**If user specified a register** (e.g., `/review-style doc.md policy`):
Go directly to Review mode. Delegate to `editorial-reviewer` agent with the specified register.

**If user asked to "translate", "adapt", "rewrite", or "convert"**:
Go to Translate mode. Delegate to `editorial-reviewer` agent in translate mode.

**If no register specified** (e.g., `/review-style doc.md`):
Go to Detect mode first:
- Run detection via `editorial-reviewer` agent
- If **High confidence**: auto-proceed to review against the detected register
- If **Medium confidence**: present the detection to the user, ask them to confirm the target register before reviewing
- If **Low confidence**: present the detection, suggest closest match, ask about custom register

### Step 3: Delegate to Agent

Spawn the `editorial-reviewer` agent with:
- The document path
- The confirmed register (or detect-mode instruction)
- The mode (detect, review, or translate)

The agent reads the register definitions from `editorial-registers.md` at runtime.

## Output

The agent produces the output. See `editorial-reviewer` agent for output formats per mode.

## Scope

**This command covers:**
- Register detection (identify which register a document uses)
- Register compliance review (evaluate against a target register)
- Register translation (adapt a document to a different register)

**Not covered:**
- General clarity review: use `/review-docs`
- Documentation structure: use `/review-docs`
- Writing new documentation: use `/write-docs`
- Security documentation review: use ordis-security-architect
