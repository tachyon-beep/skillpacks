---
description: Create a professionally designed document using Pandoc and/or Typst
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "Edit", "Agent", "AskUserQuestion"]
argument-hint: "[document_type] [source_file_or_description]"
---

# Design Document Command

You are creating a professionally designed document. Your goal is to produce output that looks like it came from a design studio, not from a default template.

## Gather Requirements

Before designing, determine:

1. **Document type**: Report, proposal, specification, resume, letter, brochure, slides?
2. **Source material**: Existing Markdown/content files, or writing from scratch?
3. **Audience**: Executives, engineers, public, regulators?
4. **Brand**: Colors, fonts, logo? Or design from scratch?
5. **Output format**: PDF (default), HTML, DOCX?

If the user provided arguments, parse them. Otherwise ask.

## Dispatch

Launch the **document-designer** agent with the gathered context:

```
Agent({
  subagent_type: "muna-document-designer:document-designer",
  description: "Design [document_type]",
  prompt: "Create a professional [document_type] for [audience]. [Include source files, brand requirements, and any specific design requests from the user.]"
})
```

The agent will:
1. Check font availability (`typst fonts`)
2. Design the template (page setup, colors, typography, layout)
3. Build the document incrementally
4. Run the quality checklist
5. Deliver both the compiled output and reusable source files
