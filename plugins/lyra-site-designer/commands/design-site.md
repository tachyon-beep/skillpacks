---
description: Design or implement a static website for a developer tool or documentation site
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "Edit", "Agent", "AskUserQuestion"]
argument-hint: "[scope: architecture|page|component|tokens] [description]"
---

# Design Site Command

You are designing or implementing a static website for a developer tool, open-source project, or documentation site. Your goal is to produce sites that communicate substance and trust — not marketing fluff.

## Gather Requirements

Before designing, determine:

1. **Scope**: Site architecture, specific page, component, or design tokens?
2. **Project type**: Developer tool, library, framework, specification, documentation?
3. **Audiences**: Who visits the site? (users, contributors, evaluators, etc.)
4. **Existing content**: What content exists? What needs to be written?
5. **Tooling preference**: Hugo, Astro, Eleventy, plain HTML, or no preference?
6. **Brand**: Colors, fonts, logo? Or design from scratch?

If the user provided arguments, parse them. Otherwise ask.

## Dispatch

Launch the **site-designer** agent with the gathered context:

```
Agent({
  subagent_type: "lyra-site-designer:site-designer",
  description: "Design [scope]",
  prompt: "Design/implement [scope] for [project]. Audiences: [audiences]. [Include existing content, tooling preference, brand requirements, and specific requests.]"
})
```

The agent will:
1. Map audiences to their needs
2. Inventory existing content
3. Design structure before style
4. Implement incrementally with working HTML/CSS
5. Verify accessibility and quality
