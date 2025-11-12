#!/bin/bash
# Convert all router skills (using-X) to slash commands

set -e

COMMANDS_DIR=".claude/commands"
mkdir -p "$COMMANDS_DIR"

# Find all using-* skills
find plugins -name "SKILL.md" -path "*/using-*/*" | sort | while read skill_file; do
    # Extract the plugin name from path
    # Example: plugins/yzmir-ai-engineering-expert/skills/using-ai-engineering/SKILL.md
    # We want: ai-engineering-expert (remove faction prefix)

    plugin_dir=$(dirname $(dirname $(dirname "$skill_file")))
    plugin_name=$(basename "$plugin_dir")

    # Extract skill name from YAML frontmatter
    skill_name=$(grep "^name:" "$skill_file" | head -1 | sed 's/name: *//' | tr -d '\r')

    # Create command name by removing "using-" prefix
    command_name=${skill_name#using-}
    command_file="$COMMANDS_DIR/${command_name}.md"

    echo "Converting: $skill_file -> $command_file"

    # Copy skill content, stripping YAML frontmatter
    # Find line where YAML ends (second ---) and copy everything after
    awk 'BEGIN{p=0} /^---$/{p++; next} p>=2' "$skill_file" > "$command_file"

    echo "  Created: $command_file"
done

echo ""
echo "Conversion complete! Created $(ls -1 $COMMANDS_DIR/*.md | wc -l) slash commands in $COMMANDS_DIR/"
echo ""
echo "Usage examples:"
echo "  /ai-engineering"
echo "  /system-archaeologist"
echo "  /deep-rl"
echo ""
echo "To see all commands: ls .claude/commands/"
