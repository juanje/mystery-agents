# Mystery Agents

AI-powered mystery party game generator using LangGraph and LangChain.

**Generate complete, playable murder mystery parties in minutes** — characters, clues, host guide, and everything ready to print!

📖 **[Read about the Game Experience](game_experience.md)** - See what playing a generated mystery is like

---

## Table of Contents

- [What is Mystery Agents?](#what-is-mystery-agents)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Usage Guide](#usage-guide)
  - [Basic Usage](#basic-usage)
  - [Configuration Options](#configuration-options)
  - [Character Image Generation](#character-image-generation)
- [What You Get](#what-you-get)
- [Technical Details](#technical-details)
  - [Architecture](#architecture)
  - [Project Structure](#project-structure)
  - [Development](#development)
- [AI Tools Disclaimer](#-ai-tools-disclaimer)
- [License](#license)

---

## What is Mystery Agents?

Mystery Agents generates **complete murder mystery party games** in the style of Cluedo and Knives Out. Give it your preferences (theme, era, language, number of players), and it creates:

- 🎭 **Unique characters** with secrets, motives, and backstories
- 🕵️ **A solvable mystery** with timeline, clues, and red herrings
- 📦 **Print-ready materials** (character sheets, host guide, clues)
- 🎨 **AI-generated portraits** for each character (optional)
- 🌍 **Cultural adaptation** - names and customs match your country/region
- 🗣️ **Native language generation** - Spanish, English, Italian, German, or Hebrew (with RTL support)
- 🎭 **Flexible killer reveal** - optionally let the killer know their identity from the start

**Perfect for:**
- Game masters planning a mystery party
- Writers exploring murder mystery structures
- Anyone who wants to host an interactive Cluedo-style event

---

## Quick Start

### 1. Prerequisites

- **Python 3.12+**
- **uv** package manager ([install here](https://github.com/astral-sh/uv))
- **PDF dependencies** (WeasyPrint):
  ```bash
  # Ubuntu/Debian
  sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0 libgdk-pixbuf2.0-0 libffi-dev libcairo2
  
  # Fedora
  sudo dnf install pango gdk-pixbuf2 cairo
  
  # macOS
  brew install pango gdk-pixbuf cairo
  export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
  ```

### 2. Install

```bash
git clone https://github.com/juanje/mystery-agents.git
cd mystery-agents
uv sync
```

### 3. Configure API Key

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey), then:

```bash
# Option 1: Create .env file (recommended)
cp env.example .env
nano .env  # Add: GOOGLE_API_KEY=your-key-here

# Option 2: Environment variable
export GOOGLE_API_KEY='your-key-here'
```

### 4. Generate Your First Game

```bash
# Create your configuration file
cp game.example.yml game.yml
nano game.yml  # Edit with your preferences

# Generate your game
uv run mystery-agents

# Or test without API calls (uses example config)
uv run mystery-agents game.example.yml --dry-run
```

**Done!** Your game package is in `output/mystery_game_xxxxx.zip` 🎉

---

## Key Features

### Core Capabilities

- ✅ **Complete Game Generation** - All materials for a live-action mystery party
- ✅ **Host-Detective Mode** - Host plays victim in Act 1, detective in Act 2
- ✅ **Act 1 Objectives** - Each player gets 2-3 goals creating social tension
- ✅ **Automatic Validation** - Three validation stages with retry loops, plus code checks of every sheet
- ✅ **Print-Ready PDFs** - Professional PDFs ready to print (markdown source optional with `--keep-work-dir`)

### Customization

- 🎨 **Themes**: Family mansion, corporate retreat, cruise ship, train, or custom
- 📅 **Eras**: Modern, 1920s, Victorian, or custom period
- 🌍 **Cultural Adaptation**: Character names, customs, settings match your country/region
- 👥 **Flexible Players**: 4-10 players, customizable gender distribution
- 🗣️ **Languages**: Spanish, English, Italian, German, Hebrew (native generation, not translation, with RTL support for Hebrew)

### AI Features

- 🖼️ **Character Portraits** - AI-generated images with consistent visual style
- 🎯 **Visual Style Guide** - Unified aesthetic (color palette, lighting, no text overlays)
- 🔄 **Parallel Generation** - Fast image creation with rate limiting
- 🤖 **Google Gemini Integration** - gemini-3.6-flash for logic and content, Nano Banana 2.1 for portraits

---

## Usage Guide

### Basic Usage

**Quick Start:**
```bash
# 1. Create your configuration file
cp game.example.yml game.yml
nano game.yml  # Edit with your preferences

# 2. Generate your game
uv run mystery-agents

# The tool automatically uses game.yml if it exists
```

**Alternative: Specify Config File:**
```bash
# Use a specific config file
uv run mystery-agents my-game.yml

# Test with example config (no API calls)
uv run mystery-agents game.example.yml --dry-run
```

**Configuration File Format (game.yml):**
```yaml
language: es              # es, en, it, de, or he
country: Spain
region: Andalucía         # Optional
epoch: modern             # modern, 1920s, victorian, custom
theme: family_mansion     # family_mansion, corporate_retreat, cruise, train, custom
players:
  male: 3
  female: 3
host_gender: male
duration_minutes: 90      # 60-180
difficulty: medium        # easy, medium, hard
murder_method: random     # random, poison, stabbing, blunt_force, shooting, strangling, fall
killer_knows_identity: false  # If true, killer knows they're the murderer
```

### Configuration Options

**CLI Flags:**
```bash
# Logging and Output
-v, --verbose        # Increase verbosity (-v=INFO logs, -vv=DEBUG logs)
--quiet              # Minimal output (only start/end/errors)
--log-file FILE      # Write logs to specified file

# Execution Options
--dry-run            # Use mock data (no API calls)
--debug              # Log LLM model responses (works with -v/-vv)
--no-images          # Skip character portrait generation
--keep-work-dir      # Keep intermediate markdown files
--output-dir DIR     # Custom output directory
--resume GAME_ID     # Resume an interrupted generation
```

**Resuming an interrupted generation:**

Progress is saved after every step in `<output-dir>/.checkpoints.sqlite`. If a run
fails (API outage, quota, Ctrl-C...), the error message shows the command to continue
from the last completed step, reusing the options the game was started with:

```bash
mystery-agents --resume 1a2b3c4d
```

Saved progress for a game is removed once it finishes.

**Command Format:**
```bash
mystery-agents [GAME_CONFIG_FILE] [OPTIONS]

# Uses game.yml by default if no config file specified
# GAME_CONFIG_FILE: Path to YAML game configuration file (optional, defaults to game.yml)
```

**Logging Modes:**

| Mode | Output | Use Case |
|------|--------|----------|
| Default | Visual progress messages | User-friendly, shows what's happening |
| `-v` | Structured INFO logs with timestamps | Debugging, CI/CD, performance analysis |
| `-vv` | INFO + DEBUG logs | Deep debugging, includes cache operations |
| `--quiet` | Minimal (config, result, errors only) | Scripts, automation |
| `--log-file` | Always writes INFO+ logs to file | Post-analysis, performance review (independent of console verbosity) |

**Examples:**
```bash
# Default: Clean visual progress
mystery-agents game.yml

# Best practice: Visual progress + detailed logs for analysis
mystery-agents game.yml --log-file generation.log

# Verbose: See what's happening with timestamps
mystery-agents game.yml -v

# Debug: Full detailed logs including cache operations
mystery-agents game.yml -vv

# Quiet: Minimal output for scripts
mystery-agents game.yml --quiet

# Debug with LLM responses logged
mystery-agents game.yml -vv --debug
```

**Environment Variables:**

Create a `.env` file (or set environment variables):
```bash
# Required
GOOGLE_API_KEY=your-api-key-here

# Optional: Override LLM models per tier
LLM_MODEL_TIER1=gemini-3.6-flash       # Plot logic, sheets, written content, reviews
LLM_MODEL_TIER2=gemini-3.6-flash       # Characters, relationships, timeline
LLM_MODEL_TIER3=gemini-3.5-flash-lite  # Placeholder: no text agent calls it

# Optional: Override temperatures (unset by default; Gemini 3.6+ models
# use fixed sampling and ignore them)
LLM_TEMPERATURE_TIER1=0.6
LLM_TEMPERATURE_TIER2=0.7
LLM_TEMPERATURE_TIER3=0.3
```

See `env.example` for full configuration options.

### Character Image Generation

**Enabled by default.** Uses Google's Nano Banana 2.1 image model.

**Features:**
- ✅ Photorealistic portraits coherent with game world
- ✅ Visual style consistency (unified color palette, lighting)
- ✅ Period-appropriate styling (1920s, Victorian, Modern)
- ✅ No text/labels in images
- ✅ Parallel generation (~1-2 minutes for 6-8 characters)
- ✅ Embedded in PDFs, saved separately

**Cost & Performance:**
- Model: `gemini-nano-banana-2.1`
- Typical game: 6-8 images in ~1-2 minutes
- **Note**: Image generation incurs API costs. Use `--no-images` to skip.

```bash
# With images (default)
uv run mystery-agents

# Without images (no cost, faster)
uv run mystery-agents --no-images
```

---

## What You Get

### Output Structure

```
output/game_xxxxx/
├── mystery_game_xxxxx.zip       # Complete package (PDFs only)
│
├── /game/                       # Host materials (⚠️ SPOILERS!)
│   ├── host_guide.pdf           # Complete instructions
│   ├── solution.pdf             # The answer
│   └── clue_reference.pdf       # All clues with metadata
│
├── /characters/                 # Player packages
│   ├── Name_character_sheet.pdf    # Character + objectives
│   ├── Name_invitation.pdf         # Player invitation
│   ├── victim_character_sheet.pdf  # Host's Act 1 role
│   ├── detective_character_sheet.pdf  # Host's Act 2 role
│   └── ...
│
└── /clues/                      # Act 2 clues (no spoilers)
    ├── clue_01.pdf
    ├── clue_02.pdf
    └── ...
```

### Key Features

- 📄 **PDF-Only Package** - Everything print-ready, no clutter
- 📁 **Flat Structure** - All files easy to access
- 🏷️ **Clear Naming** - `Name_character_sheet.pdf`, `clue_01.pdf`
- 🖼️ **Embedded Images** - Portraits in PDFs, images separate
- 🌐 **Native Language** - Generated in target language (not translated)
- 🎨 **Professional Styling** - Beautiful PDFs via WeasyPrint
- 📦 **Complete ZIP** - Ready to share and print

### Optional: Intermediate Files

**By default**, only PDFs are kept in the final package. Markdown files are generated internally but deleted after PDF conversion.

Use `--keep-work-dir` to preserve intermediate files for inspection:
```
output/
├── game_xxxxx/          # Final PDFs (default output)
└── _work_xxxxx/         # Markdown sources + images (with --keep-work-dir)
```

---

## How It Works

Mystery Agents generates your game in a multi-phase pipeline:

### Generation Pipeline

**Phase 1: World Setup**
- Loads your configuration
- Creates the game world (location, atmosphere, period)
- Validates coherence and retries if needed
- Generates a unified visual style guide

**Phase 2: Characters**
- Creates suspects with personalities, secrets, and goals
- Defines relationships between characters

**Phase 3: Crime & Logic**
- Designs the crime (victim, method, scene)
- Creates the shared timeline of the evening
- Selects the killer, their route and the deduction plan
- Places every suspect in the house (presence grid)
- Writes each suspect's sheet from the grid (alibis, cover stories, observations)
- Validates game logic and retries if needed

**Phase 4: Content & Packaging**
- Writes the clues, the host guide and the solution, in turn
- Generates AI portraits of the suspects (optional, in parallel)
- Generates host images (victim/detective portraits)
- Creates PDFs and packages everything into a ZIP

### Quality Assurance

The system includes **three automatic validation loops**:

1. **World Validation** - Ensures setting coherence (location, culture, period)
2. **Logic Validation** - Verifies killer logic and that the sheets agree with each other
3. **Content Validation** - Checks clues, guide and solution against what players hold

Code also checks what it can measure (sightings the floor plan rules out, too many
witnesses, spoilers) and fixes it or sends it back to its writer. If validation fails,
the system retries (up to 2-3 times); once the plot is valid, a game whose texts still
have issues is packaged anyway.

### Tech Stack

- **LangGraph** - Multi-agent workflow orchestration
- **Google Gemini** - Text and image generation
- **Pydantic** - Data validation
- **WeasyPrint** - PDF generation
- **Python 3.12+** with full type annotations

📚 **For developers**: See [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) for architecture details, design decisions, and contribution guidelines.

### Development

**Setup:**
```bash
# Install with dev dependencies (the dev group is synced by default)
uv sync
```

**Quality Checks:**
```bash
# Linting
uv run ruff check . --fix
uv run ruff format .

# Type checking
uv run mypy src/

# Tests (216 tests)
uv run pytest

# All checks
uv run ruff check . --fix && uv run ruff format . && uv run mypy src/ && uv run pytest
```

**Tech Stack:**
- **Python 3.12+** with type annotations
- **LangGraph** - Multi-agent workflow orchestration
- **LangChain** - LLM abstraction and tools
- **Google Gemini** - Text and image generation
- **Pydantic v2** - Data validation and modeling
- **WeasyPrint** - PDF generation from HTML/Markdown
- **Click** - CLI framework
- **pytest** - Testing framework

---

## 🤖 AI Tools Disclaimer

<details>
<summary>This project was developed with the assistance of artificial intelligence tools</summary>

**Tools used:**
- **Cursor**: Code editor with AI capabilities
- **Claude-Sonnet-4**: Anthropic's language model (claude-sonnet-4-20250514)

**Division of responsibilities:**

**AI (Cursor + Claude-Sonnet-4)**:
- 🔧 Initial code prototyping
- 📝 Generation of examples and test cases
- 🐛 Assistance in debugging and error resolution
- 📚 Documentation and comments writing
- 💡 Technical implementation suggestions

**Human (Juanje Ojeda)**:
- 🎯 Specification of objectives and requirements
- 🔍 Critical review of code and documentation
- 💬 Iterative feedback and solution refinement
- ✅ Final validation of concepts and approaches

**Collaboration philosophy**: AI tools served as a highly capable technical assistant, while all design decisions, educational objectives, and project directions were defined and validated by the human.
</details>

---

## License

MIT

## Author

- **Juanje Ojeda**
- Email: juanje.ojeda@gmail.com
- GitHub: <https://github.com/juanje/mystery-agents>
