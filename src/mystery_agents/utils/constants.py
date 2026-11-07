"""Constants used throughout the mystery agents system."""

# Directory and file names are translated per language (see the "filenames" section of
# the locales); this English name is only used by tests
HOST_GUIDE_FILENAME = "host_guide.md"

# File extensions
MARKDOWN_EXT = ".md"
TEXT_EXT = ".txt"
PDF_EXT = ".pdf"
PNG_EXT = ".png"
JPG_EXT = ".jpg"

# Game ID formatting
GAME_ID_LENGTH = 8

# Directory name patterns
GAME_DIR_PREFIX = "game_"
ZIP_FILE_PREFIX = "mystery_game_"

# LangGraph configuration
DEFAULT_RECURSION_LIMIT = 50
NODE_MAX_ATTEMPTS = 3  # Attempts per node when the LLM output doesn't fit the schema

# Default output directory
DEFAULT_OUTPUT_DIR = "./output"

# Clue generation
MIN_CLUES_PER_GAME = 5

# Language configuration
LANG_CODE_ENGLISH = "en"
DEFAULT_COUNTRY_ES = "Spain"

# Game tone (fixed - not user configurable)
GAME_TONE_DESCRIPTION = (
    "an elegant mystery with wit, balancing classic mystery elements "
    "(à la Agatha Christie) with modern cleverness (à la Knives Out)"
)
GAME_TONE_STYLE = "mystery_party"  # Short tone label for prompts

# LLM model configuration: text models per provider and tier (LLM_PROVIDER picks the
# provider, LLM_MODEL_TIER1..3 override a tier). Images always use Gemini (see below).
# Measured over batches of four games: Gemini 3.8 Flash with tier 1 on medium reasoning
# and tier 2 on low gave the best games (3.5 to 4 out of 5, 2 to 6 contradictions, V2
# passing every game first time); GPT-6 Luna is far cheaper per token but averaged 6 to 11
# contradictions at the same settings.
LLM_PROVIDER_DEFAULT = "google"
LLM_MODELS = {
    "openai": {"tier1": "gpt-6-luna", "tier2": "gpt-6-luna", "tier3": "gpt-6-luna"},
    # gemini-3.8-flash was first dropped for taking minutes per call and refusing prompts
    # (block_reason OTHER); most of the delay was its default thinking, twice 3.6's, and
    # at these efforts it ran without refusals
    "google": {
        "tier1": "gemini-3.8-flash",
        "tier2": "gemini-3.8-flash",
        "tier3": "gemini-3.5-flash-lite",
    },
}
# Reasoning effort per provider and tier (LLM_REASONING_EFFORT_TIER1..3 override it;
# None keeps the model's own default). Tier 1 (plan, grid, sheets, content, reviews)
# needs medium: at low, Luna's games averaged 11 contradictions. The mechanical tier 2
# stays fast on low.
LLM_REASONING_EFFORT: dict[str, dict[str, str | None]] = {
    "openai": {"tier1": "medium", "tier2": "low", "tier3": "low"},
    "google": {"tier1": "medium", "tier2": "low", "tier3": None},
}
# Environment variable holding each provider's API key
LLM_API_KEY_VARS = {"openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}
# Seconds per request. Without a timeout a dropped connection (e.g. the laptop
# going to sleep) hangs the run forever; failing lets the user --resume it.
# google-genai also sends it as the server deadline (X-Server-Timeout), so it must
# leave room for long generations: at 180 s Gemini cut them off with 504 errors.
LLM_REQUEST_TIMEOUT = 600.0
# Output tokens per request, reasoning included. Gemini has looped repeating text until
# its own 65k limit (a 25-minute A7.4 attempt in a measured game); half that still leaves
# several times the largest real answer, and a loop fails sooner so the node retries.
LLM_MAX_OUTPUT_TOKENS = 32768
DRY_RUN_DUMMY_API_KEY = "dry-run-dummy-key"

# Image generation configuration
IMAGE_GENERATION_MODEL = "gemini-nano-banana-2.1"
IMAGE_GENERATION_TEMPERATURE = 0.6
IMAGE_GENERATION_TIMEOUT = 120.0  # Seconds per image request
IMAGE_GENERATION_MAX_RETRIES = 3
IMAGE_GENERATION_RETRY_DELAY_BASE = 2.0  # seconds
IMAGE_GENERATION_MAX_CONCURRENT = 5  # parallel requests limit

# Mock data placeholders (for dry run mode)
MOCK_WORLD_NAME = "Thornfield Manor"
MOCK_VICTIM_NAME = "Lord Reginald Thornfield"
MOCK_DETECTIVE_NAME = "Detective Inspector Morrison"
MOCK_MURDER_TIME = "10:30 PM"
MOCK_WAIT_TIME = "2 minutes"

# Test constants
TEST_DEFAULT_PLAYERS = 6
TEST_MIN_PLAYERS = 4
TEST_DEFAULT_DURATION = 90
TEST_MIN_DURATION = 60
