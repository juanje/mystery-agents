"""Constants used throughout the mystery agents system."""

# File and directory names
HOST_DIR = "game"  # Renamed from "host" for clarity
PLAYERS_DIR = "characters"  # Renamed from "players" and flattened
CLUES_DIR = "clues"

# File names
HOST_GUIDE_FILENAME = "host_guide.md"
SOLUTION_FILENAME = "solution.md"
INVITATION_FILENAME = "invitation.txt"
CHARACTER_SHEET_FILENAME = "character_sheet.md"

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
PLAYER_DIR_PREFIX = "player_"
CLUE_FILE_PREFIX = "clue_"
ZIP_FILE_PREFIX = "mystery_game_"

# LangGraph configuration
DEFAULT_RECURSION_LIMIT = 50

# Default output directory
DEFAULT_OUTPUT_DIR = "./output"

# Timeline display limits
MAX_TIMELINE_EVENTS_DISPLAY = 50

# Clue generation
MIN_CLUES_PER_GAME = 5

# Language configuration
LANG_CODE_ENGLISH = "en"
LANG_CODE_SPANISH = "es"
LANG_CODE_ITALIAN = "it"
LANG_CODE_GERMAN = "de"
LANG_NAME_ENGLISH = "English"
LANG_NAME_SPANISH = "Spanish"
LANG_NAME_ITALIAN = "Italian"
LANG_NAME_GERMAN = "German"
DEFAULT_COUNTRY_EN = "United States"
DEFAULT_COUNTRY_ES = "Spain"
DEFAULT_COUNTRY_IT = "Italy"
DEFAULT_COUNTRY_DE = "Germany"

# Game tone (fixed - not user configurable)
GAME_TONE_DESCRIPTION = (
    "an elegant mystery with wit, balancing classic mystery elements "
    "(à la Agatha Christie) with modern cleverness (à la Knives Out)"
)
GAME_TONE_SHORT = "Cluedo meets Knives Out"
GAME_TONE_STYLE = "mystery_party"  # Internal identifier (legacy)

# LLM model configuration
LLM_MODEL_TIER1 = "gemini-2.5-pro"
LLM_MODEL_TIER2 = "gemini-2.5-pro"
LLM_MODEL_TIER3 = "gemini-2.5-flash"
LLM_TEMPERATURE_TIER1 = 0.6
LLM_TEMPERATURE_TIER2 = 0.7
LLM_TEMPERATURE_TIER3 = 0.3
DRY_RUN_DUMMY_API_KEY = "dry-run-dummy-key"

# Image generation configuration
IMAGE_GENERATION_MODEL = "models/gemini-2.5-flash-image"
IMAGE_GENERATION_TEMPERATURE = 0.6
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
