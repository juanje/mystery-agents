"""Constants used throughout the mystery agents system."""

# File and directory names
HOST_DIR = "host"
PLAYERS_DIR = "players"
CLUES_DIR = "clues"

# File names
HOST_GUIDE_FILENAME = "host_guide.md"
AUDIO_SCRIPT_FILENAME = "audio_script.txt"
SOLUTION_FILENAME = "solution.md"
INVITATION_FILENAME = "invitation.txt"
CHARACTER_SHEET_FILENAME = "character_sheet.md"
README_FILENAME = "README.txt"

# File extensions
MARKDOWN_EXT = ".md"
TEXT_EXT = ".txt"

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
