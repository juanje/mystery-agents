"""Pydantic models for game state and data structures."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

# --- Tipos básicos ---
DifficultyLevel = Literal["easy", "medium", "hard"]
Tone = Literal["mystery_party"]  # Fixed tone: elegant mystery with wit (Cluedo meets Knives Out)
Epoch = Literal["modern", "1920s", "victorian", "custom"]
Theme = Literal["family_mansion", "corporate_retreat", "cruise", "train", "custom"]


# --- Meta y configuración ---
class MetaInfo(BaseModel):
    """Metadata about the game."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: str = "v1.6"


class PlayerConfig(BaseModel):
    """Configuration for player distribution."""

    total: int = Field(ge=4, le=10)
    male: int = 0
    female: int = 0


class GameConfig(BaseModel):
    """Game configuration and preferences."""

    language: Literal["es", "en"] = "es"
    country: str = "Spain"
    region: str | None = None
    epoch: Epoch = "modern"
    custom_epoch_description: str | None = None
    theme: Theme = "family_mansion"
    custom_theme_description: str | None = None
    tone: Tone = "mystery_party"
    players: PlayerConfig = Field(default_factory=lambda: PlayerConfig(total=6))
    host_gender: Literal["male", "female"] = "male"
    duration_minutes: int = Field(90, ge=60, le=180)
    difficulty: DifficultyLevel = "medium"
    generate_images: bool = False
    dry_run: bool = False
    debug_model: bool = False
    config_file: str | None = None  # Path to YAML config file (skips wizard if provided)


# --- Mundo ---
class WorldBible(BaseModel):
    """World and setting definition."""

    epoch: str = Field(
        description="The historical period or era (e.g., 'Modern', '1920s', 'Victorian', 'Custom'). Must match the config epoch."
    )
    location_type: str = Field(
        description="Type of location where the mystery takes place (e.g., 'Mansion', 'Cruise Ship', 'Corporate Building', 'Train')."
    )
    location_name: str = Field(
        description="Specific name of the location (e.g., 'Blackwood Manor', 'SS Orient Express', 'TechCorp Headquarters')."
    )
    summary: str = Field(
        description="A 2-3 sentence summary of the setting, atmosphere, and context for the mystery party."
    )
    gathering_reason: str = Field(
        description="The in-game reason why all characters are gathered at this location (e.g., 'Memorial dinner honoring the late Lord Cavendish', 'Reading of the will', '60th birthday celebration', 'Annual family reunion'). This provides narrative context for the event."
    )
    visual_keywords: list[str] = Field(
        default_factory=list,
        description="List of visual/atmospheric keywords to describe the setting (e.g., ['gothic', 'elegant', 'candlelit', 'opulent']).",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="List of constraints or rules for the setting (e.g., 'No modern technology', 'Limited access to certain rooms').",
    )


# --- Crimen ---
class TimeWindow(BaseModel):
    """Time window for events."""

    start: str = Field(description="Start time in HH:MM format (e.g., '20:00').")
    end: str = Field(description="End time in HH:MM format (e.g., '21:30').")


class VictimSpec(BaseModel):
    """Victim specification (the host in Act 1)."""

    id: str = Field(
        default_factory=lambda: f"victim-{uuid4().hex[:8]}",
        description="Unique identifier for the victim (auto-generated if not provided).",
    )
    name: str = Field(
        description="Full name of the victim character (e.g., 'Lord Blackwood', 'CEO Sarah Chen'). This is the character the HOST will roleplay in Act 1."
    )
    age: int = Field(description="Age of the victim (integer, e.g., 45, 65).")
    gender: Literal["male", "female"] = Field(
        description="Gender of the victim: 'male' or 'female'. This should match the host's gender."
    )
    role_in_setting: str = Field(
        description="The victim's role in the setting (e.g., 'Mansion owner and patriarch', 'Ship captain', 'CEO of the company'). This should be central to the location."
    )
    public_persona: str = Field(
        description="How the victim appears to others - their public image and personality (e.g., 'Wealthy, controlling, with many secrets', 'Charming but manipulative')."
    )
    secrets: list[str] = Field(
        default_factory=list,
        description="List of the victim's secrets that will be revealed during the investigation (e.g., ['Changed will recently', 'Had affairs', 'Financial troubles']).",
    )


class MurderMethod(BaseModel):
    """Murder method specification."""

    type: Literal["stabbing", "poison", "shooting", "blunt_force", "other"] = Field(
        default="other",
        description="Type of murder method. Must be one of: 'stabbing', 'poison', 'shooting', 'blunt_force', or 'other'.",
    )
    description: str = Field(
        description="Detailed description of how the murder was committed (e.g., 'Poison in evening drink', 'Stabbed in the back with a letter opener')."
    )
    weapon_used: str = Field(
        description="The specific weapon or method used (e.g., 'Arsenic-laced brandy', 'Kitchen knife', 'Pistol', 'Candlestick')."
    )
    live_action_method_description: str | None = Field(
        default=None,
        description="Optional description of how the murder will be staged in live action (e.g., 'Host drinks from glass, then collapses after 30 seconds').",
    )


class CrimeScene(BaseModel):
    """Crime scene specification."""

    room_id: str = Field(
        description="Unique identifier for the room where the murder occurs (e.g., 'study', 'dining_room', 'master_bedroom'). This should match a room in the location."
    )
    description: str = Field(
        description="Description of the crime scene room (e.g., 'The victim's private study with mahogany desk and floor-to-ceiling bookshelves')."
    )
    scene_description_post_discovery: str | None = Field(
        default=None,
        description="Optional description of how the scene appears when discovered (e.g., 'Body found slumped in armchair', 'Victim lying on the floor near the desk').",
    )


class OpportunitySpec(BaseModel):
    """Opportunity window for a suspect."""

    character_id: str = Field(
        description="ID of the suspect character who has this opportunity. Must match a character ID from the suspects list."
    )
    can_be_alone_with_victim: bool = Field(
        description="Whether this suspect could be alone with the victim during this time window (true/false)."
    )
    time_window: TimeWindow = Field(
        description="Time window when this opportunity exists (start and end times in HH:MM format)."
    )
    notes: str = Field(
        description="Additional notes about this opportunity (e.g., 'Had access to the study', 'Was seen near the victim's room')."
    )


class CrimeSpec(BaseModel):
    """Complete crime specification."""

    victim: VictimSpec = Field(
        description="The victim character specification. This is the character the HOST will roleplay in Act 1 before the murder."
    )
    murder_method: MurderMethod = Field(
        description="Details about how the murder was committed, including method type, description, and weapon used."
    )
    crime_scene: CrimeScene = Field(
        description="Information about where the murder occurred, including room ID, description, and scene appearance."
    )
    time_of_death_approx: str = Field(
        description="Approximate time of death in HH:MM format (e.g., '22:30'). This is when the murder occurs during the party."
    )
    possible_weapons: list[str] = Field(
        default_factory=list,
        description="List of possible weapons or objects found at the scene (e.g., ['poison bottle', 'brandy glass', 'unknown vial']). Can be empty if not applicable.",
    )
    possible_opportunities: list[OpportunitySpec] = Field(
        default_factory=list,
        description="List of opportunity windows for suspects to commit the crime. IMPORTANT: Use an empty array [] by default - opportunities will be defined later in the timeline when character IDs are available. Only include opportunities if you have valid character IDs with the exact format: character_id (string), can_be_alone_with_victim (boolean), time_window (object with start/end in HH:MM), notes (string).",
    )


# --- Personajes (Sospechosos) ---
class CharacterSpec(BaseModel):
    """Character/suspect specification."""

    id: str = Field(
        default_factory=lambda: f"char-{uuid4().hex[:8]}",
        description="Unique identifier for the character (auto-generated if not provided).",
    )
    name: str = Field(
        description="Full name of the character (e.g., 'John Smith', 'Maria Garcia')."
    )
    gender: Literal["male", "female"] = Field(
        description="Gender of the character: 'male' or 'female'."
    )
    age_range: str = Field(
        description="Age range of the character (e.g., '25-30', '40-50', '60+')."
    )
    role: str = Field(
        description="The character's role or profession (e.g., 'Business partner', 'Family member', 'Servant')."
    )
    public_description: str = Field(
        description="How the character appears to others - their public image (e.g., 'Charming and outgoing businessman')."
    )
    personality_traits: list[str] = Field(
        default_factory=list,
        description="List of personality traits (e.g., ['clever', 'suspicious', 'charming']).",
    )
    relation_to_victim: str = Field(
        description="How this character relates to the victim (e.g., 'Nephew', 'Business partner', 'Former lover')."
    )
    personal_secrets: list[str] = Field(
        default_factory=list,
        description="List of the character's secrets that may be revealed (e.g., ['Has gambling debts', 'Was blackmailed by victim']).",
    )
    personal_goals: list[str] = Field(
        default_factory=list,
        description="List of the character's personal goals (e.g., ['Inherit the estate', 'Protect a secret']).",
    )
    act1_objectives: list[str] = Field(
        default_factory=list,
        description="Specific actionable objectives for Act 1 that involve interacting with other characters (e.g., 'Convince Alejandro to return the money he owes you', 'Discover who is blackmailing Elena'). These create social tension and roleplay opportunities during the party.",
    )
    motive_for_crime: str | None = Field(
        default=None,
        description="Why this character might want to kill the victim (e.g., 'Financial gain', 'Revenge for past wrongs'). Can be null.",
    )
    costume_suggestion: str | None = Field(
        default=None,
        description="Suggestion for what costume this character should wear (e.g., '1920s flapper dress', 'Business suit'). Can be null.",
    )
    live_action_killer_instructions: str | None = Field(
        default=None,
        description="Special instructions if this character is the killer (for live action). Can be null.",
    )
    image_path: str | None = Field(
        default=None,
        description="Path to the generated character portrait image (e.g., 'output/game_xxx/images/char_xxx.png'). Generated by A3.5 by default (unless --no-images is used).",
    )


class RelationshipSpec(BaseModel):
    """Relationship between characters."""

    id: str = Field(
        default_factory=lambda: f"rel-{uuid4().hex[:8]}",
        description="Unique identifier for the relationship (auto-generated if not provided).",
    )
    from_character_id: str = Field(
        description="ID of the first character in the relationship. Must match a character ID."
    )
    to_character_id: str = Field(
        description="ID of the second character in the relationship. Must match a character ID."
    )
    type: Literal["family", "romantic", "professional", "rivalry", "friendship", "other"] = Field(
        default="other",
        description="Type of relationship: 'family', 'romantic', 'professional', 'rivalry', 'friendship', or 'other'.",
    )
    description: str = Field(
        description="Description of the relationship (e.g., 'Long-standing business rivalry', 'Secret romantic affair')."
    )
    tension_level: int = Field(1, ge=1, le=3, description="Tension level from 1 (low) to 3 (high).")


# --- Timeline global ---
class GlobalEvent(BaseModel):
    """A global event in the timeline."""

    id: str = Field(
        default_factory=lambda: f"gevt-{uuid4().hex[:8]}",
        description="Unique identifier for the event (auto-generated if not provided).",
    )
    time_approx: str = Field(
        description="Approximate time of the event in HH:MM format (e.g., '20:30', '21:15')."
    )
    description: str = Field(
        description="Description of what happened in this event (e.g., 'Dinner is served', 'Guests arrive and mingle')."
    )
    character_ids_involved: list[str] = Field(
        default_factory=list,
        description="List of character IDs involved in this event. Can be empty if no characters are involved.",
    )
    room_id: str | None = Field(
        default=None,
        description="ID of the room where the event occurs (e.g., 'dining_room', 'study'). Can be null if not applicable.",
    )


class TimeBlock(BaseModel):
    """A time block containing events."""

    id: str = Field(
        default_factory=lambda: f"tb-{uuid4().hex[:8]}",
        description="Unique identifier for the time block (auto-generated if not provided).",
    )
    start: str = Field(description="Start time of the block in HH:MM format (e.g., '20:00').")
    end: str = Field(description="End time of the block in HH:MM format (e.g., '21:00').")
    events: list[GlobalEvent] = Field(
        default_factory=list,
        description="List of events that occur during this time block. Can be empty.",
    )


class GlobalTimeline(BaseModel):
    """Global timeline of events."""

    time_blocks: list[TimeBlock] = Field(
        default_factory=list,
        description="List of time blocks that make up the timeline. Each block has a start/end time and contains events.",
    )
    live_action_murder_event: GlobalEvent | None = Field(
        default=None,
        description="The specific event that represents the murder in live action (e.g., 'Lights go out and a scream is heard'). Can be null.",
    )


class KillerSelection(BaseModel):
    """Killer selection and rationale."""

    killer_id: str = Field(
        description="ID of the character who is the killer. Must match one of the suspect character IDs."
    )
    rationale: str = Field(
        description="Explanation of why this character was chosen as the killer (e.g., 'Has strongest motive and opportunity')."
    )
    modified_events: list[str] = Field(
        default_factory=list,
        description="List of descriptions of any timeline events that need to be adjusted for the solution to work. Can be empty.",
    )
    truth_narrative: str = Field(
        description="Complete narrative of what actually happened - the full solution to the mystery."
    )


# --- Timelines personales ---
class PersonalEvent(BaseModel):
    """Personal event from a character's perspective."""

    id: str = Field(default_factory=lambda: f"pevt-{uuid4().hex[:8]}")
    global_time_block_id: str
    what_they_really_did: str
    what_they_will_tell_others: str
    info_they_observed: list[str] = []
    hidden_actions: str | None = None


class PersonalTimeline(BaseModel):
    """Personal timeline for a character."""

    character_id: str
    events: list[PersonalEvent] = []
    subjective_narrative: str


PersonalTimelineByCharacter = dict[str, PersonalTimeline]


# --- Pistas y Mapas ---
ClueType = Literal["note", "object", "forensic_report", "map_snippet", "photo", "other"]


class ClueSpec(BaseModel):
    """Clue specification."""

    id: str = Field(default_factory=lambda: f"clue-{uuid4().hex[:8]}")
    type: ClueType
    title: str
    description: str
    incriminates: list[str] = []
    exonerates: list[str] = []
    relates_to_events: list[str] = []
    reveal_phase: Literal["Acto 2"] = "Acto 2"
    assigned_to_character_id: str | None = None
    is_red_herring: bool = False


class RoomSpec(BaseModel):
    """Room specification."""

    id: str = Field(default_factory=lambda: f"room-{uuid4().hex[:8]}")
    name: str
    description: str
    important_objects: list[str] = []


class MapSpec(BaseModel):
    """Map specification."""

    id: str = Field(default_factory=lambda: f"map-{uuid4().hex[:8]}")
    location_name: str
    rooms: list[RoomSpec] = []
    description: str


# --- Guía del anfitrión y audio ---
class ClueSolutionEntry(BaseModel):
    """Clue solution entry for host guide."""

    clue_id: str
    how_to_interpret: str


class DetectiveRole(BaseModel):
    """Detective role for the host in Act 2."""

    character_name: str = "Detective"
    public_description: str
    clues_to_reveal: list[ClueSolutionEntry]
    guiding_questions: list[str] = []
    final_solution_script: str


class HostGuide(BaseModel):
    """Complete host guide."""

    spoiler_free_intro: str
    host_act1_role_description: str | None = None
    setup_instructions: list[str] = []
    runtime_tips: list[str] = []
    live_action_murder_event_guide: str | None = None
    act_2_intro_script: str | None = None
    host_act2_detective_role: DetectiveRole | None = None
    solution_timeline: GlobalTimeline | None = None


class AudioScript(BaseModel):
    """Audio script for narration."""

    title: str
    approximate_duration_sec: int
    intro_narration: str


# --- Visuales y empaquetado ---
class ImagePromptSpec(BaseModel):
    """Image generation prompt specification."""

    id: str = Field(default_factory=lambda: f"imgp-{uuid4().hex[:8]}")
    target: Literal["map", "character_portrait", "object", "cover"]
    description: str
    style_tags: list[str] = []
    related_ids: list[str] = []


class VisualStyle(BaseModel):
    """Visual style specification."""

    keywords: list[str] = []
    palette_hints: list[str] = []
    typography_hints: list[str] = []
    image_prompts: list[ImagePromptSpec] = []


class FileDescriptor(BaseModel):
    """File descriptor for packaging."""

    id: str = Field(default_factory=lambda: f"file-{uuid4().hex[:8]}")
    type: Literal["pdf", "markdown", "txt", "audio_script", "image_prompt"]
    name: str
    path: str | None = None


class PackagingInfo(BaseModel):
    """Packaging information for final output."""

    host_package: list[FileDescriptor] = []
    host_guide_file: FileDescriptor | None = None
    audio_script_file: FileDescriptor | None = None
    individual_player_packages: list[FileDescriptor] = []
    index_summary: str = ""


# --- Validación y documento raíz ---
class ValidationIssue(BaseModel):
    """Validation issue found in the game state."""

    id: str = Field(
        default_factory=lambda: f"val-{uuid4().hex[:8]}",
        description="Unique identifier for the validation issue (auto-generated if not provided).",
    )
    type: Literal[
        "timeline_conflict",
        "logic_gap",
        "over_obvious",
        "too_ambiguous",
        "character_unused",
    ] = Field(
        description="Type of validation issue: 'timeline_conflict', 'logic_gap', 'over_obvious', 'too_ambiguous', or 'character_unused'."
    )
    description: str = Field(
        description="Detailed description of the validation issue (e.g., 'Timeline shows character in two places at once')."
    )
    related_ids: list[str] = Field(
        default_factory=list,
        description="List of IDs related to this issue (character IDs, event IDs, etc.). Can be empty.",
    )


class ValidationReport(BaseModel):
    """Validation report."""

    is_consistent: bool = Field(
        description="Whether the game state is logically consistent and playable. Must be true for the game to proceed."
    )
    issues: list[ValidationIssue] = Field(
        default_factory=list,
        description="List of validation issues found. Should be empty if is_consistent is true.",
    )
    suggested_fixes: list[str] = Field(
        default_factory=list,
        description="List of suggested fixes for the issues found. Can be empty if no issues.",
    )


class WorldValidation(BaseModel):
    """World validation report from V2 validator."""

    is_coherent: bool = Field(
        description="Whether the world is historically, geographically, and culturally coherent."
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of coherence issues found. Empty if is_coherent=True.",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested improvements or corrections. Empty if is_coherent=True.",
    )


# --- Clase Raíz del Estado de LangGraph ---


class GameState(BaseModel):
    """
    Central state that flows through the LangGraph.
    Contains the complete game design in all phases.
    """

    meta: MetaInfo
    config: GameConfig

    world: WorldBible | None = None
    crime: CrimeSpec | None = None

    characters: list[CharacterSpec] = []
    relationships: list[RelationshipSpec] = []

    timeline_global: GlobalTimeline | None = None
    killer_selection: KillerSelection | None = None

    personal_timelines: PersonalTimelineByCharacter = Field(default_factory=dict)

    clues: list[ClueSpec] = []
    maps: list[MapSpec] = []

    host_guide: HostGuide | None = None
    audio_script: AudioScript | None = None

    visual_style: VisualStyle | None = None

    validation: ValidationReport | None = None
    world_validation: WorldValidation | None = None

    packaging: PackagingInfo | None = None

    generated_assets: dict[str, str] = Field(default_factory=dict)

    # Retry mechanism for validation loop
    retry_count: int = 0
    max_retries: int = 3
    world_retry_count: int = 0
    max_world_retries: int = 2
