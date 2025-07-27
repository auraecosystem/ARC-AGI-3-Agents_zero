import base64
import io
import json
import logging
import textwrap
from typing import Any, Dict, List, Literal

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from ..structs import FrameData, GameAction
from .llm_agents import ReasoningLLM

logger = logging.getLogger(__name__)

game_analysis = """This game is a puzzle-platformer where the player navigates a character and manipulates a movable object to reach a target, with actions being highly dependent on context and character state.

Here's a breakdown of the game mechanics:

**1. Player Character & Movement:**
*   The player controls a **blue square with a white shape** (either an "L-shape" or a "straight bar"). This white shape appears to represent the player's "stance" or "mode."
*   **WASD keys** control horizontal and vertical movement across the dark gray path area.
*   The player's **stance (white shape) changes** based on vertical movement: Moving **Down (S)** seems to put the player in the **straight bar stance**, while moving **Up (W)** tends to switch back to the **L-shape stance**. Horizontal movement (A, D) retains the current stance.

**2. Interactive Objects:**
*   **Blue Square (fixed):** A fixed platform that can hold the orange bar.
*   **Orange Bar (movable):** A key interactive object that can exist in three states:
    *   **On the blue square:** Joined with the blue square.
    *   **Off the blue square:** Detached and located to the right of the blue square.
    *   **On the player:** Carried by the player character.
*   **Target (black square with blue dot and white shapes, top-right):** The objective of the level, implying that the player needs to bring something here or arrive in a specific state.

**3. Click Mechanics (Context-Sensitive Actions):**
The outcome of a mouse click (x,y) is highly dependent on:
*   **What is clicked:** The blue square, the orange bar, the player character, or empty space.
*   **The state of the orange bar:** Is it on the blue square, off the blue square, or on the player?
*   **The player's current stance:** L-shape or straight bar.
*   **The player's precise position** relative to the clicked object.

Specific Click Interactions Observed:

*   **Moving Orange Bar OFF the Blue Square:**
    *   Clicking **on the combined blue+orange unit** (e.g., at 0:10, 0:13, 0:26) consistently moves the orange bar to the right, off the blue square.
    *   Clicking **just to the right of the combined blue+orange unit** (e.g., at 0:05) also moves the orange bar to the right, off the blue square.
    *   **Failure Condition:** Clicking on the `blue+orange` unit when the player is in the **L-shape stance and at a specific starting position (26,32)** results in a "fail" (red screen and level reset). This implies a very specific, unforgiving "wrong move" from the starting point.

*   **Moving Orange Bar ONTO the Blue Square:**
    *   Clicking **on the detached orange bar** (e.g., at 0:08) moves it back onto the blue square.
    *   Clicking **just to the right of the empty blue square** when the orange bar is detached and to the right (e.g., at 0:11) also moves it back onto the blue square.

*   **Picking Up Orange Bar (onto player):**
    *   When the orange bar is *off* the blue square (to its right), clicking on the **empty blue square** (e.g., at 0:16) causes the orange bar to transfer onto the player character.
    *   This action causes the player's appearance to change (blue square becomes darker, white shape temporarily disappears). The orange bar then moves with the player.
    *   This action seems to require the player to be in the **L-shape stance and at a specific position (e.g., 40,26)** relative to the empty blue square.

*   **Dropping Orange Bar (from player):**
    *   Clicking **on the player character while it is carrying the orange bar** (e.g., at 0:19) results in a "fail" (red screen and level reset).
    *   This implies that the orange bar must be dropped *elsewhere*, likely on the target or another designated spot, rather than by clicking the player itself.

**4. Game State & Progress Indicators:**
*   **Lives/Attempts (Red Squares, top-right):** These squares change from red to gray upon a "fail" (red screen), indicating a limited number of attempts or "lives."
*   **Progress/Levels (Purple Squares, top-middle):** The row of gray/purple squares at the top suggests overall game progress or a sequence of levels, where purple marks completed levels.

**5. Goal of the Game:**
The primary goal appears to be navigating the player character and successfully manipulating the orange bar to reach the target area (the black square at the top-right). The most plausible win condition involves carrying the orange bar to the target and then depositing it there, or arriving at the target with the bar.

**Innovative and Novel Mechanics:**
*   **Stance-Dependent Actions:** The player's changing white shape (L-shape/bar) acts as a crucial "mode" that dictates the outcome of clicks, adding a layer of strategic decision-making beyond simple movement.
*   **Precise Positional Requirements:** Certain actions, particularly the critical "pick up" of the orange bar, are only triggered when the player is in a specific stance *and* at a precise location relative to the interactive object. This demands meticulous planning and execution.
*   **Implicit Drop Mechanic:** The explicit "fail" for clicking on the player while carrying the bar creates a puzzle in itself, forcing the player to deduce the correct way to "unload" the bar by clicking on a specific *external* location (likely the target).
*   **Hidden State and Context:** The game doesn't provide explicit instructions, requiring the player to observe, hypothesize, and experiment to understand the complex interplay of character stance, object states, and spatial relationships that govern actions. This fosters "theory of mind" as the player tries to understand the game's internal logic."""
print("length of game analysis:", len(game_analysis))

class ReasoningActionResponse(BaseModel):
    """Action response structure for reasoning agent."""

    name: Literal["ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"] = Field(
        description="The action to take."
    )
    reason: str = Field(
        description="Detailed reasoning for choosing this action",
        min_length=10,
        max_length=2000,
    )
    short_description: str = Field(
        description="Brief description of the action", min_length=5, max_length=500
    )
    hypothesis: str = Field(
        description="Current hypothesis about game mechanics",
        min_length=10,
        max_length=2000,
    )
    aggregated_findings: str = Field(
        description="Summary of discoveries and learnings so far",
        min_length=10,
        max_length=6000,
    )


class ReasoningAgent(ReasoningLLM):
    """A reasoning agent that tracks screen history and builds hypotheses about game rules."""

    MAX_ACTIONS = 400
    DO_OBSERVATION = True
    MODEL = "o4-mini"
    MESSAGE_LIMIT = 5
    REASONING_EFFORT = "high"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.history: List[ReasoningActionResponse] = []
        self.screen_history: List[bytes] = []
        self.max_screen_history = 10  # Limit screen history to prevent memory leak
        self.client = OpenAI()

    def clear_history(self) -> None:
        """Clear all history when transitioning between levels."""
        self.history = []
        self.screen_history = []

    def generate_grid_image_with_zone(
        self, grid: List[List[int]], cell_size: int = 40, zone_size: int = 20
    ) -> bytes:
        """Generate PIL image of the grid with colored cells and zone coordinates."""
        if not grid or not grid[0]:
            # Create empty image
            img = Image.new("RGB", (200, 200), color="black")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()

        height = len(grid)
        width = len(grid[0])

        # Create image
        img = Image.new("RGB", (width * cell_size, height * cell_size), color="white")
        draw = ImageDraw.Draw(img)

        # Color mapping for grid cells
        key_colors = {
            0: "#FFFFFF",
            1: "#CCCCCC",
            2: "#999999",
            3: "#666666",
            4: "#333333",
            5: "#000000",
            6: "#E53AA3",
            7: "#FF7BCC",
            8: "#F93C31",
            9: "#1E93FF",
            10: "#88D8F1",
            11: "#FFDC00",
            12: "#FF851B",
            13: "#921231",
            14: "#4FCC30",
            15: "#A356D6"
        }

        # Draw grid cells
        for y in range(height):
            for x in range(width):
                color = key_colors.get(grid[y][x], "#888888")  # default: floor

                # Draw cell
                draw.rectangle(
                    [
                        x * cell_size,
                        y * cell_size,
                        (x + 1) * cell_size,
                        (y + 1) * cell_size,
                    ],
                    fill=color,
                    outline="#000000",
                    width=1,
                )

        # Draw zone coordinates and borders
        for y in range(0, height, zone_size):
            for x in range(0, width, zone_size):
                # Draw zone coordinate label
                try:
                    font = ImageFont.load_default()
                    zone_text = f"({x},{y})"
                    draw.text(
                        (x * cell_size + 2, y * cell_size + 2),
                        zone_text,
                        fill="#FFFFFF",
                        font=font,
                    )
                except (ImportError, OSError) as e:
                    logger.debug(f"Could not load font for zone labels: {e}")
                except Exception as e:
                    logger.error(f"Failed to draw zone label at ({x},{y}): {e}")

                # Draw zone boundary
                zone_width = min(zone_size, width - x) * cell_size
                zone_height = min(zone_size, height - y) * cell_size
                draw.rectangle(
                    [
                        x * cell_size,
                        y * cell_size,
                        x * cell_size + zone_width,
                        y * cell_size + zone_height,
                    ],
                    fill=None,
                    outline="#FFD700",  # gold border for zone
                    width=2,
                )

        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    def build_functions(self) -> list[dict[str, Any]]:
        """Build JSON function description of game actions for LLM."""
        schema = ReasoningActionResponse.model_json_schema()
        # The 'name' property is the action to be taken, so we can remove it from the parameters.
        schema["properties"].pop("name", None)
        if "required" in schema:
            schema["required"].remove("name")

        functions: list[dict[str, Any]] = [
            {
                "name": action.name,
                "description": f"Take action {action.name}",
                "parameters": schema,
            }
            for action in [
                GameAction.ACTION1,
                GameAction.ACTION2,
                GameAction.ACTION3,
                GameAction.ACTION4,
                GameAction.RESET,
            ]
        ]
        return functions

    def build_tools(self) -> list[dict[str, Any]]:
        """Support models that expect tool_call format."""
        functions = self.build_functions()
        tools: list[dict[str, Any]] = []
        for f in functions:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f["name"],
                        "description": f["description"],
                        "parameters": f.get("parameters", {}),
                    },
                }
            )
        return tools

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Build the user prompt for hypothesis-driven exploration."""
        return textwrap.dedent(
            f"""
You are playing a video game.

{game_analysis}
Your ultimate goal is to understand the rules of the game and explain them to your colleagues.

The game is complex, and may look like an IQ test.

You need to determine how the game works on your own.

To do so, we will provide you with a view of the game corresponding to the bird-eye view of the game, along with the raw grid data.

You can do 5 actions:
- RESET (used to start a new game or level)
- ACTION1 (MOVE_UP)
- ACTION2 (MOVE_DOWN)
- ACTION3 (MOVE_LEFT)
- ACTION4 (MOVE_RIGHT)

You can do one action at once.

Every time an action is performed we will provide you with the previous screen and the current screen.

Determine the game rules based on how the game reacted to the previous action (based on the previous screen and the current screen).

Your goal:

1. Experiment the game to determine how it works based on the screens and your actions.
2. Analyse the impact of your actions by comparing the screens.

How to proceed:
1. Define an hypothesis and an action to validate it.
2. Once confirmed, store the findings. Summarize and aggregate them so that your colleagues can understand the game based on your learning.
3. Make sure to understand clearly the game rules, energy, walls, doors, keys, etc.

Hint:
- The game is a 2D platformer.
- The player can move up, down, left and right.
- The player has a blue body and a yellow head.
- There are walls in black.
- The door has a pink border and a shape inside.
        """
        )

    def call_llm_with_structured_output(
        self, messages: List[Dict[str, Any]]
    ) -> ReasoningActionResponse:
        """Call LLM with structured output parsing for reasoning agent."""
        try:
            tools = self.build_tools()

            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                tools=tools,
                tool_choice="required",
            )

            self.track_tokens(
                response.usage.total_tokens, response.choices[0].message.content
            )
            self.capture_reasoning_from_response(response)

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            if tool_calls:
                tool_call = tool_calls[0]
                function_args = json.loads(tool_call.function.arguments)
                function_args["name"] = tool_call.function.name
                return ReasoningActionResponse(**function_args)

            raise ValueError("LLM did not return a tool call.")

        except Exception as e:
            logger.error(f"LLM structured call failed: {e}")
            raise e

    def define_next_action(self, latest_frame: FrameData) -> ReasoningActionResponse:
        """Define next action for the reasoning agent."""
        # Generate map image
        current_grid = latest_frame.frame[0] if latest_frame.frame else []
        map_image = self.generate_grid_image_with_zone(current_grid)

        # Build messages
        system_prompt = self.build_user_prompt(latest_frame)

        # Get latest action from history
        latest_action = self.history[-1] if self.history else None

        # Build user message with images
        user_message_content: List[Dict[str, Any]] = []

        # Use the last screen from history as the 'previous_screen'
        previous_screen = self.screen_history[-1] if self.screen_history else None

        if previous_screen:
            user_message_content.extend(
                [
                    {"type": "text", "text": "Previous screen:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64.b64encode(previous_screen).decode()}",
                            "detail": "high",
                        },
                    },
                ]
            )

        raw_grid_text = self.pretty_print_3d(latest_frame.frame)
        user_message_text = f"Your previous action was: {json.dumps(latest_action.model_dump() if latest_action else None, indent=2)}\n\nWhat should you do next?"

        current_image_b64 = base64.b64encode(map_image).decode()
        user_message_content.extend(
            [
                {"type": "text", "text": user_message_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{current_image_b64}",
                        "detail": "high",
                    },
                },
            ]
        )

        # Build messages
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message_content},
        ]

        # Call LLM with structured output
        result = self.call_llm_with_structured_output(messages)

        # Store current screen for next iteration (after using it)
        self.screen_history.append(map_image)
        if len(self.screen_history) > self.max_screen_history:
            self.screen_history.pop(0)

        return result

    def choose_action(
        self, frames: List[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose action using parent class tool calling with reasoning enhancement."""
        if latest_frame.full_reset:
            self.clear_history()
            return GameAction.RESET

        if not self.history:  # First action must be RESET
            action = GameAction.RESET
            initial_response = ReasoningActionResponse(
                name="RESET",
                reason="Initial action to start the game and observe the environment.",
                short_description="Start game",
                hypothesis="The game requires a RESET to begin.",
                aggregated_findings="No findings yet.",
            )
            self.history.append(initial_response)
            return action

        # Define the next action based on reasoning
        action_response = self.define_next_action(latest_frame)
        self.history.append(action_response)

        # Map the reasoning action name to a GameAction
        action = GameAction.from_name(action_response.name)

        # Create and attach reasoning metadata
        reasoning_meta = {
            "model": self.MODEL,
            "reasoning_effort": self.REASONING_EFFORT,
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "agent_type": "reasoning_agent",
            "hypothesis": action_response.hypothesis,
            "aggregated_findings": action_response.aggregated_findings,
            "response_preview": action_response.reason[:200] + "..."
            if len(action_response.reason) > 200
            else action_response.reason,
            "action_chosen": action.name,
            "game_context": {
                "score": latest_frame.score,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
        }
        action.reasoning = reasoning_meta

        return action