

from typing import List, Union
from PIL import Image, ImageDraw, ImageFont
import io
import logging
import random
import time
from typing import Any
import json
from enum import Enum

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState

from google.genai import types
from google import genai

# Set up logger
logger = logging.getLogger(__name__)

HYPOTHESIS_ACTION_NAVIGATOR_PROMPT = """You are an agent playing a dynamic game. Your objective is to
achieve the below hypothesis by taking actions based on the game grid.

<hypothesis>
{hypothesis}
</hypothesis>

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

AVAILABLE ACTIONS:
- Move Up (W)
- Move Left (A)
- Move Down (S)
- Move Right (D)
- Click on the area by giving x,y space CLICK(x,y)
- hypothesis got achieved (ACHIEVED)
- Wrong hypothesis (WRONG_HYPOTHESIS)

Call exactly one action

Respond with a JSON:
{{
  "reason": "Your reason for the action (max 50 words)",
  "action": "The action you want to take"
}}
"""

class HypothesisAction(Enum):
    MOVE_UP = "W"
    MOVE_LEFT = "A"
    MOVE_DOWN = "S"
    MOVE_RIGHT = "D"
    CLICK = "CLICK"
    ACHIEVED = "ACHIEVED"
    WRONG_HYPOTHESIS = "WRONG_HYPOTHESIS"

hypothesis = """Hypothesis 1: The primary objective is to push the orange/blue block into the black goal square.
Because Hypothesis 1 is the only one that describes exactly the action that produces a *persistent progress change*—the purple “lit” square on the progress bar—without triggering a reset. In every other hypothesis you’re either testing death/reset conditions (blocks going off‑screen or into walls), temporary obstacles (the white T’s), respawns, or UI counters (lives and attempts).

Hypothesis 1 alone matches what we intuitively think of as the “win” event in the footage:

* **Successful goal‑entry**: The orange/blue block ends up fully inside the black square.
* **Positive feedback**: A purple square lights up on the top bar—unlike failures, there’s no red flash or reset.
* **No reset**: The level continues from that new state, proving it’s not just a mechanic or death test but *progress toward victory.*

That combination of “block in goal → purple progress marker → no reset” is precisely the signature of a completed objective, i.e. the win condition.
"""

class Random(Agent):
    """An agent that always selects actions at random, supports trial and real runs with grouping."""

    MAX_ACTIONS = 300

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)

        # Grouped runs
        self.trial_runs: list[list[FrameData]] = []
        self.real_runs: list[list[FrameData]] = []

        # Tracking current run
        self.current_trial_run: list[FrameData] = []
        self.current_real_run: list[FrameData] = []

        # Mode indicator (set externally)
        self.trial_mode: bool = False
        self._last_trial_mode: bool = self.trial_mode  # to detect changes
        
        self.client = genai.Client()
        self.hypothesis = hypothesis

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return any(
            [
                latest_frame.state is GameState.WIN,
                # uncomment to only let the agent play one time
                # latest_frame.state is GameState.GAME_OVER,
            ]
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""

        # Handle trial_mode change
        if self.trial_mode != self._last_trial_mode:
            return self.handle_mode_switch(latest_frame)

        # Append current frame to current run
        self.append_to_current_run(latest_frame)

        # If game is over or not started, reset and archive run
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.save_and_reset_current_run()
            return GameAction.RESET

        hypothesis_action_response = self.select_action_for_hypothesis(
            hypothesis=self.hypothesis,
            latest_frame=latest_frame,
        )
        action = self.hypothesis_action_to_game_action(hypothesis_action_response["action"])
        action.reasoning = {
            "desired_action": f"{action.value}",
            "my_reason": hypothesis_action_response["reason"],
        }
        # Choose a random action that isn't RESET
        # action = random.choice([a for a in GameAction if a is not GameAction.RESET])

        # Annotate action
        # if action.is_simple():
        #     action.reasoning = f"RNG told me to pick {action.value}"
        # elif action.is_complex():
        #     action.set_data(
        #         {
        #             "x": random.randint(0, 63),
        #             "y": random.randint(0, 63),
        #         }
        #     )
        #     action.reasoning = {
        #         "desired_action": f"{action.value}",
        #         "my_reason": "RNG said so!",
        #     }

        return action

    def handle_mode_switch(self, latest_frame: FrameData) -> GameAction:
        """Handles behavior when trial_mode has been switched externally."""
        # Save the last frame in the current run before switching
        self.append_to_current_run(latest_frame)
        self.save_and_reset_current_run()

        # Update internal mode tracker
        self._last_trial_mode = self.trial_mode

        # Force a reset to cleanly start the new run mode
        return GameAction.RESET

    def append_to_current_run(self, latest_frame: FrameData) -> None:
        """Appends the frame to the correct run buffer based on trial_mode."""
        if self.trial_mode:
            self.current_trial_run.append(latest_frame)
        else:
            self.current_real_run.append(latest_frame)

    def save_and_reset_current_run(self) -> None:
        """Saves the current run to the appropriate history and clears the buffer."""
        if self.trial_mode and self.current_trial_run:
            self.trial_runs.append(self.current_trial_run[:])
            self.current_trial_run.clear()
        elif not self.trial_mode and self.current_real_run:
            self.real_runs.append(self.current_real_run[:])
            self.current_real_run.clear()

    def select_action_for_hypothesis(
            self, hypothesis: str, latest_frame: FrameData
    ) -> dict[str, Union[str, HypothesisAction]]:
        current_grid = latest_frame.frame[0] if latest_frame.frame else []
        grid_image = self.generate_grid_image_with_zone(
            grid=current_grid,
        )
        response = self.client.models.generate_content(
            model='models/gemini-2.5-flash',
            contents=types.Content(
                parts=[
                    types.Part(text=HYPOTHESIS_ACTION_NAVIGATOR_PROMPT.format(hypothesis=hypothesis)),
                    types.Part.from_bytes(
                        data=grid_image,
                        mime_type='image/png'
                    )
                ]
            )
        )
        
        # Parse JSON response
        try:
            response_text = response.text
            response_text = response_text.removeprefix("```json").removesuffix("```").strip()
            response_text = response_text.strip()
            parsed = json.loads(response_text)
            reason = parsed.get("reason", "No reason provided.")
            action_text = parsed["action"].strip()

            # Determine the corresponding action
            if action_text == "W":
                action = HypothesisAction.MOVE_UP
            elif action_text == "A":
                action = HypothesisAction.MOVE_LEFT
            elif action_text == "S":
                action = HypothesisAction.MOVE_DOWN
            elif action_text == "D":
                action = HypothesisAction.MOVE_RIGHT
            elif action_text.startswith("CLICK"):
                coords = action_text[6:-1].split(',')
                action = (HypothesisAction.CLICK, (int(coords[0]), int(coords[1])))
            elif action_text == "ACHIEVED":
                action = HypothesisAction.ACHIEVED
            elif action_text == "WRONG_HYPOTHESIS":
                action = HypothesisAction.WRONG_HYPOTHESIS
            else:
                raise ValueError(f"Unknown action text: {action_text}")

            return {
                "reason": reason,
                "action": action
            }

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # raise ValueError(f"Failed to parse model response: {response_text}") from e
            return {
                "reason": "Failed to parse model response",
                "action": HypothesisAction.MOVE_UP
            }
    
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

    def hypothesis_action_to_game_action(
        self, hypothesis_action: HypothesisAction
    ) -> GameAction:
        """Convert a HypothesisAction to a GameAction."""
        if hypothesis_action == HypothesisAction.MOVE_UP:
            return GameAction.ACTION1
        elif hypothesis_action == HypothesisAction.MOVE_LEFT:
            return GameAction.ACTION2
        elif hypothesis_action == HypothesisAction.MOVE_DOWN:
            return GameAction.ACTION3
        elif hypothesis_action == HypothesisAction.MOVE_RIGHT:
            return GameAction.ACTION4
        elif isinstance(hypothesis_action, tuple) and hypothesis_action[0] == HypothesisAction.CLICK:
            x, y = hypothesis_action[1]
            action = GameAction.ACTION6
            action.set_data({"x": x, "y": y})
            return action
        elif hypothesis_action == HypothesisAction.ACHIEVED:
            return GameAction.ACTION0
        elif hypothesis_action == HypothesisAction.WRONG_HYPOTHESIS:
            return GameAction.ACTION0
        else:
            raise ValueError(f"Unknown HypothesisAction: {hypothesis_action}"
    )