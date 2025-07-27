import base64
import sys
import textwrap
from typing import List
from agents.structs import FrameData
from agents.templates.reasoning_agent import ReasoningAgent
from openai import OpenAI
import os
import base64
import io
import json
import logging
import textwrap
from typing import Any, Dict, List, Literal
from agents.structs import FrameData, GameAction, GameState
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
# logging.basicConfig(level=logging.INFO, stream=sys.stdout)

GOAL_ACHIEVEMENT_CHECK_PROMPT = """Given the goal below and the two images (before and after), determine whether the goal has been achieved.

Goal:
{current_goal}

Answer: Just say Yes or No."""


NEXT_ACTION_GENERATOR_PROMPT = """You are an game play next action generator.

Your goal is to generate a single play action to navigate to achieve

{current_goal}

The game which is designed is based on

- Easy for humans (can pick it up in <1 min of game play)
- Core Knowledge Priors (no language, trivia, cultural symbols)
- Should require no instructions to play
- Should be fun for humans and playable in 5-10 minutes
- Innovative and novel game mechanics encouraged (Hidden state, theory of mind, long term planning, navigating other agents, etc.)


Here are the actions that you can generate. The game is fully controlled by below actions:
W: Move Up
A: Move Left
S: Move Down
D: Move Right
CLICK(x,y): Click on the area by giving x,y space (x: <0, 63>, y: <0, 63>)
Sometimes, some actions has no effect. 

Hints:

{hints}
-----

<previous_actions>
The previous action "{previous_action_text}" had {game_effect_flag} effect in the game.

Previous reason: {previous_action_reason}

Re-evaluate the situation to determine which move is better.
<previous_actions>

------

 What can be next action

Your output should be json with action

Example json output:

{{
"action": "W",
"reason": "I need to move up"
}}

"""

current_goal = "You need to move the \"Orange-Capped Blue Block (6x7)\" to the target \"8x7Grid_BlackHead_BlueEye_WhiteSnout\""
hints = """- The Orange-Capped Blue Block (6x7) is the only movable object until now.
- Click action has no visible effect until now
"""
previous_action_text = "A"
previous_action_reason = "The Orange-Capped Blue Block is currently to the right of its target, the '8x7Grid_BlackHead_BlueEye_WhiteSnout'. Continuing to move left ('A') will bring it closer horizontally to the desired position within the larger grey shape."


logger = logging.getLogger(__name__)

class CustomReasoningAgent(ReasoningAgent):
    MODEL = "gemini-2.5-pro"
    NEXT_ACTION_GENERATOR_MODEL = "gemini-2.5-pro"
    GOAL_ACHIEVEMENT_CHECK_MODEL = "gemini-2.5-pro"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.current_goal = ""
        self.hints = ""
        self.previous_action_text = "RESET"
        self.previous_action_reason = "Game has been reset."

        self.current_goal = current_goal
        self.hints = hints
        self.previous_action_text = previous_action_text
        self.previous_action_reason = previous_action_reason


        # Trial/Real run support
        self.trial_runs: List[List[FrameData]] = []
        self.real_runs: List[List[FrameData]] = []

        self.current_trial_run: List[FrameData] = []
        self.current_real_run: List[FrameData] = []

        self.trial_mode: bool = False
        self._last_trial_mode: bool = self.trial_mode

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MODEL}"

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state == GameState.WIN

    def choose_action(
        self, frames: List[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if self.trial_mode != self._last_trial_mode:
            return self.handle_mode_switch(latest_frame)

        self.append_to_current_run(latest_frame)

        if latest_frame.state in [GameState.NOT_PLAYED]:
            self.save_and_reset_current_run()
            return GameAction.RESET

        is_goal_achieved_flag, goal_achievement_check_output = self.is_goal_achieved(
            previous_frame=frames[-2] if len(frames) > 1 else latest_frame,
            current_frame=latest_frame,
        )
        # Invoke the custom decision logic
        action = self.generate_next_action(latest_frame)
        reasoning = action.reasoning or {}
        reasoning["goal_achievement_check_output"] = goal_achievement_check_output
        reasoning["is_goal_achieved"] = is_goal_achieved_flag
        self.previous_action_text = self.convert_game_action_to_text(action)
        self.previous_action_reason = action.reasoning.get("reason", "No specific reason provided")

        return action

    def handle_mode_switch(self, latest_frame: FrameData) -> GameAction:
        self.append_to_current_run(latest_frame)
        self.save_and_reset_current_run()
        self._last_trial_mode = self.trial_mode
        return GameAction.RESET

    def append_to_current_run(self, latest_frame: FrameData) -> None:
        if self.trial_mode:
            self.current_trial_run.append(latest_frame)
        else:
            self.current_real_run.append(latest_frame)

    def get_current_run(self) -> List[FrameData]:
        if self.trial_mode:
            return self.current_trial_run
        else:
            return self.current_real_run

    def save_and_reset_current_run(self) -> None:
        if self.trial_mode and self.current_trial_run:
            self.trial_runs.append(self.current_trial_run[:])
            self.current_trial_run.clear()
        elif not self.trial_mode and self.current_real_run:
            self.real_runs.append(self.current_real_run[:])
            self.current_real_run.clear()

    def is_goal_achieved(
        self, previous_frame: FrameData, current_frame: FrameData
    ):
        previous_grid = previous_frame.frame[0] if previous_frame.frame else []
        previous_map_image = self.generate_grid_image_with_zone(previous_grid)
        previous_image_b64 = base64.b64encode(previous_map_image).decode()

        current_grid = current_frame.frame[0] if current_frame.frame else []
        current_map_image = self.generate_grid_image_with_zone(current_grid)
        current_image_b64 = base64.b64encode(current_map_image).decode()

        prompt = GOAL_ACHIEVEMENT_CHECK_PROMPT.format(
            current_goal=self.current_goal,
        )
        
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{previous_image_b64}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{current_image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]
        response = self.client.chat.completions.create(
                model=self.GOAL_ACHIEVEMENT_CHECK_MODEL,
                messages=messages,
                # reasoning_effort="low",
                temperature=0.01,
        )
        self.track_tokens(
            response.usage.total_tokens, response.choices[0].message.content
        )
        self.capture_reasoning_from_response(response)

        response_message_text = response.choices[0].message.content
        response_message_text = response_message_text.lower()
        logger.info(f"Response: {response_message_text}")
        if "yes" in response_message_text:
            response_flag = True
        elif "no" in response_message_text:
            response_flag = False
        else:
            logger.error(f"Unexpected response: {response_message_text}")
            response_flag = False

        return response_flag, response.choices[0].message.content

    def is_frames_equal(
        self, previous_frame: FrameData, current_frame: FrameData
    ) -> bool:
        """Check if two frames are equal."""
        if not previous_frame.frame or not current_frame.frame:
            return False
        previous_grid = previous_frame.frame[0]
        current_grid = current_frame.frame[0]
        if len(previous_grid) != len(current_grid):
            return False
        for row_prev, row_curr in zip(previous_grid, current_grid):
            if row_prev != row_curr:
                return False
        return True

    def generate_next_action(
        self,
        latest_frame: FrameData,
    ) -> GameAction:
        """Generate the next action based on the current goal and previous action."""
        current_run = self.get_current_run()
        previous_frame = current_run[-2] if len(current_run) > 1 else latest_frame

        if self.is_frames_equal(previous_frame, latest_frame):
            game_effect_flag = "no"
        else:
            game_effect_flag = "some"
        prompt = NEXT_ACTION_GENERATOR_PROMPT.format(
            current_goal=self.current_goal,
            hints=textwrap.fill(hints, width=80),
            previous_action_text=self.previous_action_text,
            previous_action_reason=self.previous_action_reason,
            game_effect_flag=game_effect_flag,
        )
        latest_grid = latest_frame.frame[0] if latest_frame.frame else []
        latest_map_image = self.generate_grid_image_with_zone(latest_grid)
        latest_image_b64 = base64.b64encode(latest_map_image).decode()

        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{latest_image_b64}",
                            "detail": "high",
                        },
                    }
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.NEXT_ACTION_GENERATOR_MODEL,
            messages=messages,
            # reasoning_effort="low",
        )
        self.track_tokens(
            response.usage.total_tokens, response.choices[0].message.content
        )
        
        response_message_text = response.choices[0].message.content.strip()
        response_message_text = response_message_text.removeprefix("```json")
        response_message_text = response_message_text.removesuffix("```")
        response_message_text = response_message_text.strip()
        logger.info(f"Next action response: {response_message_text}")

        try:
            action_data = json.loads(response_message_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse next action response: {e}")
            action_data = {"action": "UNKNOWN", "reason": "Failed to parse response"}
        return self.convert_action_text_to_game_action(
            action_text=action_data.get("action", ""),
            reason=action_data.get("reason", "")
        )
    
    
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
        img.save("current_frame.png", format="PNG")  # Save for debugging
        buffer.seek(0)  # Reset buffer position
        return buffer.getvalue()

    def convert_action_text_to_game_action(
        self, action_text: str, reason: str = ""
    ) -> GameAction:
        """Convert action text to GameAction object."""
        action_text = action_text.strip().upper()
        if action_text == "W":
            action = GameAction.ACTION1
        elif action_text == "A":
            action = GameAction.ACTION2
        elif action_text == "S":
            action = GameAction.ACTION3
        elif action_text == "D":
            action = GameAction.ACTION4
        elif action_text == "SPACE":
            action = GameAction.ACTION5
        elif action_text == "RESET":
            action = GameAction.RESET
        elif action_text.startswith("CLICK(") and action_text.endswith(")"):
            coords = action_text[6:-1].split(",")
            if len(coords) == 2:
                x, y = map(int, coords)
                action = GameAction.ACTION6
                action.set_data(
                    data={
                        "x": x,
                        "y": y
                    }
                )
        else:
            logger.warning(f"Unknown action text: {action_text}")
            return GameAction.RESET
        action.reasoning = {
            "desired_action": f"{action.value}",
            "reason": reason if reason else "No specific reason provided"
        }
        return action

    def convert_game_action_to_text(
        self, action: GameAction
    ) -> str:
        """Convert GameAction to text representation."""
        if action == GameAction.ACTION1:
            return "W"
        elif action == GameAction.ACTION2:
            return "A"
        elif action == GameAction.ACTION3:
            return "S"
        elif action == GameAction.ACTION4:
            return "D"
        elif action == GameAction.ACTION5:
            return "SPACE"
        elif action == GameAction.RESET:
            return "RESET"
        elif action == GameAction.ACTION6:
            data = action.get_data()
            if data and "x" in data and "y" in data:
                return f"CLICK({data['x']},{data['y']})"
        else:
            logger.warning(f"Unknown GameAction: {action}")
            return "UNKNOWN"
