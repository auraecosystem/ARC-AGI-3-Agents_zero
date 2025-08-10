import base64
import io
import json
import logging
import os
import random
import re
import textwrap
from collections import deque
from itertools import cycle
from typing import Dict, List

import cv2
import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import APIError
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agents.structs import FrameData, GameAction, GameState
from agents.templates.reasoning_agent import ReasoningLLM

logger = logging.getLogger(__name__)

logical_analysis_actions_summary = """This is a summary of the logical analysis actions taken in the game.

- out of 15 W actions, 5 had no some effect on the game.
- out of 15 A actions, 5 had no some effect on the game.
- out of 10 S actions, 6 had no some effect on the game.
- out of 10 D actions, 5 had no some effect on the game.
- out of 15 CLICK actions, 10 had no some effect on the game.
"""

HINTS_GENERATOR_PROMPT = """This video is a random actions (WASD and click) moves taken on unkown game.

The game is designed based on below Constraints
- Easy for humans (can pick it up in <1 min of game play)
- Core Knowledge Priors (no language, trivia, cultural symbols)
- Should require no instructions to play
- Should be fun for humans and playable in 5-10 minutes
- Innovative and novel game mechanics encouraged (Hidden state, theory of mind, long term planning, navigating other agents, etc.)

Here are the actions that your player can take
W: Move Up
A: Move Left
S: Move Down
D: Move Right
CLICK(x,y): Click on the area by giving x,y space (x: <0, 63>, y: <0, 63>)
Sometimes, some actions has no effect. 

Random Analysis Actions summary
This is a summary of the random analysis actions taken in the game.
{logical_analysis_actions_summary}

- Write hints clearly with element names clearly
- Keep the hints targeting with "focusing element names" and "final goal". Also remove unsure sentences from the hints."
- element name must be clearly mentioned with approximate size, colors and shape. [Example: (16x15grid)Red_Square_Block]

Give short hints to humans to play. 
"""

ELEMENTS_TITLE_PROMPT = """This video is a random actions (WASD and click) moves taken on unkown game.

Generate a detailed, understandable title of the elements in the game.

[Example: (16x15grid)Red_Bottom_White_Square_Block, (16x4grid)Yellow_Rectangle_Block]

Elements:"""


TOP_HYPOTHESIS_RETRIEVER_PROMPT = """Pick one hypothesis which has key impact in win as soon as possible in less moves

{multiple_hypothesis_text}

Hypothesis: """

RANDOM_HYPOTHESIS_ANALYSIS_PROMPT = """You are playing a 2D grid game. This video is a random actions (WASD and click) moves taken on unkown game by you.

The game is designed based on below Constraints
- Easy for humans (can pick it up in <1 min of game play)
- Core Knowledge Priors (no language, trivia, cultural symbols)
- Should require no instructions to play
- Should be fun for humans and playable in 5-10 minutes
- Innovative and novel game mechanics encouraged (Hidden state, theory of mind, long term planning, navigating other agents, etc.)

Here are the actions that your player can take
W: Move Up
A: Move Left
S: Move Down
D: Move Right
CLICK(x,y): Click on the area by giving x,y space (x: <0, 63>, y: <0, 63>)
Sometimes, some actions has no effect. 

Random Analysis Actions summary
This is a summary of the random analysis actions taken in the game.
{logical_analysis_actions_summary}

Sometimes, some actions has no effect. 

- Write hypothesis clearly with element names clearly
- Keep the hypothesis targeting with "focusing element names" and "final goal". Also remove unsure sentences from the hypothesis."
- element name must be clearly mentioned with approximate size, colors and shape. [Example: (16x15grid)Red_Square_Block]

Give 10 hypothesise to explore the game and understand its mechanics, objectives, and challenges. 
"""

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
GEMINI_RETRY_ATTEMPTS = 5

class GameContext(BaseModel):
    goal: str = ""
    goal_actions_count : int = 0
    max_goal_actions_limit: int
    elements_text: str = "No elements description available yet."
    hints: str = "No hints available yet."
    multiple_hypothesis_text: str = "No hypotheses available yet."
    logical_analysis_actions_summary: str = "No logical analysis actions summary available yet."

class PlayZeroAgent(ReasoningLLM):
    MODEL = "gemini-2.5-flash"
    MAX_ACTIONS = 750
    NEXT_ACTION_GENERATOR_MODEL = "gemini-2.5-flash"
    GOAL_ACHIEVEMENT_CHECK_MODEL = "gemini-2.5-flash"
    RANDOM_ANALYSIS_MODEL = "gemini-2.5-pro"
    HINT_GENERATOR_MODEL = "gemini-2.5-flash"
    ELEMENTS_TITLE_GENERATOR_MODEL = "gemini-2.5-flash" 
    TOP_HYPOTHESIS_GENERATOR_MODEL = "gemini-2.5-flash"
    RANDOM_ACTION_MAX_LIMIT = 30
    RANDOM_PROB_ENABLE_ACTION_MIN_LIMIT = 100
    RANDOM_ANALYSIS_FPS = 1
    RANDOM_ANALYSIS_SKIP_REPEATED_FRAMES_FLAG = True

    # in random choice, trigger one action based on probablity effect on game board in inerval of 20 moves,
    # use equal probability initially
    # then increase game effect probability based on the game effect

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Collect all available shared API keys
        api_keys = list(filter(None, [
            os.getenv("GEMINI_API_KEY"),
            os.getenv("GEMINI_API_KEY_1"),
            os.getenv("GEMINI_API_KEY_2"),
        ]))

        if not api_keys:
            raise ValueError("No valid GEMINI_API_KEYs found in environment variables.")

        # Initialize OpenAI clients
        self._openai_clients = [
            OpenAI(api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            for key in api_keys
        ]

        # Initialize Gemini clients
        self._gemini_clients = [
            genai.Client(api_key=key)
            for key in api_keys
        ]

        # Create round-robin iterators
        self._openai_cycle = cycle(self._openai_clients)
        self._gemini_cycle = cycle(self._gemini_clients)

        self.previous_action: GameAction = GameAction.RESET
        self.previous_action.reasoning = {
            "desired_action": f"{self.previous_action.value}",
            "reason": "Game has been reset."
        }
        self.game_context = GameContext(
            max_goal_actions_limit=self.RANDOM_ACTION_MAX_LIMIT,
        )
        self.random_play_flag: bool = True

    @property
    def previous_action_text(self) -> str:
        """Get the text representation of the previous action."""
        return self.convert_game_action_to_text(self.previous_action)

    @property
    def previous_action_reason(self) -> str:
        """Get the reasoning of the previous action."""
        return self.previous_action.reasoning.get("reason", "No specific reason provided")

    @property
    def client(self):
        """Next OpenAI client (round-robin)."""
        return next(self._openai_cycle)

    @property
    def gemini_client(self):
        """Next Gemini client (round-robin)."""
        return next(self._gemini_cycle)

    @retry(
        stop=stop_after_attempt(GEMINI_RETRY_ATTEMPTS),  # Retry up to 5 times
        wait=wait_exponential(multiplier=1, min=1, max=16),  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
        retry=retry_if_exception_type(APIError),
        reraise=True
    )
    def generate_content_using_gemini(
        self, model: str, contents: types.Content
    ) -> types.GenerateContentResponse:
        """Generate content using Gemini with retry on server error."""
        return self.gemini_client.models.generate_content(
            model=model,
            contents=contents
        )

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MODEL}"

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state == GameState.WIN

    def choose_random_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        action = random.choice([a for a in GameAction if a is not GameAction.RESET])

        # Store past positions as an attribute on the object
        if not hasattr(self, "_used_positions"):
            self._used_positions = set()
        if not hasattr(self, "_useful_positions"):
            self._useful_positions = set()
        if not hasattr(self, "original_total_random_actions_count"):
            self.original_total_random_actions_count = 0

        if self.original_total_random_actions_count > self.RANDOM_PROB_ENABLE_ACTION_MIN_LIMIT:
            total_action_counts, no_effect_actions = self.count_actions(frames)
            probabilities_for_actions = self.calculate_effect_probabilities(total_action_counts, no_effect_actions)
            # choose top 1
            top_action = max(probabilities_for_actions, key=probabilities_for_actions.get)
            logger.info(f"Top action with effect: {top_action} (Probability: {probabilities_for_actions[top_action]})")
            action = self.convert_action_text_to_game_action(top_action)

        if self.previous_action.is_complex():
            previous_xy_position = (self.previous_action.action_data.x, self.previous_action.action_data.y)
        else:
            previous_xy_position = None
        # Check if frame changed — if so, clear history
        
        if self.is_frames_equal(frames[-2] if len(frames) > 1 else latest_frame, latest_frame):
            if previous_xy_position and previous_xy_position in self._useful_positions:
                self._useful_positions.remove(previous_xy_position)
        else:
            self._used_positions.clear()
            if previous_xy_position:
                self._useful_positions.add(previous_xy_position)

        if action.is_complex():
            xy_positions = set(self.find_shape_positions(latest_frame.frame[0]))

            # Filter out positions we've already used for this unchanged frame
            available_positions = {
                pos for pos in xy_positions if pos not in self._used_positions
            }

            # If all positions were used, reset available list
            if not available_positions:
                available_positions = xy_positions
                self._used_positions.clear()

            xy_position = random.choice(list(available_positions) + list(self._useful_positions))
            self._used_positions.add(xy_position)

            action.set_data(
                {
                    "x": xy_position[0],
                    "y": xy_position[1],
                }
            )

        action.reasoning = {
            "desired_action": f"{action.value}",
            "reason": "Randomly chosen action without repeating past positions for unchanged frames",
            "useful_positions": list(self._useful_positions),
            "used_positions": list(self._used_positions),
        }
        self.original_total_random_actions_count += 1
        return action
    
    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose the next action based on the current game state and frames."""
        action = self._choose_action(frames, latest_frame)
        action.reasoning = action.reasoning or {}
        game_context_dict = self.game_context.model_dump()
        action.reasoning.update(game_context_dict)
        action.reasoning["previous_action_text"] =self.convert_game_action_to_text(self.previous_action)
        action.reasoning["previous_action_reason"] = self.previous_action.reasoning.get("reason", "No specific reason provided")
        self.previous_action = action
        return action

    def _choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        """Internal method to choose the next action based on the current game state and frames."""
        if latest_frame.state in [GameState.NOT_PLAYED]:
            action = GameAction.RESET
            action.reasoning = {
                "desired_action": f"{action.value}",
                "reason": "Game has not been played yet, resetting."
            }
            return action
            
        previous_frame = frames[-2] if len(frames) > 1 else latest_frame

        is_goal_achieved_flag = False
        goal_achievement_check_output = "no"
        if not self.random_play_flag:
            is_goal_achieved_flag, goal_achievement_check_output = self.is_goal_achieved(
                previous_frame=previous_frame,
                current_frame=latest_frame,
            )        
        if self.is_max_goal_actions_limit_reached() or is_goal_achieved_flag:
            logger.info("Maximum goal actions limit reached or goal achieved, performing game analysis.")
            if self.random_play_flag:
                self.random_play_flag = False
            self.game_context = self.do_game_analysis(frames, latest_frame)


        if self.random_play_flag:
            action = self.choose_random_action(frames, latest_frame)
        else:
            action = self.generate_next_action(
                previous_frame=previous_frame,
                latest_frame=latest_frame,
            )
            reasoning = action.reasoning or {}
            reasoning["goal_achievement_check_output"] = goal_achievement_check_output
            reasoning["is_goal_achieved_flag"] = is_goal_achieved_flag

        is_frame_repeated = self.is_frames_equal(frames[-2] if len(frames) > 1 else latest_frame, latest_frame)
        action.reasoning["game_effect_flag"] = "no" if is_frame_repeated else "some"
        self.game_context.goal_actions_count += 1
        if self.RANDOM_ANALYSIS_SKIP_REPEATED_FRAMES_FLAG and is_frame_repeated:
            logger.info("Skipping repeated frame, goal action count not increased.")
            self.game_context.goal_actions_count -= 1

        return action

    def do_game_analysis(
        self, frames: List[FrameData], latest_frame: FrameData
    ) -> GameContext:
        """Perform game analysis and return a summary."""
        logger.info("Performing game analysis on frames...")
        # generate unique file name for video
        video_file_name = f"game_analysis_{frames[-1].game_id}.mp4"
        video_file_path = os.path.join("recordings", video_file_name)
        self.generate_video_from_grids(
            frames=frames,
            video_output_path=video_file_path,
            fps=self.RANDOM_ANALYSIS_FPS,
            skip_repeated_frames=self.RANDOM_ANALYSIS_SKIP_REPEATED_FRAMES_FLAG,
        )
        logger.info(f"Video saved to {video_file_path}")
        logical_analysis_actions_summary = self.generate_logical_analysis_summary(frames)

        multiple_hypothesis_text = self.generate_multiple_random_hypotheses_from_video(
            video_file_path, logical_analysis_actions_summary
        )

        top_hypothesis = self.generate_top_hypothesis(
            multiple_hypothesis_text=multiple_hypothesis_text,
            logical_analysis_actions_summary=logical_analysis_actions_summary
        )
        hints = self.generate_hints_from_video(
            video_file_path,
            logical_analysis_actions_summary,
        )

        return GameContext(
            goal=top_hypothesis,
            goal_actions_count=0,
            max_goal_actions_limit=10,
            elements_text="No elements description available yet.",
            hints=hints,
            multiple_hypothesis_text=multiple_hypothesis_text,
            logical_analysis_actions_summary=logical_analysis_actions_summary,
        )
    
    def generate_top_hypothesis(self, multiple_hypothesis_text: str, logical_analysis_actions_summary: str) -> str:
        """Retrieve the top hypothesis from the random analysis response."""
        prompt = TOP_HYPOTHESIS_RETRIEVER_PROMPT.format(
            multiple_hypothesis_text=multiple_hypothesis_text,
            # logical_analysis_actions_summary=logical_analysis_actions_summary,
        )
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        response = self.client.chat.completions.create(
            model=self.TOP_HYPOTHESIS_GENERATOR_MODEL,
            messages=messages,
            # reasoning_effort="low",
        )
        top_hypothesis = response.choices[0].message.content.strip()
        
        self.track_tokens(
            response.usage.total_tokens, response.choices[0].message.content
        )
        self.capture_reasoning_from_response(response)

        logger.info(f"Top hypothesis retrieved: {top_hypothesis}")
        return top_hypothesis

    def generate_multiple_random_hypotheses_from_video(
        self,
        video_file_path: str,
        logical_analysis_actions_summary: str,
    ) -> str:
        """Generate random hypothesis analysis from a video file."""
        logger.info(f"Generating random hypothesis analysis from video: {video_file_path}")
        if not os.path.exists(video_file_path):
            logger.error(f"Video file does not exist: {video_file_path}")
            return "Analysis not yet done."
        video_bytes = open(video_file_path, 'rb').read()

        random_explorer_agent_response = self.generate_content_using_gemini(
            model=self.RANDOM_ANALYSIS_MODEL,
            contents=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                    ),
                    types.Part(
                            text=RANDOM_HYPOTHESIS_ANALYSIS_PROMPT.format(
                            logical_analysis_actions_summary=logical_analysis_actions_summary,
                        )
                    )
                ]
            )
        )
        
        all_random_hypothesis_text = random_explorer_agent_response.text.strip()
        self.track_tokens(
            random_explorer_agent_response.usage_metadata.total_token_count, random_explorer_agent_response.text
        )
        logger.info(f"Random analysis completed: {all_random_hypothesis_text}")
        return all_random_hypothesis_text

    def generate_hints_from_video(
        self, video_file_path: str, logical_analysis_actions_summary: str
    ) -> str:
        """Generate hints from a video file."""
        logger.info(f"Generating hints from video: {video_file_path}")
        if not os.path.exists(video_file_path):
            logger.error(f"Video file does not exist: {video_file_path}")
            return "Analysis not yet done."
        video_bytes = open(video_file_path, 'rb').read()

        hint_chat_response = self.generate_content_using_gemini(
            model=self.HINT_GENERATOR_MODEL,
            contents=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                    ),
                    types.Part(text=HINTS_GENERATOR_PROMPT.format(
                        logical_analysis_actions_summary=logical_analysis_actions_summary,
                    ))
                ]
            )
        )
        
        hints = hint_chat_response.text.strip()
        self.track_tokens(
            hint_chat_response.usage_metadata.total_token_count, hint_chat_response.text
        )
        logger.info(f"Hints generated: {hints}")
        return hints


    def generate_element_titles_from_video(
        self, video_file_path: str
    ) -> str:
        """Generate element titles from a video file."""
        logger.info(f"Generating element titles from video: {video_file_path}")
        if not os.path.exists(video_file_path):
            logger.error(f"Video file does not exist: {video_file_path}")
            return "Analysis not yet done."
        video_bytes = open(video_file_path, 'rb').read()

        elements_text_chat_response = self.generate_content_using_gemini(
            model=self.ELEMENTS_TITLE_GENERATOR_MODEL,
            contents=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                    ),
                    types.Part(text=ELEMENTS_TITLE_PROMPT)
                ]
            )
        )

        elements_text = elements_text_chat_response.text.strip()
        self.track_tokens(
            elements_text_chat_response.usage_metadata.total_token_count, elements_text_chat_response.text
        )
        logger.info(f"Elements title generated: {elements_text}")
        return elements_text

    def generate_next_action(
        self,
        previous_frame: FrameData,
        latest_frame: FrameData,
    ) -> GameAction:
        """Generate the next action based on the current goal and previous action."""

        if self.is_frames_equal(previous_frame, latest_frame):
            game_effect_flag = "no"
        else:
            game_effect_flag = "some"
        prompt = NEXT_ACTION_GENERATOR_PROMPT.format(
            current_goal=self.game_context.goal,
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
        response_message_text = response.choices[0].message.content.strip()
        self.track_tokens(
            response.usage.total_tokens, response.choices[0].message.content
        )
        
        response_message_text = self.extract_first_json_block(response_message_text).strip()
        response_message_text = response_message_text.removeprefix("```json")
        response_message_text = response_message_text.removesuffix("```")
        response_message_text = response_message_text.strip()
        logger.info(f"Next action response: {response_message_text}")

        try:
            action_data = json.loads(response_message_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse next action response: {e}")
            action_data = {"action": "UNKNOWN", "reason": "Failed to parse response"}
        if action_data["action"] == "UNKNOWN":
            action = self.choose_random_action(
                frames=self.frames,
                latest_frame=latest_frame
            )
            action.reasoning["reason"] = "Failed to parse response in next action, so using random action"
            return action
        return self.convert_action_text_to_game_action(
            action_text=action_data.get("action", ""),
            reason=action_data.get("reason", "")
        )    

    def is_max_goal_actions_limit_reached(self) -> bool:
        """Check if the maximum goal actions limit is reached."""
        return self.game_context.goal_actions_count >= self.game_context.max_goal_actions_limit

    def is_goal_achieved(
        self, previous_frame: FrameData, current_frame: FrameData
    ) -> tuple[bool, str]:
        previous_grid = previous_frame.frame[0] if previous_frame.frame else []
        previous_map_image = self.generate_grid_image_with_zone(previous_grid)
        previous_image_b64 = base64.b64encode(previous_map_image).decode()

        current_grid = current_frame.frame[0] if current_frame.frame else []
        current_map_image = self.generate_grid_image_with_zone(current_grid)
        current_image_b64 = base64.b64encode(current_map_image).decode()

        prompt = GOAL_ACHIEVEMENT_CHECK_PROMPT.format(
            current_goal=self.game_context.goal,
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
        response_message_text = response.choices[0].message.content.strip()
        self.track_tokens(
            response.usage.total_tokens, response.choices[0].message.content
        )
        self.capture_reasoning_from_response(response)

        logger.info(f"Response: {response_message_text}")
        response_message_text = response_message_text.lower()
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
        if not previous_frame or not current_frame:
            return False
        if not previous_frame.frame or not current_frame.frame:
            return False
        previous_grid = previous_frame.frame[-1]
        current_grid = current_frame.frame[0]
        if len(previous_grid) != len(current_grid):
            return False
        for row_prev, row_curr in zip(previous_grid, current_grid):
            if row_prev != row_curr:
                return False
        return True

    def convert_action_text_to_game_action(
        self, action_text: str, reason: str = ""
    ) -> GameAction:
        """Convert action text to GameAction object."""
        action_text = action_text.strip().upper()
        if action_text == "W":
            action = GameAction.ACTION1
        elif action_text == "S":
            action = GameAction.ACTION2
        elif action_text == "A":
            action = GameAction.ACTION3
        elif action_text == "D":
            action = GameAction.ACTION4
        elif action_text == "F":
            action = GameAction.ACTION5
        elif action_text == "RESET":
            action = GameAction.RESET
        elif action_text.startswith("CLICK"):
            if not(action_text.startswith("CLICK") and action_text.endswith(")")):
                coords = [0, 0]
            else:
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
            return "S"
        elif action == GameAction.ACTION3:
            return "A"
        elif action == GameAction.ACTION4:
            return "D"
        elif action == GameAction.ACTION5:
            return "F"
        elif action == GameAction.RESET:
            return "RESET"
        elif action == GameAction.ACTION6:
            data = action.action_data
            return f"CLICK({data.x},{data.y})"
        else:
            logger.warning(f"Unknown GameAction: {action}")
            return "UNKNOWN"

    def calculate_effect_probabilities(self, total_actions: Dict[str, int], no_effect_actions: Dict[str, int]) -> Dict[str, float]:
        """Calculate the probability that each action has an effect on gameplay."""
        probabilities = {}
        for action in total_actions:
            if action in ["RESET", "F"]:  # Ignore irrelevant actions
                continue
            total = total_actions[action]
            if total == 0:
                probabilities[action] = 0.0
            else:
                effective_count = total - no_effect_actions[action]
                probabilities[action] = effective_count / total
        return probabilities

    def count_actions(self, frames: List[FrameData]) -> (Dict[str, int], Dict[str, int]):
        """Count total actions and no-effect actions from frames."""
        total_action_counts = {k: 0 for k in ["W", "A", "S", "D", "F", "CLICK", "RESET"]}
        no_effect_actions = {k: 0 for k in ["W", "A", "S", "D", "F", "CLICK", "RESET"]}

        prev_frame = None
        for frame in frames:
            action = frame.action_input.id
            if action.is_complex():
                action_text = "CLICK"
            else:
                action_text = self.convert_game_action_to_text(action)

            if action_text not in total_action_counts:
                continue  # Skip unknown actions

            total_action_counts[action_text] += 1
            if self.is_frames_equal(prev_frame, frame):
                no_effect_actions[action_text] += 1
            prev_frame = frame

        return total_action_counts, no_effect_actions

    def get_actions_with_effect(self, total_actions: Dict[str, int], no_effect_actions: Dict[str, int]) -> Dict[str, int]:
        """Return actions that had an effect on the game (total - no effect)."""
        return {action: total_actions[action] - no_effect_actions[action]
                for action in total_actions if action not in ["RESET", "F"]}

    def generate_logical_analysis_summary(self, frames: List[FrameData]) -> str:
        """Generate a detailed, human-readable summary of logical analysis actions."""
        if not frames:
            return "No frames available for logical analysis."

        total_action_counts, no_effect_actions = self.count_actions(frames)
        # Remove non-relevant actions
        for key in ["RESET", "F"]:
            total_action_counts.pop(key, None)
            no_effect_actions.pop(key, None)

        summary_lines = ["Out of all user inputs recorded during gameplay:\n"]

        for action_text in total_action_counts:
            total = total_action_counts[action_text]
            no_effect = no_effect_actions[action_text]

            if total == 0:
                continue

            if no_effect == total:
                line = f"- **{action_text}** was used {total} time{'s' if total > 1 else ''}, and all had no effect on the game."
            elif no_effect == 0:
                line = f"- **{action_text}** was used {total} time{'s' if total > 1 else ''}, and all had an effect on the game."
            else:
                line = f"- **{action_text}** was used {total} time{'s' if total > 1 else ''}, with {no_effect} input{'s' if no_effect > 1 else ''} having no effect on gameplay."

            summary_lines.append(line)

        summary_text = "\n".join(summary_lines)
        logger.info(f"Logical analysis actions summary: {summary_text}")
        return summary_text

    def generate_video_from_grids(
        self,
        frames: list[FrameData],
        video_output_path="output_video.mp4",
        pixel_size=10,
        fps=10,
        skip_repeated_frames=False,
    ) -> list[FrameData]:
        # === Color palette (from key_colors as hex) ===
        key_colors = {
            0: "#FFFFFF", 1: "#CCCCCC", 2: "#999999", 3: "#666666",
            4: "#333333", 5: "#000000", 6: "#E53AA3", 7: "#FF7BCC",
            8: "#F93C31", 9: "#1E93FF", 10: "#88D8F1", 11: "#FFDC00",
            12: "#FF851B", 13: "#921231", 14: "#4FCC30", 15: "#A356D6"
        }

        # Convert to BGR for OpenCV
        palette = np.array(
            [tuple(int(color[i:i+2], 16) for i in (1, 3, 5))[::-1] for color in key_colors.values()],
            dtype=np.uint8
        )

        # === Determine frame size from first frame ===
        # select frame_data which has frames
        frame_data = [frame for frame in frames if len(frame.frame) > 0][0]
        first_grid = frame_data.frame[0]
        h, w = len(first_grid), len(first_grid[0])
        frame_size = (w * pixel_size, h * pixel_size)

        # === Setup video writer ===
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_output_path, fourcc, fps, frame_size)
        effective_frames : list[tuple[GameAction, int]] = []

        # === Frame processing loop with deduplication ===
        prev_frame = frames[0]
        for frame in frames[1:]:
            if skip_repeated_frames and self.is_frames_equal(prev_frame, frame):
                prev_frame = frame
                logger.debug("Skipping duplicate frame")
                continue  # Skip duplicate frame
            effective_frames.append(frame)
            prev_frame = frame
            for grid in frame.frame:
                grid_array = np.array(grid, dtype=np.uint8)
                color_image = palette[grid_array]  # shape: (H, W, 3)
                scaled_image = cv2.resize(color_image, frame_size, interpolation=cv2.INTER_NEAREST)
                video.write(scaled_image)

        video.release()
        return effective_frames

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

    def extract_first_json_block(self, markdown_text):
        pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(pattern, markdown_text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def find_shape_positions(self, grid):
        rows, cols = len(grid), len(grid[0])
        visited = [[False] * cols for _ in range(rows)]
        positions = []
        
        def bfs(start_r, start_c):
            """Mark all cells in the same connected shape."""
            val = grid[start_r][start_c]
            q = deque([(start_r, start_c)])
            visited[start_r][start_c] = True
            
            while q:
                r, c = q.popleft()
                for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if not visited[nr][nc] and grid[nr][nc] == val:
                            visited[nr][nc] = True
                            q.append((nr, nc))
        
        for r in range(rows):
            for c in range(cols):
                if not visited[r][c]:
                    bfs(r, c)
                    positions.append((c, r))  # one representative per shape
        
        return positions


    def get_color_for_cell_value(self, cell_value: int) -> str:
        key_colors_named = {
            0: "White",
            1: "Light Gray",
            2: "Medium Gray",
            3: "Dark Gray",
            4: "Very Dark Gray",
            5: "Black",
            6: "Magenta / Pink",
            7: "Light Pink",
            8: "Bright Red",
            9: "Bright Blue",
            10: "Sky Blue",
            11: "Bright Yellow",
            12: "Orange",
            13: "Dark Red / Maroon",
            14: "Bright Green",
            15: "Purple"
        }
        return key_colors_named.get(cell_value, "")
