import textwrap
from agents.structs import FrameData
from agents.templates.reasoning_agent import ReasoningAgent
from openai import OpenAI
import os
from enum import Enum


hypothesis = """Hypothesis 1: The primary objective is to push the orange/blue block into the black goal square.
Because the orange/blue block is the only block that can be pushed."""

class CustomReasoningAgent(ReasoningAgent):
    MODEL="gemini-2.5-flash"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.hypothesis = hypothesis


class HypothsesisNavigatorAgent(CustomReasoningAgent):
    """ A custom agent that navigates through hypotheses in a game-like environment."""
    def build_user_prompt(self, latest_frame):
        return textwrap.dedent(
            f"""
You are playing a video game.

Your ultimate goal is to achieve the below hypothesis by taking actions based on the game grid image.

<hypothesis>
{self.hypothesis}
</hypothesis>

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
