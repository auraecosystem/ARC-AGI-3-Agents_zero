import textwrap
from agents.structs import FrameData
from agents.templates.reasoning_agent import ReasoningAgent
from openai import OpenAI
import os
from enum import Enum


hypothesis = """Hypothesis 1: The primary objective is to push the orange/blue block into the black goal square.
Because the orange/blue block is the only block that can be pushed."""

class CustomReasoningAgent(ReasoningAgent):
    MODEL="gemini-2.5-pro"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.hypothesis = hypothesis


class HypothesisNavigatorAgent(CustomReasoningAgent):
    """ A custom agent that navigates through hypotheses in a game-like environment."""
#     def build_user_prompt(self, latest_frame):
#         return textwrap.dedent(
#             f"""You are an agent in a 2D grid-based video game.

# Your goal is to achieve the following hypothesis state:

# **"The primary objective is to push the orange blue block into the black goal square."**

# Mechanics:
# Orange and blue block is pushable

# You can take one of the following actions at each step:
# - RESET
# - ACTION1 (MOVE_UP)
# - ACTION2 (MOVE_DOWN)
# - ACTION3 (MOVE_LEFT)
# - ACTION4 (MOVE_RIGHT)

# Use the actions to interact with the environment and reach the hypothesis state.  
# If you determine that the hypothesis state cannot be achieved in the current configuration, choose **RESET**.

# You are provided with the current and previous game screens and raw grid data after each action to guide your decision-making.
# """
#         )
