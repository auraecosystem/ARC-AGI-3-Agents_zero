import random
import time
from typing import Any

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState


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

        # Choose a random action that isn't RESET
        action = random.choice([a for a in GameAction if a is not GameAction.RESET])

        # Annotate action
        if action.is_simple():
            action.reasoning = f"RNG told me to pick {action.value}"
        elif action.is_complex():
            action.set_data(
                {
                    "x": random.randint(0, 63),
                    "y": random.randint(0, 63),
                }
            )
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": "RNG said so!",
            }

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
