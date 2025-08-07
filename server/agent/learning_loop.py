"""
Manages the agent's learning loop process:
1. Experience collection
2. Reflection
3. Knowledge storage
4. Knowledge application
5. Performance tracking
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional

from server.agent.rag_manager import rag_manager

logger = logging.getLogger(__name__)


class LearningLoop:
    """
    Central class responsible for managing the agent's learning cycle.
    """

    def __init__(self):
        self.current_task: Optional[Dict] = None
        self.experiences: List[Dict] = []
        self.reflection_frequency: int = 5  # Number of tasks before triggering reflection
        self.tasks_since_reflection: int = 0
        self.performance_metrics: Dict[str, List[float]] = {
            "success_rate": [],
            "task_duration": [],
            "steps_taken": []
        }

    def start_task(self, task_prompt: str) -> None:
        """Begins tracking a new task."""
        self.current_task = {
            "prompt": task_prompt,
            "start_time": time.time(),
            "actions": [],
            "success": False,
            "feedback": "",
            "steps_taken": 0
        }
        logger.info(f"Learning loop tracking started for task: {task_prompt[:50]}...")

    def record_action(self, action_type: str, action_details: Dict) -> None:
        """Records an action taken during the current task."""
        if not self.current_task:
            logger.warning("Attempted to record action but no task is being tracked")
            return

        self.current_task["actions"].append({
            "type": action_type,
            "details": action_details,
            "timestamp": time.time()
        })
        self.current_task["steps_taken"] += 1

    def complete_task(self, success: bool, feedback: str) -> Dict:
        """Completes the current task and triggers reflection if needed."""
        if not self.current_task:
            logger.warning("Attempted to complete task but no task is being tracked")
            return {}

        self.current_task["success"] = success
        self.current_task["feedback"] = feedback
        self.current_task["end_time"] = time.time()
        self.current_task["duration"] = self.current_task["end_time"] - self.current_task["start_time"]

        # Store this experience
        self.experiences.append(self.current_task)

        logger.info('I learned:\n' + json.dumps(self.current_task, indent=4))

        # Update performance metrics
        self.performance_metrics["success_rate"].append(1.0 if success else 0.0)
        self.performance_metrics["task_duration"].append(self.current_task["duration"])
        self.performance_metrics["steps_taken"].append(self.current_task["steps_taken"])

        # Check if we should trigger reflection
        self.tasks_since_reflection += 1
        if self.tasks_since_reflection >= self.reflection_frequency:
            self.reflect_on_experiences()
            self.tasks_since_reflection = 0

        completed_task = self.current_task
        self.current_task = None
        return completed_task

    def reflect_on_experiences(self) -> None:
        """
        Analyzes recent experiences to extract patterns and learnings.
        In a production system, this could use a dedicated LLM call.
        """
        logger.info("Triggering reflection on recent experiences")

        recent_experiences = self.experiences[-self.reflection_frequency:]
        successful_experiences = [exp for exp in recent_experiences if exp["success"]]

        if not successful_experiences:
            logger.info("No successful experiences to learn from in recent tasks")
            return

        # For each successful experience, extract a learning
        for experience in successful_experiences:
            try:
                # Extract key pattern from successful task
                task_pattern = {
                    "task_type": self._categorize_task(experience["prompt"]),
                    "steps_count": experience["steps_taken"],
                    "successful_approach": self._summarize_approach(experience["actions"]),
                    "time_taken": experience["duration"]
                }

                # Create a learning entry
                learning_text = (
                    f"When handling tasks related to {task_pattern['task_type']}, "
                    f"a successful approach is to {task_pattern['successful_approach']}. "
                    f"This typically takes {experience['steps_taken']} steps "
                    f"and approximately {experience['duration']:.1f} seconds."
                )

                # Store in RAG system
                rag_manager.add_knowledge(
                    text=learning_text,
                    source=f"reflection_{int(time.time())}"
                )
                logger.info(f"Stored new learning: {learning_text}")

            except Exception as e:
                logger.error(f"Error during reflection process: {e}")

    def get_relevant_learnings(self, current_prompt: str) -> List[str]:
        """Retrieves relevant past learnings for the current task."""
        try:
            relevant_docs = rag_manager.retrieve_knowledge(query=current_prompt, top_k=3)
            return [doc["text"] for doc in relevant_docs]
        except Exception as e:
            logger.error(f"Error retrieving relevant learnings: {e}")
            return []

    def get_performance_report(self) -> Dict:
        """Generates a report on the agent's learning progress."""
        if not self.performance_metrics["success_rate"]:
            return {"error": "No performance data available yet"}

        # Calculate moving averages
        window_size = min(10, len(self.performance_metrics["success_rate"]))
        recent_success_rate = sum(self.performance_metrics["success_rate"][-window_size:]) / window_size

        # Calculate improvement trends
        improvement = {}
        for metric, values in self.performance_metrics.items():
            if len(values) >= window_size * 2:
                earlier_avg = sum(values[-window_size * 2:-window_size]) / window_size
                recent_avg = sum(values[-window_size:]) / window_size
                if metric == "success_rate":
                    improvement[metric] = recent_avg - earlier_avg
                else:
                    # For duration and steps, lower is better
                    improvement[metric] = earlier_avg - recent_avg

        return {
            "total_tasks": len(self.experiences),
            "recent_success_rate": recent_success_rate,
            "improvement_trends": improvement,
            "learnings_count": rag_manager.index.ntotal if hasattr(rag_manager, 'index') else "unknown"
        }

    def _categorize_task(self, prompt: str) -> str:
        """Categorizes the type of task from the prompt."""
        # This would be more sophisticated in production
        if "file" in prompt.lower() or "read" in prompt.lower():
            return "file operations"
        elif "code" in prompt.lower() or "script" in prompt.lower():
            return "coding tasks"
        elif "explain" in prompt.lower() or "how" in prompt.lower():
            return "explanations"
        else:
            return "general tasks"

    def _summarize_approach(self, actions: List[Dict]) -> str:
        """Summarizes the approach taken based on action sequence."""
        # This is a simplified implementation
        action_types = [action["type"] for action in actions]
        if "read_file" in action_types and "write_file" in action_types:
            return "first read existing files, then write new content"
        elif "list_directory" in action_types:
            return "explore the directory structure before taking action"
        elif "execute_python_code" in action_types:
            return "test code execution to verify the solution"
        else:
            return "follow a step-by-step approach"


# Singleton instance
learning_loop = LearningLoop()