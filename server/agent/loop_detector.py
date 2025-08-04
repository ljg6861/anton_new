"""
Enhanced Loop Detection System

Implements robust similarity-based loop detection with configurable thresholds
and pattern-breaking mechanisms to prevent infinite reasoning loops.
"""

import difflib
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class InstructionEntry:
    """Represents a single instruction in the history."""
    content: str
    turn: int
    timestamp: float
    agent: str  # 'planner', 'doer', 'evaluator'
    
    def get_normalized_content(self) -> str:
        """Get normalized content for similarity comparison."""
        # Remove thinking tags and normalize whitespace
        content = re.sub(r'<think>.*?</think>', '', self.content, flags=re.DOTALL)
        content = re.sub(r'\s+', ' ', content.strip())
        return content.lower()


class LoopDetector:
    """
    Detects and prevents reasoning loops using similarity analysis.
    
    Features:
    - Configurable similarity threshold (default 0.85)
    - History tracking with configurable window size
    - Pattern-breaking instruction generation
    - Support for different agent types
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.85,
                 max_history_size: int = 20,
                 loop_detection_window: int = 5):
        self.similarity_threshold = similarity_threshold
        self.max_history_size = max_history_size
        self.loop_detection_window = loop_detection_window
        
        # Use deque for efficient history management
        self.instruction_history: deque = deque(maxlen=max_history_size)
        self.loop_breaks_count = 0
        self.similar_pairs_detected = 0
    
    def add_instruction(self, content: str, turn: int, timestamp: float, agent: str):
        """Add a new instruction to the history."""
        entry = InstructionEntry(content, turn, timestamp, agent)
        self.instruction_history.append(entry)
    
    def detect_loop(self) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Detect if we're in a reasoning loop.
        
        Returns:
            Tuple of (is_loop_detected, pattern_breaking_instruction, loop_info)
        """
        if len(self.instruction_history) < 2:
            return False, None, None
        
        # Check recent instructions for similarity
        recent_instructions = list(self.instruction_history)[-self.loop_detection_window:]
        
        for i in range(len(recent_instructions) - 1):
            for j in range(i + 1, len(recent_instructions)):
                similarity = self._calculate_similarity(
                    recent_instructions[i].get_normalized_content(),
                    recent_instructions[j].get_normalized_content()
                )
                
                if similarity >= self.similarity_threshold:
                    self.similar_pairs_detected += 1
                    
                    # Check if we've detected enough similar instructions to break the loop
                    if self.similar_pairs_detected >= 2:
                        loop_info = {
                            'similarity_score': similarity,
                            'instruction_1': recent_instructions[i].content[:100] + "...",
                            'instruction_2': recent_instructions[j].content[:100] + "...",
                            'turn_1': recent_instructions[i].turn,
                            'turn_2': recent_instructions[j].turn,
                            'agent_1': recent_instructions[i].agent,
                            'agent_2': recent_instructions[j].agent
                        }
                        
                        pattern_breaking_instruction = self._generate_pattern_breaking_instruction(
                            recent_instructions[i], recent_instructions[j]
                        )
                        
                        # Reset counter after breaking
                        self.loop_breaks_count += 1
                        self.similar_pairs_detected = 0
                        
                        return True, pattern_breaking_instruction, loop_info
        
        return False, None, None
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two instruction contents."""
        if not content1 or not content2:
            return 0.0
        
        # Use sequence matching for similarity calculation
        matcher = difflib.SequenceMatcher(None, content1, content2)
        return matcher.ratio()
    
    def _generate_pattern_breaking_instruction(self, 
                                             instruction1: InstructionEntry, 
                                             instruction2: InstructionEntry) -> str:
        """Generate a pattern-breaking instruction to force a different approach."""
        agent_type = instruction1.agent
        
        pattern_breaking_instructions = {
            'planner': [
                "You appear to be repeating similar planning approaches. Try a completely different strategy or break down the task into smaller, more specific steps.",
                "The current approach seems to be cycling. Consider approaching this task from a different angle or using different tools.",
                "Break out of the current pattern by taking a step back and reconsidering the fundamental approach to this problem.",
                "You've been following a similar pattern. Try delegating a more specific, concrete task that focuses on a different aspect of the problem."
            ],
            'doer': [
                "You appear to be stuck in a loop. Try using a different tool or approach to gather the information needed.",
                "Your recent attempts are too similar. Step back and try a fundamentally different method to accomplish this task.",
                "Break the current pattern by exploring a different aspect of the problem or using alternative tools.",
                "Consider providing a FINAL ANSWER with what you've learned so far, or try a completely different approach."
            ],
            'evaluator': [
                "Your recent evaluations are very similar. Take a fresh perspective and focus on different aspects of progress.",
                "Break out of the evaluation pattern by considering different success criteria or progress indicators.",
                "Assess this result from a different angle, considering both immediate progress and long-term task advancement."
            ]
        }
        
        instructions = pattern_breaking_instructions.get(agent_type, pattern_breaking_instructions['doer'])
        # Cycle through different instructions based on break count
        instruction_index = self.loop_breaks_count % len(instructions)
        return instructions[instruction_index]
    
    def get_status(self) -> Dict:
        """Get current status of loop detection."""
        return {
            'history_size': len(self.instruction_history),
            'similar_pairs_detected': self.similar_pairs_detected,
            'loop_breaks_count': self.loop_breaks_count,
            'similarity_threshold': self.similarity_threshold
        }
    
    def reset(self):
        """Reset the loop detector for a new task."""
        self.instruction_history.clear()
        self.loop_breaks_count = 0
        self.similar_pairs_detected = 0