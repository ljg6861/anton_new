"""
Detects token-level repetition and loops in model output
"""
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class TokenLoopDetector:
    """Detects token-level repetition and loops in model output"""
    
    def __init__(self, window_size: int = 50, min_phrase_length: int = 4, repeat_threshold: int = 3):
        self.window_size = window_size
        self.min_phrase_length = min_phrase_length
        self.repeat_threshold = repeat_threshold
        self.token_buffer = []
        self.phrase_counts = {}
        self.last_warning_position = -1
        
    def add_token(self, token: str) -> bool:
        """Add a token and check for loops. Returns True if loop detected."""
        self.token_buffer.append(token.strip())
        self._maintain_buffer_size()
        
        if len(self.token_buffer) < self.min_phrase_length * 2:
            return False
            
        return self._detect_phrase_repetition()
    
    def _maintain_buffer_size(self):
        """Keep buffer within size limits"""
        if len(self.token_buffer) > self.window_size:
            self.token_buffer.pop(0)
    
    def _detect_phrase_repetition(self) -> bool:
        """Detect if phrases are repeating"""
        for phrase_len in range(self.min_phrase_length, min(len(self.token_buffer) // 2, 15)):
            if self._check_phrase_length_for_repetition(phrase_len):
                return True
        return False
    
    def _check_phrase_length_for_repetition(self, phrase_len: int) -> bool:
        """Check for repetition in phrases of a specific length"""
        phrases = self._extract_phrases(phrase_len)
        phrase_counts = self._count_recent_phrases(phrases)
        
        return self._check_repetition_threshold(phrase_counts)
    
    def _extract_phrases(self, phrase_len: int) -> List[str]:
        """Extract overlapping phrases of given length"""
        phrases = []
        for i in range(len(self.token_buffer) - phrase_len + 1):
            phrase = " ".join(self.token_buffer[i:i + phrase_len])
            phrases.append(phrase)
        return phrases
    
    def _count_recent_phrases(self, phrases: List[str]) -> Dict[str, int]:
        """Count occurrences of phrases in recent window"""
        phrase_counts = {}
        recent_phrases = phrases[-20:]  # Look at last 20 phrases
        
        for phrase in recent_phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        return phrase_counts
    
    def _check_repetition_threshold(self, phrase_counts: Dict[str, int]) -> bool:
        """Check if any phrase exceeds repetition threshold"""
        for phrase, count in phrase_counts.items():
            if self._is_significant_repetition(phrase, count):
                self._log_loop_detection(phrase, count)
                return True
        return False
    
    def _is_significant_repetition(self, phrase: str, count: int) -> bool:
        """Check if repetition is significant enough to flag"""
        return (count >= self.repeat_threshold and 
                len(phrase.strip()) > 10 and
                self._should_warn_about_position())
    
    def _should_warn_about_position(self) -> bool:
        """Check if we should warn based on position to avoid spam"""
        current_pos = len(self.token_buffer)
        should_warn = current_pos - self.last_warning_position > 20
        if should_warn:
            self.last_warning_position = current_pos
        return should_warn
    
    def _log_loop_detection(self, phrase: str, count: int):
        """Log loop detection warning"""
        logger.warning(f"Loop detected: phrase '{phrase[:50]}...' repeated {count} times")
    
    def reset(self):
        """Reset the detector for a new request"""
        self.token_buffer.clear()
        self.phrase_counts.clear()
        self.last_warning_position = -1
