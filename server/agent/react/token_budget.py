"""
Token budget allocation for different prompt sections
"""
from dataclasses import dataclass


@dataclass
class TokenBudget:
    """Token budget allocation for different prompt sections"""
    total_budget: int = 16384  # Much larger budget for the larger context window
    system_tools_pct: float = 0.15
    domain_bundle_pct: float = 0.25
    session_summary_pct: float = 0.15
    working_memory_pct: float = 0.40
    scratchpad_pct: float = 0.05
    
    @property
    def system_tools_budget(self) -> int:
        return int(self.total_budget * self.system_tools_pct)
    
    @property 
    def domain_bundle_budget(self) -> int:
        return int(self.total_budget * self.domain_bundle_pct)
        
    @property
    def session_summary_budget(self) -> int:
        return int(self.total_budget * self.session_summary_pct)
        
    @property
    def working_memory_budget(self) -> int:
        return int(self.total_budget * self.working_memory_pct)
        
    @property
    def scratchpad_budget(self) -> int:
        return int(self.total_budget * self.scratchpad_pct)
