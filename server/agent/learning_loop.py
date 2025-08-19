"""
Manages the agent's learning loop process:
1. Experience collection
2. Reflection
3. Knowledge storage 
4. Knowledge application
5. Performance tracking
6. Capability evidence tracking
"""

import logging
import json
import time
import asyncio
import httpx
from typing import AsyncGenerator, Dict, List, Any, Optional, Set
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from server.agent.rag_manager import rag_manager
from server.agent.config import MODEL_SERVER_URL, SYSTEM_ROLE, ASSISTANT_ROLE, USER_ROLE

logger = logging.getLogger(__name__)


class CapabilityConfidence(Enum):
    LOW = "low"         # Initial evidence observed
    MEDIUM = "medium"   # Multiple evidence points observed
    HIGH = "high"       # Consistently demonstrated


class LearningLoop:
    """
    Central class responsible for managing the agent's learning cycle with
    asynchronous capability tracking and LLM-powered analysis.
    """

    def __init__(self, api_base_url: str = MODEL_SERVER_URL):
        self.current_task: Optional[Dict] = None
        self.experiences: List[Dict] = []
        self.reflection_frequency: int = 5  # Number of tasks before triggering reflection
        self.tasks_since_reflection: int = 0
        self.performance_metrics: Dict[str, List[float]] = {
            "success_rate": [],
            "task_duration": [],
            "steps_taken": []
        }
        
        # Capability tracking
        self.capabilities: Dict[str, Dict] = {}
        self.capability_domains: Set[str] = {
            "file_operations", "code_generation", "data_analysis", 
            "web_interaction", "problem_solving", "explanation",
            "tool_use", "planning", "reasoning", "learning"
        }
        self.capability_evidence_threshold: int = 2  # Min examples needed to confirm capability
        
        # For async processing
        self.api_base_url = api_base_url
        self._executor = ThreadPoolExecutor(max_workers=3)
        self._processing_queue = asyncio.Queue()
        self._is_processing = False
        self._background_task = None
        
        # Start background processing only if there's an event loop
        try:
            self._start_background_processing()
        except RuntimeError:
            # No event loop running yet, will start when needed
            logger.info("No event loop available, background processing will start later")

    def _start_background_processing(self):
        """Start background task for processing capabilities and reflections"""
        try:
            loop = asyncio.get_running_loop()
            self._background_task = asyncio.create_task(self._process_queue())
            logger.info("Started background processing for learning loop")
        except RuntimeError:
            # No event loop running, this will be called later
            logger.debug("No event loop running, deferring background processing start")
        
    async def _process_queue(self):
        """Process tasks in the background"""
        self._is_processing = True
        try:
            while True:
                try:
                    task_type, task_data = await self._processing_queue.get()
                    
                    if task_type == "capability_analysis":
                        await self._analyze_capability_with_llm(**task_data)
                    elif task_type == "reflection":
                        await self._reflect_on_experiences_with_llm(**task_data)
                    
                    self._processing_queue.task_done()
                except Exception as e:
                    logger.error(f"Error in background processing: {e}", exc_info=True)
                    await asyncio.sleep(1)  # Prevent tight loop if errors occur
        except asyncio.CancelledError:
            logger.info("Background processing task cancelled")
            self._is_processing = False

    def start_task(self, task_prompt: str) -> None:
        """Begins tracking a new task."""
        self.current_task = {
            "prompt": task_prompt,
            "start_time": time.time(),
            "actions": [],
            "success": False,
            "feedback": "",
            "steps_taken": 0,
            "potential_capabilities": self._identify_potential_capabilities(task_prompt),
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
        """
        Completes the current task and triggers reflection if needed.
        This method is synchronous but queues async processing.
        """
        if not self.current_task:
            logger.warning("Attempted to complete task but no task is being tracked")
            return {}

        self.current_task["success"] = success
        self.current_task["feedback"] = feedback
        self.current_task["end_time"] = time.time()
        self.current_task["duration"] = self.current_task["end_time"] - self.current_task["start_time"]

        # Store this experience
        self.experiences.append(self.current_task)

        logger.info(f"Task completed (success: {success})")
        logger.debug('Task details:\n' + json.dumps(self.current_task, indent=4))

        # Update performance metrics
        self.performance_metrics["success_rate"].append(1.0 if success else 0.0)
        self.performance_metrics["task_duration"].append(self.current_task["duration"])
        self.performance_metrics["steps_taken"].append(self.current_task["steps_taken"])

        # Queue capability analysis if the task was successful
        if success:
            logger.info("Queueing asynchronous capability analysis")
            task_data = {"task": self.current_task.copy()}
            asyncio.create_task(self._add_to_queue("capability_analysis", task_data))

        # Queue reflection if we've reached the frequency threshold
        self.tasks_since_reflection += 1
        if self.tasks_since_reflection >= self.reflection_frequency:
            logger.info("Queueing asynchronous reflection")
            recent_experiences = self.experiences[-self.reflection_frequency:]
            task_data = {"experiences": recent_experiences}
            asyncio.create_task(self._add_to_queue("reflection", task_data))
            self.tasks_since_reflection = 0

        completed_task = self.current_task.copy()
        self.current_task = None
        return completed_task
    
    async def _add_to_queue(self, task_type: str, task_data: Dict):
        """Add a task to the processing queue"""
        await self._processing_queue.put((task_type, task_data))
        
    async def _analyze_capability_with_llm(self, task: Dict) -> None:
        """
        Uses the LLM to analyze capabilities demonstrated in a completed task.
        This runs asynchronously to avoid blocking the main process.
        """
        if not task.get("success", False):
            logger.info("Skipping capability analysis for unsuccessful task")
            return
            
        try:
            logger.info(f"Analyzing capabilities for task: {task['prompt'][:50]}...")
            
            # Create a prompt for capability analysis
            actions_text = "\n".join([
                f"- {i+1}. {action['type']}: {json.dumps(action['details'])[:100]}..."
                for i, action in enumerate(task.get("actions", []))
            ])
            
            prompt = f"""Analyze the completed task and identify capabilities that were demonstrated.
            
Task prompt: {task['prompt']}

Actions taken:
{actions_text}

Feedback: {task.get('feedback', 'No feedback provided')}

Identify the specific capabilities demonstrated in this task. Consider the following capability domains:
{', '.join(self.capability_domains)}

For each capability you identify:
1. Provide the capability name
2. Rate the confidence level (LOW, MEDIUM, HIGH) that this capability was demonstrated
3. Explain the specific evidence from the task that demonstrates this capability
4. Describe the approach used to demonstrate this capability
5. Suggest how this capability could be improved

Format your response as JSON with this structure:
{{
    "capabilities": [
        {{
            "name": "capability_name",
            "confidence": "MEDIUM",
            "evidence": "specific evidence from the task",
            "approach": "approach used to demonstrate this capability",
            "improvement": "how this capability could be improved"
        }}
    ]
}}

Think carefully about which capabilities were truly demonstrated. Only include capabilities with clear evidence.
"""

            # Call the LLM for capability analysis
            messages = [
                {"role": SYSTEM_ROLE, "content": "You are a capability analyzer for an AI system. Analyze completed tasks to identify capabilities."},
                {"role": USER_ROLE, "content": prompt}
            ]
            
            response = await self._call_llm_api(messages)
            
            # Parse the LLM response to extract capability data
            capability_data = self._extract_json_from_response(response)
            
            if not capability_data or "capabilities" not in capability_data:
                logger.warning("Failed to extract capability data from LLM response")
                return
                
            # Update capability registry with the analysis results
            for capability in capability_data["capabilities"]:
                await self._register_capability_evidence(
                    capability["name"], 
                    task,
                    capability["confidence"], 
                    capability["evidence"],
                    capability["approach"],
                    capability.get("improvement", "")
                )
                
        except Exception as e:
            logger.error(f"Error during capability analysis: {e}", exc_info=True)
    
    async def _register_capability_evidence(
        self, 
        capability_name: str, 
        task: Dict, 
        confidence_level: str, 
        evidence_text: str,
        approach_text: str,
        improvement_text: str
    ) -> None:
        """
        Registers evidence of a capability in the system.
        """
        capability_name = capability_name.lower().strip()
        # Normalize domain name to match our predefined set
        matched_domain = self._match_capability_domain(capability_name)
        
        if not matched_domain:
            logger.warning(f"Capability '{capability_name}' doesn't match any known domain")
            return
            
        # Convert confidence level string to enum
        try:
            confidence = CapabilityConfidence[confidence_level.upper()]
        except (KeyError, AttributeError):
            confidence = CapabilityConfidence.LOW
        
        # Create evidence record
        evidence = {
            "task_prompt": task["prompt"],
            "evidence_text": evidence_text,
            "approach": approach_text,
            "improvement": improvement_text,
            "actions": [action["type"] for action in task["actions"]],
            "timestamp": time.time(),
            "duration": task["duration"],
            "steps_taken": task["steps_taken"],
            "confidence": confidence.value
        }
        
        # Update capabilities registry
        if matched_domain not in self.capabilities:
            # Create new capability entry
            self.capabilities[matched_domain] = {
                "first_seen": time.time(),
                "evidence": [evidence],
                "confidence": confidence.value,
                "best_evidence": evidence,
                "improvement_suggestions": [improvement_text] if improvement_text else []
            }
            logger.info(f"Registered first evidence of capability: {matched_domain}")
            
            # Create initial knowledge entry for this capability
            await self._create_capability_knowledge(matched_domain, evidence)
            
        else:
            # Update existing capability
            self.capabilities[matched_domain]["evidence"].append(evidence)
            evidence_count = len(self.capabilities[matched_domain]["evidence"])
            
            # Update confidence based on evidence count and this evidence's confidence
            confidence_values = [CapabilityConfidence[e["confidence"].upper()].value 
                               if isinstance(e["confidence"], str) else e["confidence"] 
                               for e in self.capabilities[matched_domain]["evidence"]]
            
            # Use the most frequent confidence level
            from collections import Counter
            most_common_confidence = Counter(confidence_values).most_common(1)[0][0]
            self.capabilities[matched_domain]["confidence"] = most_common_confidence
                
            # Check if this is better evidence (higher confidence or fewer steps)
            current_best = self.capabilities[matched_domain]["best_evidence"]
            current_best_confidence = CapabilityConfidence[current_best["confidence"].upper()].value if isinstance(current_best["confidence"], str) else current_best["confidence"]
            new_confidence = confidence.value
            
            if (new_confidence > current_best_confidence or 
                (new_confidence == current_best_confidence and evidence["steps_taken"] < current_best["steps_taken"])):
                self.capabilities[matched_domain]["best_evidence"] = evidence
                
                # Update RAG knowledge with this better evidence
                await self._create_capability_knowledge(matched_domain, evidence)
                
            # Add unique improvement suggestion
            if improvement_text and improvement_text not in self.capabilities[matched_domain]["improvement_suggestions"]:
                self.capabilities[matched_domain]["improvement_suggestions"].append(improvement_text)
                
            logger.info(f"Added evidence for capability: {matched_domain} (total: {evidence_count})")

    async def _create_capability_knowledge(self, capability_domain: str, evidence: Dict) -> None:
        """Creates knowledge entries for capability evidence"""
        capability_learning = (
            f"CAPABILITY EVIDENCE - {capability_domain}: "
            f"I have demonstrated the ability to {evidence['approach']}. "
            f"Evidence: {evidence['evidence_text'][:200]}. "
            f"This took {evidence['steps_taken']} steps and {evidence['duration']:.1f} seconds."
        )
        
        if evidence.get("improvement"):
            capability_learning += f" To improve: {evidence['improvement']}"
        
        # Store in RAG system with specific metadata for capabilities
        rag_manager.add_knowledge(
            text=capability_learning,
            source=f"capability_{capability_domain}_{int(time.time())}",
        )
        logger.info(f"Created knowledge entry for capability: {capability_domain}")

    async def _reflect_on_experiences_with_llm(self, experiences: List[Dict]) -> None:
        """
        Uses the LLM to reflect on experiences and extract patterns/learnings.
        This runs asynchronously to avoid blocking the main process.
        """
        if not experiences:
            logger.info("No experiences to reflect on")
            return
            
        successful_experiences = [exp for exp in experiences if exp.get("success", False)]
        
        if not successful_experiences:
            logger.info("No successful experiences to learn from in recent tasks")
            return
            
        try:
            logger.info("Reflecting on recent experiences...")
            
            # Format experiences for the prompt
            experiences_text = ""
            for i, exp in enumerate(successful_experiences[:5]):  # Limit to 5 experiences
                actions_summary = ", ".join([action["type"] for action in exp.get("actions", [])][:5])
                experiences_text += f"""
Experience {i+1}:
- Task: {exp["prompt"][:200]}...
- Actions: {actions_summary}...
- Success: {exp["success"]}
- Steps: {exp["steps_taken"]}
- Duration: {exp["duration"]:.1f} seconds
"""
                
            prompt = f"""Reflect on these successful task experiences and extract patterns, insights, and learnings:

{experiences_text}

For each pattern or insight you identify:
1. Describe the pattern or insight
2. Explain how it can be applied to future tasks
3. Note any conditions or limitations of this pattern

Format your response as JSON with this structure:
{{
    "learnings": [
        {{
            "pattern": "description of the pattern",
            "application": "how to apply this pattern",
            "conditions": "when this pattern is applicable",
            "task_types": ["list", "of", "relevant", "task", "types"]
        }}
    ]
}}

Focus on extracting meaningful, actionable patterns that would help an AI assistant perform better on future tasks.
"""

            # Call LLM for reflection
            messages = [
                {"role": SYSTEM_ROLE, "content": "You are a reflection system for an AI assistant, extracting learnings from past experiences."},
                {"role": USER_ROLE, "content": prompt}
            ]
            
            response = await self._call_llm_api(messages)
            
            # Parse the response to extract learnings
            reflection_data = self._extract_json_from_response(response)
            
            if not reflection_data or "learnings" not in reflection_data:
                logger.warning("Failed to extract learning data from LLM response")
                return
                
            # Store learnings in the RAG system
            for learning in reflection_data["learnings"]:
                learning_text = (
                    f"PATTERN: {learning['pattern']} "
                    f"APPLICATION: {learning['application']} "
                    f"CONDITIONS: {learning['conditions']}"
                )
                
                # Add task type tags for better retrieval
                task_types = learning.get("task_types", [])
                
                # Store in RAG system with metadata
                rag_manager.add_knowledge(
                    text=learning_text,
                    source=f"reflection_{int(time.time())}",
                )
                logger.info(f"Stored new learning pattern: {learning['pattern'][:50]}...")
                
        except Exception as e:
            logger.error(f"Error during reflection process: {e}", exc_info=True)

    async def get_relevant_learnings(self, current_prompt: str) -> List[str]:
        """Retrieves relevant past learnings for the current task."""
        try:
            # Identify potential capability domains
            potential_domains = self._identify_potential_capabilities(current_prompt)
            
            # Build a more focused query combining the prompt with capability domains
            enhanced_query = current_prompt
            if potential_domains:
                domain_text = " ".join(potential_domains)
                enhanced_query = f"{current_prompt} {domain_text}"
            
            # Get relevant documents with metadata filtering
            relevant_docs = rag_manager.retrieve_knowledge(
                query=enhanced_query, 
                top_k=5,
            )
            
            # Get capability-specific evidence
            capability_learnings = []
            for domain in potential_domains:
                if domain in self.capabilities and self.capabilities[domain]["confidence"] != CapabilityConfidence.LOW.value:
                    best_evidence = self.capabilities[domain].get("best_evidence")
                    if best_evidence:
                        capability_learnings.append(
                            f"CAPABILITY ({domain}): {best_evidence['approach']} (confidence: {self.capabilities[domain]['confidence']})"
                        )
            
            # Combine and deduplicate learnings
            all_learnings = [doc["text"] for doc in relevant_docs] + capability_learnings
            return list(dict.fromkeys(all_learnings))  # Remove duplicates while preserving order
            
        except Exception as e:
            logger.error(f"Error retrieving relevant learnings: {e}", exc_info=True)
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

        # Add capability information to the report
        capability_summary = {
            "total_capabilities": len(self.capabilities),
            "capability_domains": {
                domain: {
                    "evidence_count": len(data["evidence"]),
                    "confidence": data.get("confidence", CapabilityConfidence.LOW.value)
                }
                for domain, data in self.capabilities.items()
            },
            "strongest_capabilities": sorted(
                [(domain, data.get("confidence", CapabilityConfidence.LOW.value)) 
                for domain, data in self.capabilities.items()],
                key=lambda x: (
                    0 if x[1] == CapabilityConfidence.HIGH.value else 
                    1 if x[1] == CapabilityConfidence.MEDIUM.value else 2,
                    -len(self.capabilities[x[0]]["evidence"])
                )
            )[:3]
        }

        return {
            "total_tasks": len(self.experiences),
            "recent_success_rate": recent_success_rate,
            "improvement_trends": improvement,
            "learnings_count": rag_manager.index.ntotal if hasattr(rag_manager, 'index') else "unknown",
            "capabilities": capability_summary
        }
        
    def get_capability_evidence(self, capability_domain: str) -> List[Dict]:
        """
        Retrieves evidence of a specific capability.
        Returns examples of the agent demonstrating this capability.
        """
        if capability_domain not in self.capabilities:
            return []
        
        return self.capabilities[capability_domain]["evidence"]
    
    def get_capabilities_by_confidence(self, min_confidence: CapabilityConfidence = CapabilityConfidence.MEDIUM) -> Dict[str, Dict]:
        """
        Returns capabilities that meet or exceed the given confidence level.
        """
        return {
            domain: data for domain, data in self.capabilities.items()
            if data.get("confidence", CapabilityConfidence.LOW.value) >= min_confidence.value
        }

    async def _call_llm_api(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Execute LLM request directly, replacing dependency on doer.py
        Tools parameter removed as it's unused by Ollama - tools are described in system prompt instead.
        """
        request_payload = {
            "messages": messages,
            "temperature": 0.6,
            'complex': True,
        }

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", f"{self.api_base_url}/v1/chat/stream", json=request_payload) as response:
                    response.raise_for_status()
                    
                    full_response_content = ""
                    # Iterate over the streamed chunks
                    async for chunk in response.aiter_text():
                        for line in chunk.split('\n'):
                            if line.startswith('data: '):
                                content = line[6:]  # Remove 'data: ' prefix
                                if content == '[DONE]':
                                    return
                                full_response_content += content
                            elif line.strip():
                                # Fallback for non-SSE format
                                full_response_content += line
                learning_response = full_response_content.split('</think>')[1] if len(full_response_content.split('</think>')) > 1 else full_response_content
                logger.info(f"ReActAgent: Received response from model server: {learning_response}...")
                return learning_response

            except httpx.RequestError as e:
                # Added self here to match your original method signature
                logger.error(f"ReActAgent: API request to model server failed: {e}")
                return f"\n[ERROR: Could not connect to the model server: {e}]\n"
            except Exception as e:
                logger.error(f"ReActAgent: An unexpected error occurred during model streaming: {e}", exc_info=True)
                return f"\n[ERROR: An unexpected error occurred: {e}]\n"
    
    def _extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON from LLM response, handling various formats"""
        try:
            # Try direct JSON parsing
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
                    
            # Try to find any JSON-like structure
            json_pattern = re.search(r'({[\s\S]*})', response)
            if json_pattern:
                try:
                    return json.loads(json_pattern.group(1))
                except json.JSONDecodeError:
                    pass
                    
            logger.warning("Could not extract valid JSON from response")
            return {}
        
    def _identify_potential_capabilities(self, prompt: str) -> List[str]:
        """
        Identifies which capability domains a task might involve.
        Uses simple keyword matching but could use more sophisticated classification.
        """
        domains = []
        prompt_lower = prompt.lower()
        
        domain_keywords = {
            "file_operations": ["file", "read", "write", "save", "open", "directory", "folder", "path"],
            "code_generation": ["code", "function", "script", "program", "implement", "develop", "programming"],
            "data_analysis": ["data", "analyze", "csv", "statistics", "plot", "dataset", "visualization"],
            "web_interaction": ["web", "url", "http", "api", "request", "website", "browser", "internet"],
            "problem_solving": ["solve", "problem", "puzzle", "challenge", "solution", "figure out", "resolve"],
            "explanation": ["explain", "describe", "summarize", "how does", "what is", "why", "teach"],
            "tool_use": ["use", "tool", "execute", "run", "apply", "utility", "command"],
            "planning": ["plan", "steps", "strategy", "approach", "roadmap", "organize", "structure"],
            "reasoning": ["reason", "logic", "deduce", "infer", "why", "because", "therefore", "conclude"],
            "learning": ["learn", "adapt", "improve", "train", "understand", "study", "practice"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                domains.append(domain)
                
        return domains
    
    def _match_capability_domain(self, capability_name: str) -> Optional[str]:
        """Match a capability name to a standard domain"""
        capability_lower = capability_name.lower()
        
        # Direct match
        if capability_lower in self.capability_domains:
            return capability_lower
            
        # Fuzzy match based on substring
        for domain in self.capability_domains:
            if domain in capability_lower or capability_lower in domain:
                return domain
                
        # Keyword-based matching as fallback
        domain_keywords = {
            "file_operations": ["file", "directory", "folder", "read", "write"],
            "code_generation": ["code", "program", "script", "implement"],
            "data_analysis": ["data", "analysis", "statistics", "visualization"],
            "web_interaction": ["web", "http", "api", "request"],
            "problem_solving": ["problem", "solution", "solve", "resolve"],
            "explanation": ["explain", "description", "summarize"],
            "tool_use": ["tool", "utility", "execute", "run"],
            "planning": ["plan", "strategy", "steps", "approach"],
            "reasoning": ["reason", "logic", "inference", "deduction"],
            "learning": ["learn", "adapt", "improve", "study"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in capability_lower for keyword in keywords):
                return domain
                
        # No match found
        return None


# Singleton instance
learning_loop = LearningLoop()