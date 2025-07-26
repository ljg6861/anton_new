# agent/agent_loop.py

"""
The core orchestrator for the multi-turn, streaming agent.
"""
import asyncio
import json
from typing import Any

from server.agent import config

# External dependency for streaming
from server.agent.streaming import stream_response

class RAGHandler:
    """
    Performs asynchronous, post-response evaluation and learning.

    If an agent's response is deemed low-quality, this handler generates
    and stores recommendations for future improvement without blocking the
    main agent loop.
    """

    def __init__(self, model: Any, tokenizer: Any, logger: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger
        self.config = config

    async def _evaluate_performance(self, conversation_history: list[dict]) -> dict:
        """
        Uses an LLM to evaluate the agent's final response in the conversation.
        """
        self.logger.info("--- 🧐 Evaluating agent performance... ---")

        evaluation_prompt_content = f"""
        Act as an impartial performance reviewer. Analyze the last assistant response in the conversation history below based on helpfulness, accuracy, and relevance.

        Return a single JSON object with three keys:
        1. "evaluation": A string, either "good" or "bad".
        2. "reason": A brief explanation for your evaluation.
        3. "recommendation": If the evaluation is "bad", provide a concise, actionable recommendation for future responses. Otherwise, null.

        CONVERSATION HISTORY:
        {json.dumps(conversation_history, indent=2)}

        evaluation_prompt
        """

        messages = [{"role": "user", "content": evaluation_prompt_content}]
        gen_kwargs = self.config.get_generation_kwargs(self.tokenizer, temperature=0.1, max_new_tokens=512)

        json_str = ""
        # This uses the real stream_response function to get the evaluation
        async for token in stream_response(self.model, self.tokenizer, messages, gen_kwargs):
            json_str += token

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            self.logger.error("Failed to decode performance evaluation JSON.")
            return {"evaluation": "error", "reason": "Invalid JSON from model."}

    async def _store_in_knowledge_base(self, recommendation: str):
        """
        Simulates storing a recommendation in a persistent knowledge base (e.g., Vector DB).
        """
        self.logger.info(f"--- 💾 Storing recommendation in RAG DB ---")
        self.logger.info(f"Recommendation: '{recommendation}'")
        # In a real application, you would connect to your DB here.
        # e.g., await vector_db.add_document(recommendation, metadata={"source": "agent_review"})
        await asyncio.sleep(0.5)  # Simulate DB write latency
        self.logger.info("--- ✅ Recommendation stored. ---")

    async def review_and_learn(self, conversation_history: list[dict]):
        """
        The main public method to orchestrate the review and learning process.
        """
        self.logger.info("--- 🚀 Spawning Async Review & Learn Task ---")
        try:
            evaluation_result = await self._evaluate_performance(conversation_history)

            evaluation = evaluation_result.get("evaluation")
            reason = evaluation_result.get("reason")
            recommendation = evaluation_result.get("recommendation")

            self.logger.info(f"Performance Evaluation: {evaluation.upper()}. Reason: {reason}")

            if evaluation == "bad" and recommendation:
                await self._store_in_knowledge_base(recommendation)
            else:
                self.logger.info("--- ✅ No action needed. Performance was adequate. ---")

        except Exception as e:
            self.logger.error(f"--- ❌ Async Review Task Failed: {e} ---")