class LearningProgressTool:
    """A tool for checking the agent's learning progress and performance metrics."""

    function = {
        "type": "function",
        "function": {
            "name": "check_learning_progress",
            "description": "Returns metrics about the agent's learning progress and performance improvements over time.",
            "parameters": {
                "type": "object",
                "properties": {}  # No parameters needed
            }
        }
    }

    def run(self, arguments: dict) -> str:
        """Returns metrics about the agent's learning and performance."""
        from server.agent.learning_loop import learning_loop

        report = learning_loop.get_performance_report()

        if "error" in report:
            return f"No learning metrics available yet. The agent needs to complete more tasks to generate performance data."

        output = [
            "📈 Learning Progress Report:",
            f"Total tasks completed: {report['total_tasks']}",
            f"Recent success rate: {report['recent_success_rate'] * 100:.1f}%",
            f"Knowledge entries: {report['learnings_count']}"
        ]

        if report.get('improvement_trends'):
            output.append("\nImprovement Trends:")
            for metric, value in report['improvement_trends'].items():
                trend = "improved" if value > 0 else "declined"
                output.append(f"- {metric}: {trend} by {abs(value):.2f}")

        return "\n".join(output)