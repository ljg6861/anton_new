"""
Background service that periodically refreshes the code index.
"""
import time
import threading
import logging
from datetime import datetime

from server.agent.code_indexer import code_indexer
from server.agent.rag_manager import rag_manager

logger = logging.getLogger(__name__)


class CodeIndexRefresher:
    """
    A service that periodically refreshes the code index in the background.
    """

    def __init__(self, refresh_interval_hours: float = 24.0):
        """
        Initialize the refresher service.

        Args:
            refresh_interval_hours: How often to refresh the index (in hours)
        """
        self.refresh_interval = refresh_interval_hours * 3600  # Convert to seconds
        self.running = False
        self.thread = None
        self.last_refresh_time = None

    def start(self):
        """Start the background refresh thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self.thread.start()
        logger.info("Code index refresher service started.")

    def stop(self):
        """Stop the background refresh thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        logger.info("Code index refresher service stopped.")

    def _refresh_loop(self):
        """Main refresh loop that runs in the background."""
        while self.running:
            try:
                # Perform the refresh
                logger.info("Starting scheduled code index refresh...")
                start_time = time.time()
                files_updated = code_indexer.refresh_index()

                # Save the RAG index if any files were updated
                if files_updated > 0:
                    rag_manager.save()

                elapsed_time = time.time() - start_time
                self.last_refresh_time = datetime.now()

                logger.info(
                    f"Code index refresh completed in {elapsed_time:.1f} seconds. "
                    f"{files_updated} files updated."
                )

                # Sleep until next refresh interval
                time.sleep(self.refresh_interval)

            except Exception as e:
                logger.error(f"Error during code index refresh: {e}", exc_info=True)
                # If there was an error, wait a bit before retrying
                time.sleep(300)  # 5 minutes

    def force_refresh(self) -> int:
        """
        Force an immediate refresh of the code index.

        Returns:
            Number of files updated
        """
        try:
            logger.info("Forcing code index refresh...")
            start_time = time.time()
            files_updated = code_indexer.refresh_index()

            # Save the RAG index if any files were updated
            if files_updated > 0:
                rag_manager.save()

            elapsed_time = time.time() - start_time
            self.last_refresh_time = datetime.now()

            logger.info(
                f"Forced code index refresh completed in {elapsed_time:.1f} seconds. "
                f"{files_updated} files updated."
            )

            return files_updated

        except Exception as e:
            logger.error(f"Error during forced code index refresh: {e}", exc_info=True)
            return 0

    def get_status(self) -> dict:
        """
        Get the current status of the refresher service.

        Returns:
            Dictionary with status information
        """
        return {
            "running": self.running,
            "refresh_interval_hours": self.refresh_interval / 3600,
            "last_refresh": self.last_refresh_time.isoformat() if self.last_refresh_time else None
        }


# Create a global instance
code_refresher = CodeIndexRefresher()