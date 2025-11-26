import logging
from datetime import datetime

from sqlalchemy import text

from backend.utils.database import db_manager

logger = logging.getLogger(__name__)


class FeedbackAgent:
    """
    Handles:
    - Logging each user interaction into memory.interaction
    - Storing user feedback (star rating + optional comment) into memory.feedback
    - (Optional) fetching recent interactions for analysis / dashboards
    """

    def __init__(self) -> None:
        logger.info("🤖 Feedback agent initialized")

    def log_interaction(
        self,
        session_id: str,
        user_query: str,
        agent_name: str,
        dataset_table: str | None,
        response_summary: str | None,
        chart_reference: str | None = None,
    ) -> int:
        """
        Insert one row into memory.interaction and return its id.

        Assumes this table exists:

            memory.interaction(
                id              SERIAL PRIMARY KEY,
                session_id      TEXT NOT NULL,
                user_query      TEXT NOT NULL,
                agent_name      TEXT NOT NULL,
                dataset_table   TEXT,
                response_summary TEXT,
                chart_reference  TEXT,
                created_at      TIMESTAMP NOT NULL DEFAULT NOW()
            );
        """
        query = """
            INSERT INTO memory.interaction
                (session_id, user_query, agent_name,
                 dataset_table, response_summary, chart_reference, created_at)
            VALUES (:session_id, :user_query, :agent_name,
                    :dataset_table, :response_summary, :chart_reference, :created_at)
            RETURNING id;
        """

        params = {
            "session_id": session_id,
            "user_query": user_query,
            "agent_name": agent_name,
            "dataset_table": dataset_table,
            "response_summary": response_summary,
            "chart_reference": chart_reference,
            "created_at": datetime.utcnow(),
        }

        with db_manager.engine.begin() as conn:
            result = conn.execute(text(query), params)
            interaction_id = result.scalar_one()

        logger.info(f"💾 Logged interaction {interaction_id}")
        return interaction_id

    def store_feedback(
        self,
        interaction_id: int,
        rating: int,
        comment: str | None = None,
    ) -> None:
        """
        Store rating + optional comment for a given interaction_id.

        Assumes this table exists:

            memory.feedback(
                id              SERIAL PRIMARY KEY,
                interaction_id  INTEGER NOT NULL REFERENCES memory.interaction(id),
                rating          INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                comment         TEXT,
                created_at      TIMESTAMP NOT NULL DEFAULT NOW()
            );
        """
        query = """
            INSERT INTO memory.feedback (interaction_id, rating, comment, created_at)
            VALUES (:interaction_id, :rating, :comment, :created_at);
        """

        params = {
            "interaction_id": interaction_id,
            "rating": rating,
            "comment": comment,
            "created_at": datetime.utcnow(),
        }

        with db_manager.engine.begin() as conn:
            conn.execute(text(query), params)

        logger.info(f"⭐ Stored feedback for interaction {interaction_id}: {rating} stars")

    def get_recent_interactions(self, limit: int = 20):
        """
        Optional helper: retrieve recent interactions for dashboards/analysis.
        """
        query = """
            SELECT
                i.id,
                i.created_at,
                i.session_id,
                i.user_query,
                i.agent_name,
                i.dataset_table,
                f.rating,
                f.comment
            FROM memory.interaction i
            LEFT JOIN memory.feedback f
              ON f.interaction_id = i.id
            ORDER BY i.created_at DESC
            LIMIT :limit;
        """

        try:
            return db_manager.execute_query_dict(query, {"limit": limit})
        except Exception as e:
            logger.error(f"❌ Failed to load recent interactions: {e}")
            return []