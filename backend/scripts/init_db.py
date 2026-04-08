import logging
import time

from core.database import Base, engine
from core.models import QueryHistory, User
from sqlalchemy.exc import OperationalError, SQLAlchemyError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_db():
    """Initialize the database by creating all tables."""
    max_retries = 5
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            logger.info("Attempting to create database tables...")
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully!")
            return
        except OperationalError as e:
            if "could not translate host name" in str(e) and attempt < max_retries - 1:
                logger.warning(
                    f"Failed to connect to database. Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(f"Error creating database tables: {e}")
                raise
        except SQLAlchemyError as e:
            logger.error(f"Error creating database tables: {e}")
            raise


if __name__ == "__main__":
    init_db()
