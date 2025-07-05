"""
Session management module for Walta Framework.
"""

import time
import uuid
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SessionError(Exception):
    """Custom exception for session-related issues."""
    pass

class WaltaSession:
    """
    Represents a single session in the Walta Framework.
    """
    def __init__(self, user_id: str, expiry: int):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.expiry = expiry
        self.metadata: Dict = {}

    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return time.time() - self.last_accessed > self.expiry

    def touch(self) -> None:
        """Update the last accessed time."""
        self.last_accessed = time.time()

    def to_dict(self) -> Dict:
        """Convert session to dictionary representation."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "last_accessed": datetime.fromtimestamp(self.last_accessed).isoformat(),
            "expiry": self.expiry,
            "metadata": self.metadata
        }

class SessionManager:
    """
    Manages user sessions with automatic expiration.
    """
    def __init__(self, session_timeout: int = 3600):
        self.session_timeout = session_timeout
        self.sessions: Dict[str, WaltaSession] = {}
        logger.info(f"SessionManager initialized with timeout of {session_timeout} seconds")

    def create_session(self, user_id: str, metadata: Optional[Dict] = None) -> WaltaSession:
        """
        Create a new session for a user.
        """
        try:
            session = WaltaSession(user_id, self.session_timeout)
            if metadata:
                session.metadata = metadata
            self.sessions[session.id] = session
            logger.info(f"Created new session {session.id[:8]}... for user {user_id}")
            return session
        except Exception as e:
            logger.error(f"Failed to create session for user {user_id}: {e}")
            raise SessionError(f"Session creation failed: {e}")

    def get_session(self, session_id: str) -> Optional[WaltaSession]:
        """
        Retrieve a session by ID. Returns None if session doesn't exist or is expired.
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.debug(f"Session {session_id[:8]}... not found")
            return None

        if session.is_expired():
            logger.info(f"Session {session_id[:8]}... has expired")
            self.invalidate_session(session_id)
            return None

        session.touch()
        return session

    def validate_session(self, session_id: str) -> bool:
        """
        Check if a session is valid and not expired.
        """
        return self.get_session(session_id) is not None

    def invalidate_session(self, session_id: str) -> None:
        """
        Invalidate and remove a session.
        """
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            logger.info(f"Invalidated session {session_id[:8]}... for user {session.user_id}")

    def cleanup_expired(self) -> int:
        """
        Remove all expired sessions and return the count of removed sessions.
        """
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.is_expired()
        ]
        
        for session_id in expired_sessions:
            self.invalidate_session(session_id)

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return len(expired_sessions)

    def get_active_sessions(self, user_id: Optional[str] = None) -> Dict[str, WaltaSession]:
        """
        Get all active (non-expired) sessions, optionally filtered by user_id.
        """
        active_sessions = {}
        for session_id, session in self.sessions.items():
            if not session.is_expired():
                if user_id is None or session.user_id == user_id:
                    active_sessions[session_id] = session
        return active_sessions

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        Get detailed information about a session.
        """
        session = self.get_session(session_id)
        if session:
            return session.to_dict()
        return None
