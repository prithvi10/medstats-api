"""Tests for professor access helpers."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from src.medstats.services.review_context import professor_can_access_submission


class TestProfessorAccess:
    @pytest.mark.asyncio
    async def test_access_by_professor_id(self):
        db = AsyncMock()
        db.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))))

        submission = MagicMock()
        submission.professor_id = uuid4()
        submission.student_id = uuid4()

        assert await professor_can_access_submission(db, str(submission.professor_id), submission)

    @pytest.mark.asyncio
    async def test_access_by_linked_student(self):
        student_id = uuid4()
        prof_id = uuid4()

        db = AsyncMock()
        db.execute = AsyncMock(
            return_value=MagicMock(all=MagicMock(return_value=[(student_id,)]))
        )

        submission = MagicMock()
        submission.professor_id = None
        submission.student_id = student_id

        assert await professor_can_access_submission(db, str(prof_id), submission)

    @pytest.mark.asyncio
    async def test_denied_without_link(self):
        db = AsyncMock()
        db.execute = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=[])))

        submission = MagicMock()
        submission.professor_id = None
        submission.student_id = uuid4()

        assert not await professor_can_access_submission(db, str(uuid4()), submission)
