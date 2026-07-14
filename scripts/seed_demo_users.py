#!/usr/bin/env python3
"""Seed demo professor/student accounts and enrollment link."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from src.shared.db.session import async_session_factory
from src.shared.security.auth import hash_password
from src.medstats.db.models import User, UserRole, ProfessorStudent


DEMO_USERS = [
    {
        "email": "prof@medstats.io",
        "password": "admin123",
        "name": "Dr. Demo Professor",
        "role": UserRole.PROFESSOR,
        "department": "Orthopaedics",
        "institution": "Demo Medical College",
    },
    {
        "email": "student@medstats.io",
        "password": "admin123",
        "name": "Demo Student",
        "role": UserRole.STUDENT,
        "department": "Orthopaedics",
        "institution": "Demo Medical College",
    },
]


async def seed():
    async with async_session_factory() as db:
        created = {}
        for spec in DEMO_USERS:
            result = await db.execute(select(User).where(User.email == spec["email"]))
            user = result.scalar_one_or_none()
            if not user:
                user = User(
                    email=spec["email"],
                    password_hash=hash_password(spec["password"]),
                    name=spec["name"],
                    role=spec["role"],
                    department=spec["department"],
                    institution=spec["institution"],
                )
                db.add(user)
                await db.flush()
                print(f"Created {spec['role'].value}: {spec['email']}")
            else:
                print(f"Exists: {spec['email']}")
            created[spec["role"]] = user

        prof = created[UserRole.PROFESSOR]
        student = created[UserRole.STUDENT]

        link_result = await db.execute(
            select(ProfessorStudent).where(
                ProfessorStudent.professor_id == prof.id,
                ProfessorStudent.student_id == student.id,
            )
        )
        if not link_result.scalar_one_or_none():
            db.add(ProfessorStudent(professor_id=prof.id, student_id=student.id, academic_year="2025-26"))
            print(f"Linked professor {prof.email} ↔ student {student.email}")

        await db.commit()
        print("\nDemo accounts ready:")
        print("  Professor: prof@medstats.io / admin123")
        print("  Student:   student@medstats.io / admin123")


if __name__ == "__main__":
    asyncio.run(seed())
