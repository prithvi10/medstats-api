#!/usr/bin/env python3
"""Seed admin user for portal access."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from src.shared.db.session import async_session_factory
from src.shared.security.auth import hash_password
from src.medstats.db.models import User, UserRole


async def seed_admin():
    """Create or update admin user."""
    admin_email = "admin@medstats.io"
    admin_password = "MedstatsAdmin@10"

    async with async_session_factory() as db:
        result = await db.execute(select(User).where(User.email == admin_email))
        user = result.scalar_one_or_none()

        if not user:
            user = User(
                email=admin_email,
                password_hash=hash_password(admin_password),
                name="MedStats Admin",
                role=UserRole.ADMIN,
                institution="MedStats",
                department="Administration",
                is_active=True,
            )
            db.add(user)
            await db.commit()
            print(f"✓ Created admin user: {admin_email}")
        else:
            # Update password and ensure admin role
            user.password_hash = hash_password(admin_password)
            user.role = UserRole.ADMIN
            user.is_active = True
            await db.commit()
            print(f"✓ Updated admin user: {admin_email}")

        print("\nAdmin credentials:")
        print(f"  Email:    {admin_email}")
        print(f"  Password: {admin_password}")
        print(f"\nAccess admin portal at: http://localhost:3000/admin")


if __name__ == "__main__":
    asyncio.run(seed_admin())
