#!/usr/bin/env python3
"""
Test script to exercise the MedStats API end-to-end via HTTP.
Runs against a local instance at http://localhost:8000.

Usage:
    python scripts/test_api.py [--base-url http://localhost:8000]
"""
import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"

def main():
    parser = argparse.ArgumentParser(description="MedStats API E2E Test Script")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()

    BASE = args.base_url.rstrip("/")
    print(f"=== MedStats API Test Script ===")
    print(f"Target: {BASE}\n")

    # 1. Health check
    print("1. Health check...")
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        assert r.status_code == 200, f"Health check failed: {r.status_code}"
        print(f"   OK: {r.json()}\n")
    except requests.ConnectionError:
        print(f"   FAILED: Cannot connect to {BASE}")
        print("   Make sure the server is running: ./scripts/run_local.sh")
        sys.exit(1)

    # 2. Register professor
    print("2. Register professor...")
    prof_data = {
        "email": f"prof_{int(time.time())}@test.com",
        "password": "TestPass123!",
        "full_name": "Dr. Test Professor",
        "role": "professor",
        "department": "Orthopaedics",
        "institution": "Test Medical College",
    }
    r = requests.post(f"{BASE}/api/auth/register", json=prof_data)
    if r.status_code == 201:
        prof_user = r.json()
        print(f"   Created: {prof_user['email']} (id: {prof_user['id']})\n")
    elif r.status_code == 400 and "already registered" in r.text.lower():
        print("   Already exists, skipping.\n")
    else:
        print(f"   Response: {r.status_code} - {r.text}\n")

    # 3. Register student
    print("3. Register student...")
    student_data = {
        "email": f"student_{int(time.time())}@test.com",
        "password": "TestPass123!",
        "full_name": "Test Student",
        "role": "student",
        "department": "Orthopaedics",
        "institution": "Test Medical College",
    }
    r = requests.post(f"{BASE}/api/auth/register", json=student_data)
    if r.status_code == 201:
        student_user = r.json()
        print(f"   Created: {student_user['email']} (id: {student_user['id']})\n")
    else:
        print(f"   Response: {r.status_code} - {r.text}\n")

    # 4. Login as student
    print("4. Login as student...")
    r = requests.post(f"{BASE}/api/auth/login", json={
        "email": student_data["email"],
        "password": student_data["password"],
    })
    if r.status_code == 200:
        token_data = r.json()
        student_token = token_data["access_token"]
        print(f"   Token obtained: {student_token[:20]}...\n")
    else:
        print(f"   Login failed: {r.status_code} - {r.text}")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {student_token}"}

    # 5. Get current user
    print("5. Get current user (student)...")
    r = requests.get(f"{BASE}/api/auth/me", headers=headers)
    assert r.status_code == 200, f"Failed: {r.status_code}"
    me = r.json()
    print(f"   Logged in as: {me['full_name']} ({me['role']})\n")

    # 6. Create submission with file upload
    print("6. Create submission with MEDICINE 5 data...")
    excel_path = FIXTURES_DIR / "medicine5_data.xlsx"
    discussion_path = FIXTURES_DIR / "DISCUSSION.docx"

    if not excel_path.exists():
        print(f"   SKIP: fixture not found at {excel_path}\n")
    else:
        files = [
            ("files", ("medicine5_data.xlsx", open(excel_path, "rb"), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")),
            ("files", ("DISCUSSION.docx", open(discussion_path, "rb"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ]
        data = {
            "title": "Outcome of TENS for Paediatric Long Bone Fractures",
            "intent": "Evaluate outcome of titanium elastic nail fixation for paediatric fractures including demographics, injury patterns, operative outcomes, and complications.",
        }
        r = requests.post(f"{BASE}/api/student/submissions", headers=headers, data=data, files=files)
        if r.status_code in (200, 201):
            submission = r.json()
            submission_id = submission["id"]
            print(f"   Created submission: {submission_id}")
            print(f"   Files: {len(submission.get('files', []))}\n")
        else:
            print(f"   Response: {r.status_code} - {r.text}\n")
            submission_id = None

    # 7. List submissions
    print("7. List student submissions...")
    r = requests.get(f"{BASE}/api/student/submissions", headers=headers)
    if r.status_code == 200:
        subs = r.json()
        print(f"   Found {len(subs)} submission(s)\n")
    else:
        print(f"   Response: {r.status_code} - {r.text}\n")

    # 8. Trigger analysis (if submission was created)
    if submission_id:
        print("8. Trigger analysis pipeline...")
        r = requests.post(f"{BASE}/api/analysis/trigger/{submission_id}", headers=headers)
        if r.status_code in (200, 202):
            job = r.json()
            job_id = job.get("job_id") or job.get("id")
            print(f"   Analysis job started: {job_id}\n")

            # 9. Poll for status
            print("9. Polling analysis status...")
            for i in range(10):
                time.sleep(2)
                r = requests.get(f"{BASE}/api/analysis/status/{job_id}", headers=headers)
                if r.status_code == 200:
                    status = r.json()
                    progress = status.get("progress", 0)
                    state = status.get("status", "unknown")
                    print(f"   [{i+1}] Status: {state}, Progress: {progress}%")
                    if state in ("COMPLETED", "FAILED"):
                        break
                else:
                    print(f"   [{i+1}] Poll failed: {r.status_code}")
                    break
            print()
        else:
            print(f"   Response: {r.status_code} - {r.text}\n")

    # 10. Login as professor and check
    print("10. Login as professor...")
    r = requests.post(f"{BASE}/api/auth/login", json={
        "email": prof_data["email"],
        "password": prof_data["password"],
    })
    if r.status_code == 200:
        prof_token = r.json()["access_token"]
        prof_headers = {"Authorization": f"Bearer {prof_token}"}
        print(f"   Token obtained.\n")

        print("11. List professor's students...")
        r = requests.get(f"{BASE}/api/professor/students", headers=prof_headers)
        print(f"   Response: {r.status_code} - {r.json() if r.status_code == 200 else r.text}\n")
    else:
        print(f"   Login failed: {r.status_code}\n")

    # 12. Admin metrics
    print("12. Admin dashboard metrics...")
    r = requests.get(f"{BASE}/api/admin/metrics", headers=headers)
    print(f"   Response: {r.status_code} - {r.json() if r.status_code == 200 else r.text}\n")

    print("=" * 60)
    print("TEST SCRIPT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
