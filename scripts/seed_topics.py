import os
import sys

# Add the project root directory to Python's module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import uuid

from sqlalchemy import delete

from app.core.database import AsyncSessionLocal
from app.models.topic import Topic

TOPICS = [
    {
        "name": "Mathematics",
        "slug": "mathematics",
        "description": "Core math curriculum",
        "grade_level": 10,
        "order_index": 1,
        "subtopics": [
            {"name": "Algebra", "slug": "algebra", "grade_level": 9},
            {"name": "Calculus", "slug": "calculus", "grade_level": 11},
            {"name": "Geometry", "slug": "geometry", "grade_level": 10},
            {"name": "Statistics", "slug": "statistics", "grade_level": 11},
        ],
    },
    {
        "name": "Physics",
        "slug": "physics",
        "description": "Classical and modern physics",
        "grade_level": 11,
        "order_index": 2,
        "subtopics": [
            {"name": "Mechanics", "slug": "mechanics", "grade_level": 11},
            {"name": "Thermodynamics", "slug": "thermodynamics", "grade_level": 12},
            {"name": "Electromagnetism", "slug": "electromagnetism", "grade_level": 12},
        ],
    },
    {
        "name": "Chemistry",
        "slug": "chemistry",
        "description": "General chemistry",
        "grade_level": 10,
        "order_index": 3,
        "subtopics": [
            {"name": "Organic Chemistry", "slug": "organic-chemistry", "grade_level": 11},
            {"name": "Inorganic Chemistry", "slug": "inorganic-chemistry", "grade_level": 10},
        ],
    },
    {
        "name": "Biology",
        "slug": "biology",
        "description": "Life sciences",
        "grade_level": 9,
        "order_index": 4,
        "subtopics": [
            {"name": "Cell Biology", "slug": "cell-biology", "grade_level": 9},
            {"name": "Genetics", "slug": "genetics", "grade_level": 10},
            {"name": "Ecology", "slug": "ecology", "grade_level": 9},
        ],
    },
    {
        "name": "Computer Science",
        "slug": "computer-science",
        "description": "Programming and CS fundamentals",
        "grade_level": 10,
        "order_index": 5,
        "subtopics": [
            {"name": "Data Structures", "slug": "data-structures", "grade_level": 11},
            {"name": "Algorithms", "slug": "algorithms", "grade_level": 11},
            {"name": "Machine Learning", "slug": "machine-learning", "grade_level": 12},
        ],
    },
]


async def seed():
    async with AsyncSessionLocal() as db:
        print("Clearing old messed up topics...")
        # 1. Delete subtopics first (to avoid foreign key errors)
        await db.execute(delete(Topic).where(Topic.parent_id.isnot(None)))
        # 2. Delete parent topics
        await db.execute(delete(Topic).where(Topic.parent_id.is_(None)))
        await db.commit()

        print("Inserting perfectly organized topics...")
        for t_data in TOPICS:
            subtopics_data = t_data.pop("subtopics", [])
            parent = Topic(id=uuid.uuid4(), **t_data)
            db.add(parent)
            await db.flush()

            for s in subtopics_data:
                sub = Topic(
                    id=uuid.uuid4(), parent_id=parent.id, order_index=0, description="", **s
                )
                db.add(sub)

        await db.commit()
    print("Topics seeded successfully!")


if __name__ == "__main__":
    asyncio.run(seed())
