import asyncio
import os
import sys
import uuid

# Add the project root directory to Python's module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text

from app.core.database import AsyncSessionLocal

TOPICS = [
    {
        "name": "Mathematics",
        "slug": "mathematics",
        "description": "Core math",
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
        "description": "Physics",
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
        "description": "Chemistry",
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
        "description": "Biology",
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
        "description": "CS",
        "grade_level": 10,
        "order_index": 5,
        "subtopics": [
            {"name": "Data Structures", "slug": "data-structures", "grade_level": 11},
            {"name": "Algorithms", "slug": "algorithms", "grade_level": 11},
            {"name": "Machine Learning", "slug": "machine-learning", "grade_level": 12},
        ],
    },
]


async def reset_topics():
    async with AsyncSessionLocal() as db:
        print("1. Wiping old topics from the database...")
        # Using "topics" table explicitly
        await db.execute(text("DELETE FROM topics WHERE parent_id IS NOT NULL;"))
        await db.execute(text("DELETE FROM topics WHERE parent_id IS NULL;"))

        print("2. Inserting correct topics...")
        for t_data in TOPICS:
            parent_id = str(uuid.uuid4())
            await db.execute(
                text(
                    "INSERT INTO topics (id, name, slug, description, grade_level, order_index) "
                    "VALUES (:id, :name, :slug, :desc, :grade, :idx)"
                ),
                {
                    "id": parent_id,
                    "name": t_data["name"],
                    "slug": t_data["slug"],
                    "desc": t_data["description"],
                    "grade": t_data["grade_level"],
                    "idx": t_data["order_index"],
                },
            )

            for s in t_data["subtopics"]:
                sub_id = str(uuid.uuid4())
                await db.execute(
                    text(
                        "INSERT INTO topics (id, parent_id, name, slug, description, grade_level, order_index) "
                        "VALUES (:id, :pid, :name, :slug, '', :grade, 0)"
                    ),
                    {
                        "id": sub_id,
                        "pid": parent_id,
                        "name": s["name"],
                        "slug": s["slug"],
                        "grade": s["grade_level"],
                    },
                )

        await db.commit()
        print("Done! Database is perfectly organized.")


if __name__ == "__main__":
    asyncio.run(reset_topics())
