"""SQLite persistence. Connections belong to the calling thread."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "zoneguard.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, created TEXT NOT NULL, status TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, job_id TEXT NOT NULL, seconds REAL NOT NULL, zone_id TEXT NOT NULL, class TEXT NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS events_job ON events(job_id,seconds);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=20)
        try:
            db.row_factory = sqlite3.Row
            with db:
                yield db
        finally:
            db.close()

    def save_job(self, job):
        with self.connect() as db:
            db.execute("INSERT INTO jobs VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,data=excluded.data",
                       (job["id"], job["created"], job["status"], json.dumps(job, ensure_ascii=False)))

    def jobs(self):
        with self.connect() as db:
            return [json.loads(row["data"]) for row in db.execute("SELECT data FROM jobs ORDER BY created DESC LIMIT 100")]

    def job(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def add_event(self, job_id, event):
        with self.connect() as db:
            db.execute("INSERT INTO events VALUES(?,?,?,?,?,?)", (event["id"], job_id, event["confirmed_seconds"], event["zone_id"], event["class"], json.dumps(event, ensure_ascii=False)))

    def events(self, job_id):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT data FROM events WHERE job_id=? ORDER BY seconds,id", (job_id,))]

    def recover(self):
        # Pending jobs are never silently treated as complete after a crash.
        with self.connect() as db:
            pending = [json.loads(row[0]) for row in db.execute("SELECT data FROM jobs WHERE status IN ('queued','running')")]
        for job in pending:
            job.update(status="error", error="Server đã dừng giữa phiên. Các bằng chứng đã lưu được giữ lại; hãy tạo phiên mới.")
            self.save_job(job)

    def delete_job(self, job_id):
        with self.connect() as db:
            db.execute("DELETE FROM events WHERE job_id=?", (job_id,))
            db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
