from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
from pathlib import Path

# SQLite — zero config, works everywhere
DB_PATH = Path("data/spectra.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class WatchZone(Base):
    __tablename__ = "watch_zones"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    client_name = Column(String, nullable=False)
    priority = Column(String, default="medium")
    polygon_geojson = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    active = Column(Boolean, default=True)
    description = Column(Text, nullable=True)


class Detection(Base):
    __tablename__ = "detections"

    id = Column(String, primary_key=True)
    watch_zone_id = Column(String, nullable=True)
    scene = Column(String, nullable=True)
    detected_at = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float, default=0.0)
    area_km2 = Column(Float, default=0.0)
    spill_pixels = Column(Integer, default=0)
    polygon_geojson = Column(Text, nullable=True)
    detected = Column(Boolean, default=False)
    alert_sent = Column(Boolean, default=False)
    alert_recipients = Column(Text, nullable=True)
    wind_speed = Column(Float, nullable=True)
    wind_reliable = Column(Boolean, nullable=True)
    optical_confirmed = Column(Boolean, nullable=True)
    lookalike_score = Column(Float, nullable=True)
    status = Column(String, default="complete")


class AlertLog(Base):
    __tablename__ = "alert_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    detection_id = Column(String, nullable=False)
    recipient = Column(String, nullable=False)
    sent_at = Column(DateTime, default=datetime.utcnow)
    success = Column(Boolean, default=True)


def init_db():
    Base.metadata.create_all(bind=engine)
    print(f"Database ready at {DB_PATH}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
