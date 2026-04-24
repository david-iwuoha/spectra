from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
from pathlib import Path

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

    # ── Phase D: Wind + Drift Intelligence ────────────────────────────────
    wind_speed_ms = Column(Float, nullable=True)
    wind_direction_deg = Column(Float, nullable=True)
    wind_u = Column(Float, nullable=True)
    wind_v = Column(Float, nullable=True)

    sar_validity = Column(String, nullable=True)  # valid|too_low|too_high|borderline|unavailable|error
    sar_validity_detail = Column(String, nullable=True)

    lookalike_wind_risk = Column(String, nullable=True)  # high|medium|low|unknown
    lookalike_wind_note = Column(String, nullable=True)

    drift_bearing_deg = Column(Float, nullable=True)
    drift_speed_ms = Column(Float, nullable=True)
    drift_24h_km = Column(Float, nullable=True)

    drift_geojson = Column(Text, nullable=True)  # JSON string

    wind_fetched_at = Column(String, nullable=True)  # ISO-8601
    wind_data_source = Column(String, nullable=True)
    # ──────────────────────────────────────────────────────────────────────


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