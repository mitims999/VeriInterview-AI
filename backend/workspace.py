"""
Organization workspace listings: the data the dashboard and navigation are built from.

A real organization sees only its own candidates/interviews (queried by organization_id).
The demo organization sees the bundled, entirely fictional synthetic dataset (data/synthetic),
read straight from disk - it never touches database rows that belong to a real organization.
"""
from __future__ import annotations
import json
import os
import threading
from datetime import timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.auth.deps import is_demo, require_user
from backend.config import DATA_DIR
from backend.database import models
from backend.database.session import get_db

router = APIRouter(prefix="/api/workspace", tags=["workspace"])


# Scores of the fictional demo records, produced by the real pipeline. The first analysis in a process
# loads the embedding model (tens of seconds), so scores are warmed in a background thread at startup and
# the listing never waits for them: until a record is scored its risk fields are simply null.
_DEMO_RISK: dict[str, tuple[float, str]] = {}
_warm_lock = threading.Lock()


def warm_demo_scores() -> None:
    from backend.modules.pipeline import run_full_analysis
    for fname in sorted(os.listdir(DATA_DIR)) if os.path.isdir(DATA_DIR) else []:
        key = fname[:-5]
        if not fname.endswith(".json") or key in _DEMO_RISK:
            continue
        try:
            with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as f:
                a = run_full_analysis(json.load(f))["assessment"]
            with _warm_lock:
                _DEMO_RISK[key] = (round(a["risk_score"], 1), a["risk_level"])
        except Exception:   # a bad fixture must not take the server down; the row just stays unscored
            continue


def start_demo_warmup() -> None:
    threading.Thread(target=warm_demo_scores, name="demo-score-warmup", daemon=True).start()


def _demo_risk(key: str) -> tuple[float | None, str | None]:
    with _warm_lock:
        return _DEMO_RISK.get(key, (None, None))


def save_report(db: Session, user: models.User, result: dict, mode: str) -> int | None:
    """Keep an analysis for the user's organization. The demo organization stores nothing."""
    if is_demo(user):
        return None
    a = result.get("assessment", {})
    row = models.SavedReport(organization_id=user.organization_id, created_by_user_id=user.id,
                             candidate_name=a.get("candidate_name") or "Unknown", mode=mode,
                             risk_score=a.get("risk_score"), risk_level=a.get("risk_level"), result_json=result)
    db.add(row)
    db.commit()
    return row.id


def _demo_records() -> list[dict]:
    out = []
    if not os.path.isdir(DATA_DIR):
        return out
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as f:
            rec = json.load(f)
        key = fname[:-5]
        score, level = _demo_risk(key)
        out.append({
            "id": f"demo:{key}", "candidate_name": rec.get("candidate_name", key), "status": "COMPLETED",
            "created_at": None, "question_count": len(rec.get("qa_list", [])), "source": "demo",
            "scenario_label": rec.get("scenario_label", ""), "risk_score": score, "risk_level": level,
        })
    return out


def _iso(dt):
    return dt.replace(tzinfo=timezone.utc).isoformat() if dt else None


@router.get("/interviews")
def list_interviews(user: models.User = Depends(require_user), db: Session = Depends(get_db)):
    if is_demo(user):
        return _demo_records()
    org = user.organization_id
    interviews = (db.query(models.Interview).filter(models.Interview.organization_id == org)
                  .order_by(models.Interview.created_at.desc()).all())
    reports = (db.query(models.SavedReport).filter(models.SavedReport.organization_id == org)
               .order_by(models.SavedReport.created_at.desc()).all())
    items = [{
        "id": f"interview:{r.id}", "candidate_name": r.candidate.name if r.candidate else "Unknown",
        "status": r.status, "created_at": _iso(r.created_at), "question_count": len(r.questions),
        "source": "interview", "scenario_label": "", "risk_score": None, "risk_level": None,
        "_at": r.created_at,
    } for r in interviews] + [{
        "id": f"report:{r.id}", "candidate_name": r.candidate_name, "status": "COMPLETED",
        "created_at": _iso(r.created_at), "question_count": len(r.result_json.get("qa_list", [])),
        "source": "report", "scenario_label": "", "risk_score": r.risk_score, "risk_level": r.risk_level,
        "_at": r.created_at,
    } for r in reports]
    items.sort(key=lambda i: i["_at"], reverse=True)
    for i in items:
        i.pop("_at")
    return items


@router.get("/reports/{report_id}")
def get_saved_report(report_id: int, user: models.User = Depends(require_user), db: Session = Depends(get_db)):
    row = (db.query(models.SavedReport)
           .filter(models.SavedReport.id == report_id, models.SavedReport.organization_id == user.organization_id)
           .first())
    if row is None:
        raise HTTPException(404, "Report not found")
    return row.result_json


@router.get("/candidates")
def list_candidates(user: models.User = Depends(require_user), db: Session = Depends(get_db)):
    if is_demo(user):
        return [{"id": r["id"], "name": r["candidate_name"], "interview_count": 1, "created_at": None,
                 "source": "demo"} for r in _demo_records()]
    counts = dict(db.query(models.Interview.candidate_id, func.count(models.Interview.id))
                  .filter(models.Interview.organization_id == user.organization_id)
                  .group_by(models.Interview.candidate_id).all())
    rows = (db.query(models.Candidate).filter(models.Candidate.organization_id == user.organization_id)
            .order_by(models.Candidate.created_at.desc()).all())
    out = [{"id": str(c.id), "name": c.name, "interview_count": counts.get(c.id, 0),
            "created_at": _iso(c.created_at), "source": "organization"} for c in rows]
    known = {c["name"] for c in out}
    for name, n, last in (db.query(models.SavedReport.candidate_name, func.count(models.SavedReport.id),
                                   func.max(models.SavedReport.created_at))
                          .filter(models.SavedReport.organization_id == user.organization_id)
                          .group_by(models.SavedReport.candidate_name).all()):
        if name not in known:   # candidates who only exist through saved analyses
            out.append({"id": f"name:{name}", "name": name, "interview_count": n, "created_at": _iso(last),
                        "source": "report"})
    return out


@router.get("/summary")
def summary(user: models.User = Depends(require_user), db: Session = Depends(get_db)):
    if is_demo(user):
        recs = _demo_records()
        return {"is_demo": True, "organization": user.organization.name, "interviews_total": len(recs),
                "candidates_total": len(recs), "completed": len(recs), "in_progress": 0}
    org = user.organization_id
    q = db.query(models.Interview).filter(models.Interview.organization_id == org)
    reports = db.query(models.SavedReport).filter(models.SavedReport.organization_id == org)
    total = q.count() + reports.count()
    completed = q.filter(models.Interview.status == "COMPLETED").count() + reports.count()
    names = {n for (n,) in db.query(models.Candidate.name).filter(models.Candidate.organization_id == org)}
    names |= {n for (n,) in reports.with_entities(models.SavedReport.candidate_name)}
    return {"is_demo": False, "organization": user.organization.name, "interviews_total": total,
            "candidates_total": len(names), "completed": completed, "in_progress": total - completed}
