"""Reserve API costs before dispatch; retain uncertain charges across restarts.

Amounts are integer billionths of a US dollar. This is an application allowance,
not a guarantee about the provider's invoice, taxes, or use by other clients.
"""
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sqlite3
import uuid


def dollars(value):
    try:
        amount = Decimal(str(value))
        if not amount.is_finite() or amount < 0:
            raise ValueError
        return int(amount * 1_000_000_000)
    except (InvalidOperation, ValueError, OverflowError):
        raise ValueError("Specify a finite, nonnegative monthly model budget") from None


class ApiBudget:
    def __init__(self, path, monthly_usd=0):
        self.path = Path(path)
        self.limit = dollars(monthly_usd)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS model_api_calls (
                id TEXT PRIMARY KEY, month TEXT NOT NULL, model TEXT NOT NULL,
                reserved INTEGER NOT NULL, charged INTEGER, state TEXT NOT NULL,
                prompt_tokens INTEGER, completion_tokens INTEGER)""")

    def connect(self):
        return sqlite3.connect(self.path, timeout=15)

    @staticmethod
    def month():
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def reserve(self, model, amount):
        if type(amount) is not int or amount <= 0:
            raise ValueError("Invalid model cost reservation")
        month, identifier = self.month(), str(uuid.uuid4())
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            used = db.execute("SELECT COALESCE(SUM(COALESCE(charged,reserved)),0) "
                              "FROM model_api_calls WHERE month=?", (month,)).fetchone()[0]
            if self.limit == 0 or used + amount > self.limit:
                raise RuntimeError("The monthly model API allowance is disabled or exhausted; no request was sent")
            db.execute("INSERT INTO model_api_calls(id,month,model,reserved,state) VALUES(?,?,?,?,?)",
                       (identifier, month, model, amount, "reserved"))
        return identifier

    def settle(self, identifier, amount, prompt_tokens, completion_tokens):
        if type(amount) is not int or amount < 0:
            raise ValueError("Invalid model usage cost")
        with closing(self.connect()) as db, db:
            row = db.execute("SELECT reserved,state FROM model_api_calls WHERE id=?", (identifier,)).fetchone()
            if not row or row[1] != "reserved":
                raise RuntimeError("The model usage reservation cannot be reconciled")
            # Never release a reservation for a malformed/oversized usage receipt.
            if amount > row[0]:
                raise RuntimeError("Provider usage exceeded the reserved allowance; review pricing before continuing")
            db.execute("UPDATE model_api_calls SET charged=?,state='settled',prompt_tokens=?,completion_tokens=? WHERE id=?",
                       (amount, prompt_tokens, completion_tokens, identifier))

    def status(self):
        with closing(self.connect()) as db:
            used, held = db.execute("SELECT COALESCE(SUM(COALESCE(charged,reserved)),0), "
                                   "COALESCE(SUM(CASE WHEN state='reserved' THEN 1 ELSE 0 END),0) "
                                   "FROM model_api_calls WHERE month=?", (self.month(),)).fetchone()
        return {"month_utc": self.month(), "limit_usd": self.limit / 1e9,
                "accounted_usd": used / 1e9, "uncertain_or_inflight_calls": held,
                "remaining_usd": max(0, self.limit - used) / 1e9}
