"""Join FMA tracks.csv into the tracks table.

tracks.csv has a quirky three-row header: a group row (album/artist/track/set),
a subfield row (title/name/genre_top/...), then a scaffolding row. Columns are
resolved by (group, subfield) name rather than fixed index. Uses stdlib csv --
pandas would be a heavy dependency for a one-shot join on a RAM-tight machine.
"""
import csv

WANTED = {
    "title": ("track", "title"),
    "artist": ("artist", "name"),
    "genre": ("track", "genre_top"),
    "duration": ("track", "duration"),
}


def read_csv(csv_path):
    """Yield (track_id, title, artist, genre, duration)."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        groups, subs = next(reader), next(reader)
        next(reader)  # scaffolding row

        idx = {}
        for field, key in WANTED.items():
            matches = [i for i in range(len(groups)) if (groups[i], subs[i]) == key]
            if not matches:
                raise RuntimeError(f"column {key} not found in {csv_path}")
            idx[field] = matches[0]

        for row in reader:
            if not row or not row[0]:
                continue
            def get(field):
                v = row[idx[field]].strip()
                return v or None  # genre_top is blank for many tracks -> NULL
            duration = get("duration")
            yield (
                int(row[0]), get("title"), get("artist"), get("genre"),
                float(duration) if duration else None,
            )


def load(conn, csv_path):
    """Idempotent metadata join. Separate from embedding, so a metadata failure
    never costs GPU time."""
    updated = 0
    for tid, title, artist, genre, duration in read_csv(csv_path):
        cur = conn.execute(
            "UPDATE tracks SET title=?, artist=?, genre=?, duration=? WHERE track_id=?",
            (title, artist, genre, duration, tid),
        )
        updated += cur.rowcount
    conn.commit()
    return updated
