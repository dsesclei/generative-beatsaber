import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from rich import print as rprint
from rich.progress import Progress
from sqlitedict import SqliteDict


def fetch_maps(before=None):
    params = {
        "pageSize": 100,
        "automapper": False,
        "sort": "FIRST_PUBLISHED",
        **({"before": before} if before else {}),
    }
    rprint(f"[green]Fetching before {before}")
    response = requests.get("https://api.beatsaver.com/maps/latest", params=params)
    response.raise_for_status()
    return response.json().get("docs", [])


def update_database(db, maps):
    if not maps:
        return 0, 0
    db_maps = db.get("maps", {})
    new_maps = {map_data["id"]: map_data for map_data in maps if map_data["id"] not in db_maps}
    db_maps.update(new_maps)
    db["maps"] = db_maps
    db["before"] = min(maps, key=lambda m: m["uploaded"])["uploaded"]
    db.commit()
    return len(new_maps), len(db_maps)


def fetch_metadata(db):
    new_maps = -1
    while new_maps != 0:
        maps = fetch_maps(db.get("before", None))
        new_maps, total_maps = update_database(db, maps)
        rprint(f"[green]Fetched metadata for {new_maps} maps, {total_maps} total")
        time.sleep(0.1)


def download_map(zip_dir, beatmap):
    latest_version = max(
        beatmap["versions"],
        key=lambda v: datetime.fromisoformat(v["createdAt"].rstrip("Z")),
    )
    try:
        response = requests.get(latest_version["downloadURL"])
        response.raise_for_status()
        with open(os.path.join(zip_dir, f"{beatmap['id']}.zip"), "wb") as f:
            f.write(response.content)
    except Exception:
        pass


def download_zips(db, data_path):
    zip_dir = data_path / "zips"
    os.makedirs(zip_dir, exist_ok=True)

    maps = list(db.get("maps", {}).values())
    maps.sort(key=lambda m: m["stats"]["score"], reverse=True)
    queue = [m for m in maps if not os.path.exists(os.path.join(zip_dir, f"{m['id']}.zip"))]

    rprint(f"[green]{len(maps)} total, {len(queue)} remaining")
    with Progress() as progress:
        task = progress.add_task("[cyan]Downloading maps...", total=len(queue))
        for beatmap in queue:
            progress.update(
                task,
                description=f"[green]Downloading {beatmap['name']} ({beatmap['stats']['score']}))",
            )
            download_map(zip_dir, beatmap)
            progress.update(task, advance=1)
            time.sleep(0.1)


def main():
    if len(sys.argv) < 2:
        rprint("[red]No command specified")
        return

    command, data_path = sys.argv[1], sys.argv[2]
    data_path = Path(data_path).resolve()
    with SqliteDict(data_path / "metadata.sqlite", autocommit=True) as db:
        if command == "metadata":
            fetch_metadata(db)
        elif command == "download":
            download_zips(db, data_path)
        else:
            rprint(f"[red]Unknown choice: {command}")


if __name__ == "__main__":
    main()
