"""
Scene Room Mapper — maps 3D waypoints to room types using Matterport3D .house files.
Used to enrich train/val_seen annotation context with actual room sequences.
"""
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

SCENE_DIR = Path("/mnt/nvme0/vln_habitat/habitat_data/scene_datasets/mp3d")

# Matterport3D room type codes → human-readable names
ROOM_CODES = {
    'a': 'bathroom',
    'b': 'bedroom',
    'c': 'closet',
    'd': 'dining room',
    'e': 'entryway',
    'f': 'family room',
    'g': 'garage',
    'h': 'hallway',
    'i': 'office',
    'j': 'laundry room',
    'k': 'kitchen',
    'l': 'living room',
    'm': 'meeting room',
    'n': 'lounge',
    'o': 'office',
    'p': 'porch',
    'r': 'rec room',
    's': 'staircase',
    't': 'bathroom',
    'u': 'utility room',
    'v': 'TV room',
    'w': 'gym',
    'x': 'outdoor',
    'y': 'balcony',
    'z': 'room',
    'B': 'bar',
    'C': 'classroom',
    'D': 'dining area',
    'S': 'sitting room',
    'Z': 'room',
}


def parse_house_file(scene_id: str) -> List[dict]:
    """Parse .house file and return list of rooms with type and bbox."""
    house_path = SCENE_DIR / scene_id / f"{scene_id}.house"
    if not house_path.exists():
        return []

    rooms = []
    with open(house_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != 'R':
                continue
            # R room_idx level 0 0 code bbox_center bbox_min bbox_max
            # Format: R idx lev 0 0 code cx cy cz xmin ymin zmin xmax ymax zmax ...
            if len(parts) < 12:
                continue
            try:
                code = parts[5]
                cx, cy, cz = float(parts[6]), float(parts[7]), float(parts[8])
                xmin, ymin, zmin = float(parts[9]), float(parts[10]), float(parts[11])
                xmax, ymax, zmax = float(parts[12]), float(parts[13]), float(parts[14])
                room_name = ROOM_CODES.get(code, 'room')
                rooms.append({
                    'code': code,
                    'name': room_name,
                    'bbox': (xmin, ymin, zmin, xmax, ymax, zmax),
                    'center': (cx, cy, cz),
                })
            except (ValueError, IndexError):
                continue
    return rooms


@lru_cache(maxsize=128)
def get_scene_rooms(scene_id: str) -> tuple:
    """Cached room list for a scene."""
    return tuple(parse_house_file(scene_id))


def point_in_room(pos: Tuple[float, float, float], room: dict, margin: float = 0.3) -> bool:
    """Check if a 3D point is inside a room's bounding box (with margin)."""
    x, y, z = pos
    xmin, ymin, zmin, xmax, ymax, zmax = room['bbox']
    return (xmin - margin <= x <= xmax + margin and
            ymin - margin <= y <= ymax + margin and
            zmin - margin <= z <= zmax + margin)


def closest_room(pos: Tuple[float, float, float], rooms: List[dict]) -> Optional[dict]:
    """Find the room whose center is closest to the position."""
    if not rooms:
        return None
    def dist(r):
        cx, cy, cz = r['center']
        x, y, z = pos
        return (x-cx)**2 + (y-cy)**2 + (z-cz)**2
    return min(rooms, key=dist)


def waypoint_to_room(pos: List[float], rooms: List[dict]) -> str:
    """Map a waypoint position to a room name."""
    t = tuple(pos)
    # Try exact containment first
    for room in rooms:
        if point_in_room(t, room):
            return room['name']
    # Fall back to closest room
    r = closest_room(t, rooms)
    return r['name'] if r else 'room'


def get_path_room_sequence(
    reference_path: List[List[float]],
    scene_id: str,
) -> List[str]:
    """
    Extract room sequence from path waypoints.
    Returns deduplicated list of room names along the path.
    """
    rooms = list(get_scene_rooms(scene_id))
    if not rooms:
        return []

    path_rooms = []
    prev_room = None
    for pos in reference_path:
        room_name = waypoint_to_room(pos, rooms)
        if room_name != prev_room:
            path_rooms.append(room_name)
            prev_room = room_name
    return path_rooms


def build_scene_context(
    reference_path: List[List[float]],
    scene_id_raw: str,
) -> dict:
    """
    Build scene-enriched context for train annotation.
    Returns dict with room_sequence, start_room, end_room.
    """
    scene_id = scene_id_raw.split('/')[-2] if '/' in scene_id_raw else scene_id_raw.replace('.glb', '')
    room_seq = get_path_room_sequence(reference_path, scene_id)

    if not room_seq:
        return {"room_sequence": "", "start_room": "room", "end_room": "room"}

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for r in room_seq:
        if r not in seen:
            deduped.append(r)
            seen.add(r)

    start_room = deduped[0]
    end_room = deduped[-1]
    room_seq_str = " → ".join(deduped) if len(deduped) > 1 else deduped[0]

    return {
        "room_sequence": room_seq_str,
        "start_room": start_room,
        "end_room": end_room,
        "all_rooms": deduped,
    }
