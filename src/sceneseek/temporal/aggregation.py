from __future__ import annotations

from sceneseek.domain import SearchResult
from sceneseek.retrieval.index import Neighbor


def aggregate_neighbors(
    neighbors: list[Neighbor],
    *,
    limit: int,
    merge_gap: float = 2.0,
    max_moments_per_video: int = 2,
) -> list[SearchResult]:
    images: list[SearchResult] = []
    video_groups: dict[str, list[Neighbor]] = {}
    for neighbor in neighbors:
        item = neighbor.item
        if item.media_type == "image":
            images.append(
                SearchResult(
                    media_id=item.media_id,
                    media_type="image",
                    score=neighbor.score,
                    coarse_score=neighbor.score,
                    path=item.path,
                    width=item.width,
                    height=item.height,
                )
            )
        else:
            video_groups.setdefault(item.media_id, []).append(neighbor)

    videos: list[SearchResult] = []
    for media_id, group in video_groups.items():
        intervals: list[SearchResult] = []
        for neighbor in sorted(group, key=lambda value: value.item.start_sec or 0):
            item = neighbor.item
            start = float(item.start_sec or 0)
            end = float(item.end_sec or start)
            if intervals and start <= float(intervals[-1].end_sec or 0) + merge_gap:
                current = intervals[-1]
                current.end_sec = max(float(current.end_sec or 0), end)
                if neighbor.score > current.score:
                    current.score = neighbor.score
                    current.coarse_score = neighbor.score
                    current.thumbnail_sec = (start + end) / 2
            else:
                intervals.append(
                    SearchResult(
                        media_id=media_id,
                        media_type="video",
                        score=neighbor.score,
                        coarse_score=neighbor.score,
                        path=item.path,
                        width=item.width,
                        height=item.height,
                        duration=item.duration,
                        start_sec=start,
                        end_sec=end,
                        thumbnail_sec=(start + end) / 2,
                    )
                )
        videos.extend(
            sorted(intervals, key=lambda value: value.score, reverse=True)[:max_moments_per_video]
        )

    combined = images + videos
    combined.sort(key=lambda value: value.score, reverse=True)
    return combined[:limit]
