# Oblique homing failure — 2026-09-23 16:52-16:56 (table, PID 64676)

Three go_home attempts from a ~60°+ oblique angle all failed:
attempt 1 `too_close`, attempt 2 `lost`, attempt 3 `alignment`.
Log: `spark-homing-fail-oblique.log` (140 lines).

## What the data shows

- Plane measurement is REPRODUCIBLE, not mirrored-noise:
  97.4° / 98.6° / 102.9° dock-plane-vs-heading across the three attempts
  (views only 11-12° apart — weak separation but consistent results).
- Waypoint sidle steps moved the marker lateral offset the WRONG way:
  x -63 → -86 → -111mm over three steps, z 250 → 197.
  The ±31° turn pairs in the log are waypoint_step's face-waypoint/restore
  turns; the 30mm drives DO execute (drive "errors" are stop-path noise).
- Attempt 1 drifted into plane_distance < 175mm while still oblique
  (`too_close`): at oblique angles the 180-200mm standoff is reached
  radially long before the heading aligns, so no waypoint stepping happens
  and only marker-centering turns remain — she orbits without converging.
- Frontal approaches converge (morning plane7/entry20 success); oblique
  ones cannot. The 5° final-entry gate correctly refuses.

## The fix direction (next session)

When the measured dock normal is > ~30° off her heading:
1. Retreat to ~350mm RADIAL distance (oblique too_close needs more than
   2×40mm — the angle, not the distance, is the problem).
2. Arc reposition: guarded 30mm steps that circle toward the dock normal
   line while keeping the marker in view, re-measuring the plane each
   quarter-arc; stop when |normal| < ~25° and hand off to the existing
   frontal approach.
3. Consider wider plane-view separation (±10° today; ±15-20° gives the
   ambiguity resolver more leverage).

Also queued: log drive on_error's (id, status, ts) fully — today's logs
only kept the id.
