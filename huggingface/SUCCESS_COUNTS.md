# Successful-run counts

The shared backend records Object and Scene generation separately, only after
their GPU calls return successfully. The Scene Space embeds the same backend,
so a scene generated through either entry point counts once for Scene. Mesh
extraction, camera previews, downloads, visits, and failed requests do not count.

Badges at the right edge of each title row show the last 30 days' total and today's
successful runs. The **Daily success history** panel contains the daily table
for today and the preceding 29 days. Dates use
`America/Los_Angeles`. History before instrumentation starts is unavailable;
the total includes the available recorded activity without waiting for a full month.

The private dataset `CaiYuanhao/space-success-stats` stores the aggregate counts.
Add a long-lived fine-grained token with read/write permission only for that
dataset as the `SPACE_STATS_TOKEN` Secret on **DiffusionGS-Object**. The static
Scene Space requires no secret. `SPACE_STATS_REPO` can optionally select another
initialized private dataset. Never put tokens in repository files. The expiring
interactive OAuth login is not used as a runtime credential.

For a differently named Secret, set the non-sensitive `SPACE_STATS_TOKEN_NAME`
Variable to that name. Existing deployments use `cyh_track_liuliang`.

Recording runs in the CPU caller after the ZeroGPU worker returns, avoiding
forked-process state or temporary-disk persistence. Optimistic commits and event
IDs handle concurrent writers and uncertain commit responses without double
counting. No images, prompts, IP addresses, or user IDs are stored. An outage
does not prevent returning generated results; failed counter writes produce an
anonymous `SUCCESS_COUNT_NOT_SAVED` log entry for recovery after retries.

Badges load on page opening and update automatically after successful generation.
Open **Daily success history** and click **Refresh counts** to reload from storage. Cached remote reads expire after 30 seconds.

```sh
python -m unittest discover -s tests -p test_space_success_stats.py -v
```
