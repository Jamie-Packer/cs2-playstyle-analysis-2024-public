# CS2 Playstyles Dataset (2024) — Dataset Card

**Note:** This dataset card intentionally embeds the **authoritative Data Dictionary verbatim** below.  
- Primary dataset repo: https://github.com/Jamie-Packer/cs2-playstyle-dataset-2024  
- Local copy of dataset in this analysis repo: `data/processed/cs2_playstyles_2024.csv`  
- License: **CC BY 4.0** — `LICENSE-DATA` (copied verbatim from the dataset repo).

---
# Data Dictionary — CS2 Playstyle & Roles Dataset (2024)

**Row = player (unique by `steamid`).**  
`*_t` = T side, `*_ct` = CT side, `*_overall` = combined.

---

## Identifiers

| Column            | Type   | Description |
|-------------------|--------|-------------|
| `steamid`         | int64  | SteamID64 (primary key). |
| `player_name`     | string | Player handle. |
| `team_clan_name`  | string | Team name. |
| `map_count`       | int32  | Number of maps included for this player in the aggregation window. |

---

## Core playstyle features

All percentages are on **0–100** scale.

| Column              | Type   | Units/Range | Definition |
|---------------------|--------|-------------|------------|
| `tapd_t`            | float  | seconds     | **Time Alive Per Death** on T: mean time of death within rounds on T. |
| `tapd_ct`           | float  | seconds     | TAPD on CT. |
| `tapd_overall`      | float  | seconds     | Round-weighted mean across sides. |
| `oap_t`             | float  | % (0–100)   | **Opening Attempt %** on T: opening duels participated / team T rounds × 100. |
| `oap_ct`            | float  | % (0–100)   | OAP on CT. |
| `oap_overall`       | float  | % (0–100)   | Round-weighted across sides. |
| `podt_t`            | float  | % (0–100)   | **Proportion of Deaths Traded** on T within a 5s window. |
| `podt_ct`           | float  | % (0–100)   | PODT on CT. |
| `podt_overall`      | float  | % (0–100)   | Round-weighted across sides. |
| `pokt_t`            | float  | % (0–100)   | **Proportion of Kills which were Trades** on T within a 5s window. |
| `pokt_ct`           | float  | % (0–100)   | POKT on CT. |
| `pokt_overall`      | float  | % (0–100)   | Round-weighted across sides. |

> **Trading window:** 5 seconds; opening duels are the first kill event of a round.  
> All side/overall aggregations are **round-weighted across maps**.

---

## Positioning features — **team-relative ranking** (0.2–1.0)

Values are **ordinal ranks within a team on a given map**, scaled to the interval (0,1].  
With 5 players and no ties, the set is `{0.2, 0.4, 0.6, 0.8, 1.0}` where **higher = typically further from teammates**.

**Sampling rules (to capture structured setups):**  
- Pre-plant only; snapshots every **3s** starting **10s after freeze end**.  
- Alive-only; include a player only when their side still has **exactly four alive teammates** (full 5v5).  
- For each map and team: compute per-player **map-level means** from 3D distances, then convert to **team-relative ranks**.  
- Finally, compute **round-weighted means across maps** for T, CT, and overall.

| Column                  | Type  | Range   | Definition |
|-------------------------|-------|---------|------------|
| `adnt_rank_t`           | float | 0.2–1.0 | **Average Distance from Nearest Teammate** — team-relative rank on T (higher = more isolated). |
| `adnt_rank_ct`          | float | 0.2–1.0 | ADNT rank on CT. |
| `adnt_rank_overall`     | float | 0.2–1.0 | Round-weighted rank across sides. |
| `adat_rank_t`           | float | 0.2–1.0 | **Average Distance from Average Teammate** — team-relative rank on T (higher = further from centre). |
| `adat_rank_ct`          | float | 0.2–1.0 | ADAT rank on CT. |
| `adat_rank_overall`     | float | 0.2–1.0 | Round-weighted rank across sides. |

> **Interpretation caveat:** These are **team-relative ranking scores**, not statistical percentiles and not physical units; they enable comparisons of *tendency within team* across different maps without map-size bias.

---

## Roles

| Column         | Type   | Description |
|----------------|--------|-------------|
| `role_t`       | string | T-side role (Spacetaker/Lurker/Half-Lurker/AWPer). |
| `role_ct`      | string | CT-side role (Anchor/Rotator/Mixed/AWPer). |
| `role_overall` | string | Overall/primary role label. |

> **Source:** Roles are adapted from [**Harry Richards’ Positions Database**](https://public.tableau.com/app/profile/harry.richards4213/viz/OLDPositionsDatabaseArchived/PositionsDatabaseNER0cs) (credited; see README). Some entries may be missing where a mapping was not available. Some may have been attributed by **Jamie Packer** personally.

---

## Provenance & License

- **Source processing:** awpy-based demo parsing and custom feature engineering.  
- **License:** **CC BY 4.0** — you may use and adapt with attribution to **Jamie Packer**; please also credit [**Harry Richards’ Positions Database**](https://public.tableau.com/app/profile/harry.richards4213/viz/OLDPositionsDatabaseArchived/PositionsDatabaseNER0cs) when roles are used.
