# ADR 0009: One ingestion path for level-1 and level-2 bars

Status: accepted

Extends ADR 0008 (frozen interface item 2: "`hftf data pull-taq` and the manifests — the only
way data enters the repo").

## Context
The TAQ pull is level 1 and needs a WRDS subscription. The project also has to be able to run
on a level-2 book file from a venue or an employer, whose format is not known in advance, and
on whatever grid a horizon group picks (30 s, 1 min, 5 min). Waiting for the data before
building the path would leave the three groups blocked, and a second parallel ingestion path
would mean two bar schemas to keep in step.

## Decision
- **The import wizard is the second ingestion path**, not a new data source per vendor. A file
  is described by an `ImportMapping` (which column is what), and that mapping is saved next to
  the imported Parquet so a re-import is one click.
- **`ImportMapping` carries the book**: `book_levels` plus `book_columns` mapping the canonical
  `bid_px_1`, `bid_sz_1`, `ask_px_1`, `ask_sz_1`, … to source columns. `guess_mapping` finds
  them under the common spellings (`bid_px_1`, `bid_price_1`, `bid_size_1`, …) and stops at the
  first incomplete level.
- **Everything mechanically derivable is derived, nothing mapped is overwritten**: level 1 of
  the book gives the NBBO fields, the quotes give the mid, the spread and the microprice, and
  the mid gives `close`. A level-1 file therefore needs no price column at all. A file that
  maps both level 1 and the best quote must agree on them, or the import is refused.
- **Bar-end labelling is enforced at the boundary, not assumed.** `ImportMapping.bar_label`
  says where the file's timestamps sit and a start-labelled intraday file is shifted by one bar
  on import, so everything on disk is right-closed. The dataset then carries
  `bar_label = "end"` and `price_basis`, which is what `ml` and the quality checks read.
- **The bar size is checked against the horizon**, because an imported file's frequency is
  whatever the mapping declared. Labelling 1 min bars as a 5 s grid used to drop every row on
  the `bar_gap` label rule; it now raises and names both frequencies.
- **Window lengths that are counted in bars are parameters.** `basic.vol_windows` defaults to
  `[20, 100]`, which is eight minutes on a 5 s grid and more than a session on a 5 min grid.

## Consequences
- A group can develop against `synthetic`/`quotes`, switch to pulled TAQ bars, and later to an
  employer book file, by changing `data.source` / `manifest` in its experiment config. Nothing
  in labels, CV, training, evaluation or the economic test changes.
- Level-2 feature work has a schema to target before the data arrives
  (`ml/features/book.py`, `core/columns.book_columns`).
- Frozen interface item 2 now reads: TAQ pulls **and** file imports, both producing the same
  canonical bar; the schema (item 1) is what is actually frozen.
- The importer does not reconstruct an order book from a diff/message stream. A venue that
  ships messages rather than snapshots needs a replay step first, which stays out of scope
  until a format is known (ADR 0008, level-2 placeholder).
