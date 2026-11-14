#!/usr/bin/env python3
"""Prepare sparse clips from camera collector Canny SQLite shards."""
import argparse
import bisect
import csv
import json
import math
import sqlite3
import sys
import zlib
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import numpy as np

CLASSES = ['stand-and-walk', 'sit', 'lie-in-bed', 'lie-on-floor', 'stand-to-sit',
           'sit-to-stand', 'sit-to-lie-in-bed', 'lie-in-bed-to-sit',
           'lie-on-floor-to-stand', 'fall']


def open_shard(path, immutable=False):
    # Normal read-only mode keeps SQLite WAL visible. Immutable mode is only
    # safe for closed, standalone snapshots that have no SQLite sidecars.
    if immutable:
        sidecars = [Path(str(path) + suffix) for suffix in ('-wal', '-shm', '-journal')]
        existing = [str(sidecar) for sidecar in sidecars if sidecar.exists()]
        if existing:
            raise ValueError('Immutable snapshots must not have SQLite sidecars: ' + ', '.join(existing))
    uri = 'file:' + quote(str(path.resolve()).replace('\\', '/'), safe='/:') + '?mode=ro'
    if immutable:
        uri += '&immutable=1'
    db = sqlite3.connect(uri, uri=True)
    db.execute('PRAGMA query_only=ON')
    first = db.execute('SELECT sequence, mono_ns FROM frames ORDER BY sequence LIMIT 1').fetchone()
    last = db.execute('SELECT sequence, mono_ns FROM frames ORDER BY sequence DESC LIMIT 1').fetchone()
    if not first or not last:
        raise ValueError(f'No frames in {path}')
    return {'path': path, 'db': db, 'first': first, 'last': last}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sqlite', action='append', nargs='+', required=True, help='All relevant SQLite shards, including partial shards')
    p.add_argument('--annotations', required=True)
    p.add_argument('--output-dir', required=True, help='Must be empty or not exist')
    p.add_argument('--stride-seconds', type=float, default=60.0)
    p.add_argument('--max-gap-seconds', type=float, default=0.25)
    p.add_argument('--immutable-snapshots', action='store_true',
                   help='Read closed, standalone SQLite snapshots as immutable; rejects any WAL, SHM, or journal sidecar. Never use with active databases.')
    p.add_argument('--window-labeling', choices=['pure', 'center'], default='pure',
                   help='pure: all frames share one annotation interval; center: assign center label while requiring a fully annotated single-person window')
    p.add_argument('--max-clips', type=int, default=None, help='Optional limit for a small preparation smoke test')
    a = p.parse_args()
    if a.stride_seconds <= 0 or a.max_gap_seconds <= 0 or (a.max_clips is not None and a.max_clips <= 0):
        p.error('stride, max gap, and max clips must be positive')
    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        p.error('output directory must be empty; choose a new directory')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'clips').mkdir()
    shards = [open_shard(Path(x), immutable=a.immutable_snapshots) for group in a.sqlite for x in group]
    shards.sort(key=lambda s: s['first'][0])
    for left, right in zip(shards, shards[1:]):
        if left['last'][0] >= right['first'][0]:
            raise ValueError('SQLite shards have overlapping sequence ranges')
    with open(a.annotations, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2 or not {'start_frame', 'activity_label', 'presence_count', 'monotonic_ns'} <= set(rows[0]):
        raise ValueError('Annotations need start_frame, activity_label, presence_count, monotonic_ns and an END row')
    if rows[-1].get('original_label', '').strip().upper() != 'END':
        raise ValueError('Last annotation must be END, the exclusive endpoint')
    for r in rows:
        r['seq'] = int(r['start_frame'])
        r['mono'] = int(r['monotonic_ns'])
    if any(x['seq'] >= y['seq'] or x['mono'] >= y['mono'] for x, y in zip(rows, rows[1:])):
        raise ValueError('Annotation starts must have strictly increasing sequence and monotonic timestamps')
    # Representative fine IDs are internal compatibility fields only. Ground truth is coarse_label.
    mapping = {'stand_and_walk':('stand-and-walk','stand-and-walk'), 'sit':('sit','sit'),
               'stand_to_sit':('stand-to-sit','stand-to-sit'), 'sit_to_stand':('sit-to-stand','sit-to-stand'),
               'lie':('lie-in-bed','lie'), 'sit_to_lie':('sit-to-lie-in-bed','sit-to-lie'),
               'lie_to_sit':('lie-in-bed-to-sit','lie-to-sit')}
    # Frame IDs restart between recordings, so require matching acquisition timestamps.
    # Check every annotated boundary available in the supplied shards.
    matched_boundaries = 0
    missing_boundaries = 0
    for row in rows:
        matching_shards = [s for s in shards if s['first'][0] <= row['seq'] <= s['last'][0]]
        if not matching_shards:
            missing_boundaries += 1
            continue
        frame = matching_shards[0]['db'].execute('SELECT mono_ns, utc_ns FROM frames WHERE sequence=?', (row['seq'],)).fetchone()
        if frame is None:
            missing_boundaries += 1
            continue
        if frame[0] != row['mono'] or (row.get('utc_ns') and frame[1] != int(row['utc_ns'])):
            raise ValueError(f'Annotation timestamp does not match SQLite for sequence {row["seq"]}; check recording and run')
        matched_boundaries += 1
    if not matched_boundaries:
        raise ValueError('No annotation boundaries matched the supplied SQLite shards')
    stats = Counter()
    label_counts = Counter()
    too_short_by_label = Counter()
    shorter_than_window_by_label = Counter()
    eligible_intervals_by_label = Counter()
    annotation_starts = [row['seq'] for row in rows]
    sources = [{'path': str(s['path']), 'first_sequence': s['first'][0], 'last_sequence': s['last'][0],
                'wal_present': Path(str(s['path']) + '-wal').exists()} for s in shards]
    step_ns = int(a.stride_seconds * 1e9)
    gap_ns = int(a.max_gap_seconds * 1e9)
    stop = False
    used_centers = set()
    with (out / 'manifest.jsonl').open('w', encoding='utf-8', newline='\n') as manifest:
        for interval_index, (start, end) in enumerate(zip(rows, rows[1:])):
            stats['intervals_total'] += 1
            activity = start['activity_label'].strip()
            if start['presence_count'].strip() != '1':
                stats['intervals_not_single_person'] += 1
                continue
            if activity not in mapping:
                stats['intervals_unmapped_label'] += 1
                continue
            # All 32 consecutive source frames must be inside this interval.
            if end['seq'] - start['seq'] < 32:
                stats['intervals_shorter_than_window'] += 1
                shorter_than_window_by_label[activity] += 1
                if a.window_labeling == 'pure':
                    stats['intervals_too_short'] += 1
                    too_short_by_label[activity] += 1
                    continue
            stats['intervals_eligible'] += 1
            eligible_intervals_by_label[activity] += 1
            middle = (start['mono'] + end['mono']) // 2
            targets = [middle]
            t = middle - step_ns
            while t > start['mono']:
                targets.append(t)
                t -= step_ns
            t = middle + step_ns
            while t < end['mono']:
                targets.append(t)
                t += step_ns
            for target in sorted(targets):
                stats['candidate_centers'] += 1
                # Choose the latest frame at or before the target time.
                candidates = []
                for s in shards:
                    if s['first'][1] <= target <= s['last'][1]:
                        row = s['db'].execute('SELECT sequence, mono_ns FROM frames WHERE mono_ns <= ? ORDER BY mono_ns DESC LIMIT 1', (target,)).fetchone()
                        if row:
                            candidates.append((abs(row[1] - target), row[0]))
                if not candidates:
                    stats['candidate_outside_sqlite'] += 1
                    continue
                distance, center = min(candidates)
                if distance > gap_ns:
                    stats['candidate_too_far_from_target'] += 1
                    continue
                if not start['seq'] <= center < end['seq']:
                    stats['candidate_center_outside_interval'] += 1
                    continue
                if center in used_centers:
                    stats['candidate_duplicate_center'] += 1
                    continue
                used_centers.add(center)
                # For an even number of frames, anchor is the 17th frame.
                lo, hi = center - 16, center + 15
                if a.window_labeling == 'pure' and (lo < start['seq'] or hi >= end['seq']):
                    stats['candidate_crosses_label_boundary'] += 1
                    continue
                first_interval = bisect.bisect_right(annotation_starts, lo) - 1
                last_interval = bisect.bisect_right(annotation_starts, hi) - 1
                if first_interval < 0 or last_interval >= len(rows) - 1:
                    stats['candidate_outside_annotation'] += 1
                    continue
                context = []
                for position in range(first_interval, last_interval + 1):
                    context_row = rows[position]
                    context_activity = context_row['activity_label'].strip()
                    if context_row['presence_count'].strip() != '1' or context_activity not in mapping:
                        break
                    n_frames = min(hi + 1, rows[position + 1]['seq']) - max(lo, context_row['seq'])
                    context.append({'annotation_label':context_activity, 'coarse_label':mapping[context_activity][1],
                                    'frame_count':n_frames})
                else:
                    if sum(part['frame_count'] for part in context) != 32:
                        raise ValueError('Internal error: annotation context does not cover the 32-frame window')
                if len(context) != last_interval - first_interval + 1:
                    stats['candidate_non_single_or_unsupported_context'] += 1
                    continue
                records = []
                for s in shards:
                    if s['last'][0] >= lo and s['first'][0] <= hi:
                        records.extend(s['db'].execute('SELECT sequence, utc_ns, mono_ns, width, height, status, edge_data, edge_encoding FROM frames WHERE sequence BETWEEN ? AND ? ORDER BY sequence', (lo, hi)).fetchall())
                records.sort(key=lambda r:r[0])
                if [r[0] for r in records] != list(range(lo, hi + 1)):
                    stats['candidate_missing_frames'] += 1
                    continue
                if any(r[5] != 'ok' or r[6] is None for r in records):
                    stats['candidate_bad_status_or_missing_edge'] += 1
                    continue
                if any(not 0 < b[2] - x[2] <= gap_ns for x,b in zip(records, records[1:])):
                    stats['candidate_timestamp_gap'] += 1
                    continue
                try:
                    frames=[]
                    shape=(records[0][4],records[0][3])
                    packed_length=(shape[0]*shape[1]+7)//8
                    for r in records:
                        if (r[4],r[3]) != shape or r[7] != 'packbits-little-zlib-v1':
                            raise ValueError('unexpected frame dimensions or encoding')
                        decoder=zlib.decompressobj()
                        raw=decoder.decompress(r[6]) + decoder.flush()
                        if not decoder.eof or decoder.unused_data:
                            raise ValueError('invalid zlib stream')
                        if len(raw) != packed_length:
                            raise ValueError('invalid packed bitmap length')
                        frames.append(np.frombuffer(raw,dtype=np.uint8))
                    packed=np.stack(frames)
                except (ValueError, zlib.error) as exc:
                    stats['candidate_decode_errors'] += 1
                    print(f'Skipped sequence {center}: {exc}', file=sys.stderr)
                    continue
                name=f'clip_{stats["clips_written"]:07d}.npz'
                rel=f'clips/{name}'
                np.savez_compressed(out / rel, packed_frames=packed, height=np.int64(shape[0]), width=np.int64(shape[1]))
                label, coarse_label=mapping[activity]
                item={'path':rel, 'adapter_label_id':CLASSES.index(label), 'adapter_label':label,
                      'coarse_label':coarse_label, 'source_annotation_label':activity, 'annotation_interval_index':interval_index,
                      'annotation_start_sequence':start['seq'], 'annotation_end_sequence_exclusive':end['seq'],
                      'window_labeling':a.window_labeling, 'annotation_context':context,
                      'center_interval_fraction':next(part['frame_count'] / 32 for pos, part in enumerate(context, start=first_interval) if pos == interval_index),
                      'ground_truth_label_fraction':sum(part['frame_count'] for part in context if part['coarse_label'] == coarse_label) / 32,
                      'start_sequence':lo, 'end_sequence':hi, 'center_sequence':center,
                      'start_utc_ns':records[0][1], 'end_utc_ns':records[-1][1],
                      'start_monotonic_ns':records[0][2], 'end_monotonic_ns':records[-1][2]}
                manifest.write(json.dumps(item, separators=(',',':'))+'\n')
                stats['clips_written'] += 1
                label_counts[coarse_label] += 1
                if a.max_clips and stats['clips_written'] >= a.max_clips:
                    stop=True
                    break
            if stop:
                break
    summary={'settings': {'stride_seconds':a.stride_seconds, 'max_gap_seconds':a.max_gap_seconds,
                          'window_labeling':a.window_labeling,
                          'source_frames':32, 'temporal_sampling':'consecutive_source_frames',
                          'output_frames':32, 'max_clips':a.max_clips},
             'annotations':str(Path(a.annotations)), 'shards':sources, 'label_mapping':mapping, 'mapping_note':'Pairs are internal adapter class and coarse ground truth; bed-specific adapter classes do not claim bed location.',
             'annotation_boundaries_matched':matched_boundaries, 'annotation_boundaries_not_in_supplied_shards':missing_boundaries,
             'counts':dict(stats), 'clips_by_label':dict(label_counts),
             'intervals_too_short_by_annotation_label':dict(too_short_by_label),
             'intervals_shorter_than_window_by_annotation_label':dict(shorter_than_window_by_label),
             'eligible_intervals_by_annotation_label':dict(eligible_intervals_by_label),
             'stopped_at_max_clips':stop}
    (out / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    for s in shards:
        s['db'].close()
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
