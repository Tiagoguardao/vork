//! Retained GPU record columns shared by text and the engine packet writer.

/// One storage buffer of a packet as the CPU holds it: the records its collect uploaded, kept so a patch compares
/// a changed item's record with the one the buffer holds and writes it where it stands. Record `i` is the
/// buffer's bytes from `i * size_of::<T>()`.
pub struct Column<T> {
    records: Vec<T>,
    // The unchanged tail of regular matrix records can be reproduced exactly
    // from its first record and the strides of its nonconstant words.
    tails: Vec<Option<Box<RecordTail<T>>>>,
    partial: Vec<PartialMatrix>,
    /// GPU records, including the suffix whose CPU mirror has been evicted.
    len: usize,
    /// Records a patch asked for and has not committed, by index (`Column::put`).
    pending: Vec<(u32, std::ops::Range<usize>, T)>,
}

const PAGE: usize = 256;
struct RecordTail<T> {
    first: T,
    steps: Vec<(usize, u32)>,
}

struct PartialMatrix {
    index: u32,
    matrix: [f32; 16],
    padding: Option<[f32; 2]>,
}

impl<T: bytemuck::Pod> RecordTail<T> {
    fn of(rows: &[T]) -> Option<Self> {
        let first = rows[0];
        let word = |row: &T, at: usize| u32::from_ne_bytes(bytemuck::bytes_of(row)[at..at+4].try_into().unwrap());
        let second = rows.get(1).unwrap_or(&first);
        let mut steps = Vec::new();
        for at in (64..std::mem::size_of::<T>()).step_by(4) {
            let initial = word(&first, at);
            let step = word(second, at).wrapping_sub(initial);
            if !rows.iter().enumerate().all(|(i, row)| word(row, at) == initial.wrapping_add(step.wrapping_mul(i as u32))) {
                return None;
            }
            if step != 0 { steps.push((at, step)); }
        }
        Some(Self { first, steps })
    }
    fn record(&self, row: u32) -> T {
        let mut record = self.first;
        for &(at, step) in &self.steps {
            let bytes = &mut bytemuck::bytes_of_mut(&mut record)[at..at+4];
            let first = u32::from_ne_bytes((&*bytes).try_into().unwrap());
            bytes.copy_from_slice(&first.wrapping_add(row.wrapping_mul(step)).to_ne_bytes());
        }
        record
    }
}

impl<T> Default for Column<T> {
    fn default() -> Self {
        Self { records: Vec::new(), tails: Vec::new(), partial: Vec::new(), len: 0, pending: Vec::new() }
    }
}

impl<T: bytemuck::Pod> Column<T> {
    pub fn of(records: Vec<T>) -> Self {
        Self { len: records.len(), records, ..Self::default() }
    }

    /// Before dropping the mirrors, keep inexpensive exact recipes for bridging
    /// neighboring matrix writes. Irregular pages use individual field uploads.
    pub fn retain_tails(&mut self, budget: &mut usize) {
        if self.records.len() != self.len || self.len == 0 || !self.tails.is_empty() { return; }
        let pages = self.len.div_ceil(PAGE);
        let mut held = pages * std::mem::size_of::<Option<Box<RecordTail<T>>>>();
        if held >= *budget { return; }
        let mut tails = Vec::with_capacity(pages);
        for rows in self.records.chunks(PAGE) {
            let tail = RecordTail::of(rows).and_then(|tail| {
                let bytes = std::mem::size_of_val(&tail) + tail.steps.capacity() * std::mem::size_of::<(usize, u32)>();
                if held + bytes > *budget { return None; }
                held += bytes;
                Some(Box::new(tail))
            });
            tails.push(tail);
        }
        if tails.iter().any(Option::is_some) {
            self.tails = tails;
            *budget -= held;
        }
    }

    /// The placement proof covers the matrix and, for vector text, the two
    /// padding words. Every other word stays at its value on the GPU.
    pub fn put_matrix(&mut self, index: u32, matrix: &[f32; 16], padding: Option<[f32; 2]>) -> Result<(), &'static str> {
        if index as usize >= self.len { return Err("matrix destination"); }
        let i = index as usize;
        let old = self.records.get(i).copied().or_else(|| {
            Some(self.tails.get(i / PAGE)?.as_ref()?.record((i % PAGE) as u32))
        });
        if let Some(mut record) = old {
            let bytes = bytemuck::bytes_of_mut(&mut record);
            bytes[..64].copy_from_slice(bytemuck::bytes_of(matrix));
            if let Some(words) = padding {
                bytes[140..144].copy_from_slice(&words[0].to_ne_bytes());
                bytes[156..160].copy_from_slice(&words[1].to_ne_bytes());
            }
            self.pending.push((index, 0..if padding.is_some() { 160 } else { 64 }, record));
        } else {
            self.partial.push(PartialMatrix { index, matrix: *matrix, padding });
        }
        Ok(())
    }

    /// Ask for `record` at `index`. `Ok(true)`: it differs from the one held and is pending until
    /// [`Column::commit`]; `Ok(false)`: the buffer already holds it. `Err`: the column has no such record.
    pub fn put(&mut self, index: u32, record: &T) -> std::result::Result<bool, &'static str> {
        self.put_range(index, record, 0..std::mem::size_of::<T>())
    }

    /// A complete slot is available, including alignment padding. Preserve it for
    /// coalescing across an evicted mirror, and narrow a resident write to changed words.
    pub(crate) fn put_words(&mut self, index: u32, record: &T, mut range: std::ops::Range<usize>) -> Result<bool, &'static str> {
        if let Some(held) = self.records.get(index as usize) {
            let old = bytemuck::bytes_of(held).get(range.clone()).ok_or("record")?;
            let new = bytemuck::bytes_of(record).get(range.clone()).ok_or("record")?;
            let mut words = old.chunks_exact(4).zip(new.chunks_exact(4));
            let Some(first) = words.position(|(a, b)| a != b) else { return Ok(false); };
            let trailing = words.rev().take_while(|(a, b)| a == b).count();
            range.start += first * 4;
            range.end -= trailing * 4;
        }
        self.put_range(index, record, range)
    }

    /// The owner proves that bytes outside this aligned range still stand on the GPU.
    /// Keep the complete value so consecutive writes can share an upload.
    pub fn put_range(&mut self, index: u32, record: &T, range: std::ops::Range<usize>) -> std::result::Result<bool, &'static str> {
        if index as usize >= self.len || range.is_empty() || range.end > std::mem::size_of::<T>() {
            return Err("record");
        }
        if self.records.get(index as usize).is_some_and(|held| bytemuck::bytes_of(held) == bytemuck::bytes_of(record)) {
            return Ok(false);
        }
        if range.end > 64 {
            let i = index as usize;
            if let Some(tail) = self.tails.get_mut(i / PAGE) {
                if tail.as_ref().is_some_and(|t| bytemuck::bytes_of(&t.record((i % PAGE) as u32))[64..] != bytemuck::bytes_of(record)[64..]) {
                    *tail = None;
                }
            }
        }
        self.pending.push((index, range, *record));
        Ok(true)
    }
}

/// A [`Column`] without its record type, for what is done to every column of a packet alike.
pub trait PendingRecords {
    /// The bytes the column's buffer holds.
    fn bytes(&self) -> u64;
    /// Keep as many complete CPU records as fit. GPU offsets and bounds stay valid.
    fn fit_budget(&mut self, bytes: usize) -> usize;
    fn is_pending(&self) -> bool;
    /// Forget what was asked for.
    fn discard(&mut self);
    /// Take the pending records into the column and hand `upload` the column from its first changed record to
    /// its last, as (byte offset in the buffer, bytes): one write for a column, however its changed records lie.
    fn commit(&mut self, upload: &mut dyn FnMut(u64, &[u8]));
}

impl<T: bytemuck::Pod> PendingRecords for Column<T> {
    fn bytes(&self) -> u64 {
        self.len as u64 * std::mem::size_of::<T>() as u64
    }
    fn fit_budget(&mut self, bytes: usize) -> usize {
        let size = std::mem::size_of::<T>().max(1);
        if self.records.capacity() > bytes / size {
            self.records.truncate(bytes / size);
            self.records.shrink_to_fit();
        }
        self.records.capacity() * size
    }
    fn is_pending(&self) -> bool {
        !self.pending.is_empty() || !self.partial.is_empty()
    }
    fn discard(&mut self) {
        self.pending = Vec::new();
        self.partial = Vec::new();
    }
    fn commit(&mut self, upload: &mut dyn FnMut(u64, &[u8])) {
        for write in std::mem::take(&mut self.partial) {
            let offset = u64::from(write.index) * std::mem::size_of::<T>() as u64;
            upload(offset, bytemuck::bytes_of(&write.matrix));
            if let Some(words) = write.padding {
                upload(offset + 140, bytemuck::bytes_of(&words[0]));
                upload(offset + 156, bytemuck::bytes_of(&words[1]));
            }
        }
        if self.pending.is_empty() { return; }
        if !self.pending.is_sorted_by_key(|(index, ..)| *index) {
            self.pending.sort_by_key(|(index, ..)| *index);
        }
        let split = self.pending.partition_point(|(index, ..)| (*index as usize) < self.records.len());
        let (mirrored, evicted) = self.pending.split_at(split);
        let size = std::mem::size_of::<T>() as u64;
        for (index, _, value) in mirrored { self.records[*index as usize] = *value; }
        // Consecutive glyphs may change just one word in each aligned uniform.
        // Bridge nearby mirrored words in one upload, including their known intervening
        // values. Distant edits and holes in the mirror cannot use that range.
        for run in mirrored.chunk_by(|a, b| {
            let gap = b.0.saturating_sub(a.0).saturating_sub(1);
            u64::from(gap) * size <= crate::text_draw::UNIFORM_ALIGN
        }) {
            let first = run[0].0;
            let last = run.last().unwrap().0;
            let data = bytemuck::cast_slice(&self.records[first as usize..=last as usize]);
            let start = run.iter().take_while(|r| r.0 == first).map(|r| r.1.start).min().unwrap();
            let end = data.len() - size as usize + run.iter().rev().take_while(|r| r.0 == last).map(|r| r.1.end).max().unwrap();
            upload(u64::from(first) * size + start as u64, &data[start..end]);
        }
        let mut packed = Vec::new();
        for run in evicted.chunk_by(|a, b| b.0 == a.0.saturating_add(1)) {
            let offset = u64::from(run[0].0) * size;
            if let [(_, range, value)] = run {
                upload(offset + range.start as u64, &bytemuck::bytes_of(value)[range.clone()]);
            }
            else {
                packed.clear();
                packed.extend(run.iter().map(|(_, _, value)| *value));
                let start = run[0].1.start;
                let end = (run.len() - 1) * size as usize + run.last().unwrap().1.end;
                upload(offset + start as u64, &bytemuck::cast_slice(&packed)[start..end]);
            }
        }
        self.pending = Vec::new();
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matrix_writes_preserve_evicted_uniform_words_after_a_paint_edit() {
        type Slot = [u32; 64];
        let slots: Vec<Slot> = (0..512u32).map(|i| {
            let mut slot = [0x11111111; 64];
            // The first page has a strided depth. The second has an irregular paint word.
            slot[20] = 1000 - i;
            if i >= 256 { slot[18] = i % 7; }
            slot
        }).collect();
        let mut gpu = bytemuck::cast_slice::<Slot, u8>(&slots).to_vec();
        let mut expected = slots.clone();
        let mut column = Column::of(slots);
        let mut budget = 4096;
        column.retain_tails(&mut budget);
        assert!(column.tails[0].is_some() && column.tails[1].is_none());
        column.fit_budget(0);
        let commit = |c: &mut Column<Slot>, gpu: &mut [u8]| c.commit(&mut |offset, data| {
            gpu[offset as usize..offset as usize + data.len()].copy_from_slice(data);
        });
        for round in 1..=2 {
            let matrix = [round as f32; 16];
            let padding = [0.75, 0.001];
            for &i in &[1, 2, 255, 256, 290] {
                column.put_matrix(i, &matrix, Some(padding)).unwrap();
                let row = &mut expected[i as usize];
                row[..16].copy_from_slice(&matrix.map(f32::to_bits));
                row[35] = padding[0].to_bits();
                row[39] = padding[1].to_bits();
            }
            commit(&mut column, &mut gpu);
            assert_eq!(gpu, bytemuck::cast_slice::<Slot, u8>(&expected));
            expected[2][18] += 1;
            column.put(2, &expected[2]).unwrap();
            commit(&mut column, &mut gpu);
        }
        column.put_matrix(1, &[9.; 16], None).unwrap();
        column.discard();
        commit(&mut column, &mut gpu);
        assert_eq!(gpu, bytemuck::cast_slice::<Slot, u8>(&expected));
    }

    #[test]
    fn aligned_edits_share_an_upload_and_distant_edits_stay_separate() {
        let mut column = Column::of(vec![0u32; 2048]);
        for i in [12, 76, 140, 1800] { column.put(i, &1).unwrap(); }
        let mut ranges = Vec::new();
        column.commit(&mut |offset, values| ranges.push((offset, values.len())));
        assert_eq!(ranges, [(48, 516), (7200, 4)]);
    }

    #[test]
    fn evicted_records_keep_their_gpu_addresses_and_only_upload_on_commit() {
        let mut column = Column::of(vec![0u32; 8]);
        assert_eq!(column.fit_budget(9), 8);
        assert_eq!(column.bytes(), 32);
        assert_eq!(column.put(8, &1), Err("record"));
        assert_eq!(column.put(0, &0), Ok(false));
        assert_eq!(column.put(7, &1), Ok(true));
        column.discard();
        column.put(5, &123).unwrap();
        column.put(1, &456).unwrap();
        let mut gpu = [0u32; 8];
        column.commit(&mut |offset, data| {
            let at = offset as usize / 4;
            gpu[at] = bytemuck::pod_read_unaligned(data);
        });
        assert_eq!(gpu, [0, 456, 0, 0, 0, 123, 0, 0]);
    }

    #[test]
    fn a_column_uploads_once_from_its_first_changed_record_to_its_last() {
        let mut column = Column::of((0..8u32).map(|i| [i, 0]).collect::<Vec<[u32; 2]>>());
        // A record the buffer already holds asks for nothing.
        assert_eq!(column.put(3, &[3, 0]), Ok(false));
        assert!(!column.is_pending());
        // Asked out of order, with unchanged records between them.
        assert_eq!(column.put(5, &[5, 9]), Ok(true));
        assert_eq!(column.put(2, &[2, 9]), Ok(true));
        assert_eq!(column.put(1, &[1, 9]), Ok(true));
        assert_eq!(column.put(8, &[8, 9]), Err("record"));
        // Nothing is taken before the commit.
        assert_eq!(column.records[2], [2, 0]);
        let mut uploads: Vec<(u64, Vec<u8>)> = Vec::new();
        column.commit(&mut |offset, data| uploads.push((offset, data.to_vec())));
        let bytes = |records: &[[u32; 2]]| bytemuck::cast_slice::<[u32; 2], u8>(records).to_vec();
        assert_eq!(uploads, vec![(8, bytes(&[[1, 9], [2, 9], [3, 0], [4, 0], [5, 9]]))]);
        assert!(!column.is_pending());
        assert_eq!(column.records, vec![[0, 0], [1, 9], [2, 9], [3, 0], [4, 0], [5, 9], [6, 0], [7, 0]]);
        assert_eq!(column.bytes(), 64);
        // A discarded record is never taken, and a column with nothing pending uploads nothing.
        assert_eq!(column.put(0, &[7, 7]), Ok(true));
        column.discard();
        column.commit(&mut |_, _| panic!("nothing is pending"));
        assert_eq!(column.records[0], [0, 0]);
    }

}
