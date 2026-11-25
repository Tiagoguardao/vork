//! Finalized displayed ordinals. One append page and one read page stay resident.
//! A record carries its effective anchor so a reverse lookup needs at most two reads.
use super::{MissingCropOffsets, NativeMatrix, ProbedMatrix};
use anyhow::{Context, Result};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
};

const RECORD: usize = 32;
pub(super) const PAGE: usize = 4096;
const BYTES: usize = PAGE * RECORD;

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Record {
    pub pts: i64,
    pub raw_pts: Option<i64>,
    pub position: i64,
    pub matrix: ProbedMatrix,
    pub anchor: usize,
    pub rejected: bool,
}

#[derive(Default)]
pub(super) struct Records {
    len: usize,
    disk: Option<File>,
    tail: Vec<u8>,
    cache: Vec<u8>,
    cached: Option<usize>,
}

// The fourth value of the existing two matrix bits preserves the fallback reason.
fn matrix_bits(matrix: ProbedMatrix) -> u64 {
    match matrix {
        Ok(None) => 0,
        Ok(Some(matrix)) => matrix as u64,
        Err(MissingCropOffsets) => 3,
    }
}

impl Records {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn push(
        &mut self,
        pts: i64,
        raw_pts: Option<i64>,
        position: i64,
        matrix: ProbedMatrix,
        anchor: usize,
    ) -> Result<()> {
        if self.tail.len() == BYTES {
            if self.disk.is_none() {
                self.disk = Some(tempfile::tempfile().context("create movie index spill file")?);
            }
            let file = self.disk.as_mut().unwrap();
            file.seek(SeekFrom::End(0))?;
            file.write_all(&self.tail)
                .context("write movie index page")?;
            self.tail.clear();
        }
        // Ordinal zero is the unseeked start. The index file's byte offset
        // reaches the u64 file-size limit well before this flag word fills up.
        let flags = u64::try_from(anchor)?
            .checked_mul(8)
            .context("movie index too long")?
            | matrix_bits(matrix);
        self.tail.extend_from_slice(&pts.to_le_bytes());
        self.tail.extend_from_slice(&flags.to_le_bytes());
        self.tail
            .extend_from_slice(&raw_pts.unwrap_or(i64::MIN).to_le_bytes());
        self.tail.extend_from_slice(&position.to_le_bytes());
        self.len += 1;
        Ok(())
    }

    /// Mark an unsafe landing on its own record, preserving the decoded key index. Readers whose records
    /// point here consult this word too, including across spill pages.
    pub fn disable_anchor(&mut self, index: usize) -> Result<()> {
        let record = self.get(index)?;
        if record.anchor != index || record.rejected {
            return Ok(());
        }
        let flags =
            ((record.anchor as u64) << 3 | 4 | matrix_bits(record.matrix)).to_le_bytes();
        let at = index % PAGE * RECORD + 8;
        if index / PAGE == (self.len - 1) / PAGE {
            self.tail[at..at + 8].copy_from_slice(&flags);
        } else {
            let file = self.disk.as_mut().context("missing movie index page")?;
            file.seek(SeekFrom::Start(u64::try_from(index)? * RECORD as u64 + 8))?;
            file.write_all(&flags).context("update movie seek hint")?;
            self.cache[at..at + 8].copy_from_slice(&flags);
        }
        Ok(())
    }

    pub fn get(&mut self, index: usize) -> Result<Record> {
        anyhow::ensure!(index < self.len, "movie index ordinal {index} missing");
        let page = index / PAGE;
        let data = if page == (self.len - 1) / PAGE {
            &self.tail
        } else {
            if self.cached != Some(page) {
                let file = self.disk.as_mut().context("missing movie index page")?;
                let pos = u64::try_from(page)?
                    .checked_mul(BYTES as u64)
                    .context("movie index too long")?;
                file.seek(SeekFrom::Start(pos))?;
                self.cached = None;
                self.cache.resize(BYTES, 0);
                file.read_exact(&mut self.cache)?;
                self.cached = Some(page);
            }
            &self.cache
        };
        let at = index % PAGE * RECORD;
        let pts = i64::from_le_bytes(data[at..at + 8].try_into()?);
        let flags = u64::from_le_bytes(data[at + 8..at + 16].try_into()?);
        let raw = i64::from_le_bytes(data[at + 16..at + 24].try_into()?);
        Ok(Record {
            pts,
            position: i64::from_le_bytes(data[at + 24..at + 32].try_into()?),
            raw_pts: (raw != i64::MIN).then_some(raw),
            matrix: match flags & 3 {
                1 => Ok(Some(NativeMatrix::Bt709)),
                2 => Ok(Some(NativeMatrix::Bt601)),
                3 => Err(MissingCropOffsets),
                _ => Ok(None),
            },
            anchor: (flags >> 3).try_into()?,
            rejected: flags & 4 != 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_pages_stay_bounded_as_the_file_grows() -> Result<()> {
        let mut records = Records::default();
        let matrices = [Ok(None), Ok(Some(NativeMatrix::Bt601)), Ok(Some(NativeMatrix::Bt709)), Err(MissingCropOffsets)];
        for i in 0..17 * PAGE {
            records.push(
                (i / 2) as i64,
                Some(i as i64),
                i as i64 * 42,
                matrices[i % matrices.len()],
                i / 20 * 20,
            )?;
        }
        for i in (0..records.len()).rev().step_by(23) {
            assert_eq!(
                records.get(i)?,
                Record {
                    pts: (i / 2) as i64,
                    raw_pts: Some(i as i64),
                    position: i as i64 * 42,
                    matrix: matrices[i % matrices.len()],
                    anchor: i / 20 * 20,
                    rejected: false
                }
            );
        }
        records.disable_anchor(20)?;
        records.disable_anchor(17 * PAGE - (17 * PAGE % 20) - 20)?;
        let _ = records.get(PAGE * 2)?;
        assert!(records.get(20)?.rejected);
        assert_eq!(records.get(20)?.anchor, 20);
        assert_eq!(
            records.tail.capacity() + records.cache.capacity(),
            2 * BYTES
        );
        assert_eq!(records.disk.unwrap().metadata()?.len(), (16 * BYTES) as u64);
        Ok(())
    }
}
