//! Independent writes to changed rows of a dense scene column.

/// Apply edits in disjoint slices. Rows within a run must increase strictly.
/// A new run begins on an equal or descending row, preserving write order for
/// repeated edits. Returns false if any visit requests a full evaluation.
pub fn visit_changed<T: Send, E: Send>(
    column: &mut [T], edits: &mut [E], row: impl Fn(&E) -> usize + Sync,
    visit: impl Fn(&mut T, &mut E) -> bool + Sync,
) -> bool {
    fn run<T: Send, E: Send>(
        column: &mut [T], base: usize, edits: &mut [E], grain: usize,
        row: &(impl Fn(&E) -> usize + Sync), visit: &(impl Fn(&mut T, &mut E) -> bool + Sync),
    ) -> bool {
        if edits.len() <= grain {
            return edits.iter_mut().all(|edit| visit(&mut column[row(edit) - base], edit));
        }
        let (left, right) = edits.split_at_mut(edits.len() / 2);
        let next = row(&right[0]);
        let (before, after) = column.split_at_mut(next - base);
        let (a, b) = rayon::join(
            || run(before, base, left, grain, row, visit),
            || run(after, next, right, grain, row, visit),
        );
        a && b
    }
    // Equal-cost row edits need at most one chunk per worker in a sorted run.
    let grain = if edits.len() <= 2048 { 2048 } else {
        edits.len().div_ceil(rayon::current_num_threads()).max(2048)
    };
    edits.chunk_by_mut(|a, b| row(a) < row(b))
        .all(|run_edits| run(column, 0, run_edits, grain, &row, &visit))
}
