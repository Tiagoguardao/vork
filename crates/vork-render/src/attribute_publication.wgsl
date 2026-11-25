@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read> rows: array<array<u32, 3>>;
@group(0) @binding(3) var<storage, read_write> records: array<u32>;
struct Column { width: u32, count_offset: u32, stride: u32, length: u32 }
@group(0) @binding(4) var<uniform> column: Column;

@compute @workgroup_size(64)
fn publish(@builtin(global_invocation_id) id: vec3<u32>) {
    var live = 0xffffffffu;
    if column.count_offset != 0xffffffffu { live = counts[column.count_offset]; }
    for (var i = id.x; i < column.length; i += column.stride) {
        let row = rows[i];
        // Installation orders destinations by source index, including repeated
        // references from partitioned draws. No lane walks the inactive tail.
        if row[0] >= live { break; }
        for (var word = 0u; word < column.width; word++) {
            records[row[2] + word] = source[row[1] + word];
        }
    }
}
