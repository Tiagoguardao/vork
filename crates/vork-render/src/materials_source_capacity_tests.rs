    use super::*;
    #[test]
    fn every_input_can_bind_a_distinct_full_width_source() -> Result<()> {
        let mut builder=RecordBuilder::openpbr();
        for input in 0..=OCCLUSION_INPUT {
            builder.bind_index(input,Tap::plain((1u64<<40)+input as u64),4)?;
            assert_eq!(builder.record.inputs[input]&constants::MATERIAL_TAP_INDEX_MASK,input as u32+1);
            assert_eq!(builder.record.inputs[input]>>28,4);
            assert_eq!(builder.taps[input][8],256);
        }
        assert!(builder.taps.len()>15);
        Ok(())
    }
