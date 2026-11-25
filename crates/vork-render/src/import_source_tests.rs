    use super::*;

    fn container(chunks: &[(&[u8;4],&[u8])]) -> Vec<u8> {
        let mut bytes=b"glTF\x02\0\0\0\0\0\0\0".to_vec();
        for &(kind,data) in chunks {
            bytes.extend((data.len() as u32).to_le_bytes());bytes.extend(kind);bytes.extend(data);
        }
        let length=bytes.len() as u32;bytes[8..12].copy_from_slice(&length.to_le_bytes());bytes
    }

    fn admission(bytes: &[u8],base: &Path) -> Result<(serde_json::Value,Option<u64>,Vec<std::path::PathBuf>)> {
        let (gltf,length)=source_document(&mut std::io::Cursor::new(bytes),bytes.len() as u64)?;
        assert!(gltf.blob.is_none(),"admission must not manufacture a BIN payload");
        let files=validate_sources(&gltf.document,length,base)?;
        Ok((serde_json::to_value(gltf.document.as_json())?,length,files))
    }

    fn assert_same_admission(bytes: &[u8],base: &Path) {
        let old=(|| -> Result<_> {
            let gltf=from_slice(bytes)?;let length=gltf.blob.as_ref().map(|b|b.len() as u64);
            let files=validate_sources(&gltf.document,length,base)?;
            Ok((serde_json::to_value(gltf.document.as_json())?,length,files))
        })().map_err(|e|format!("{e:#}"));
        assert_eq!(admission(bytes,base).map_err(|e|format!("{e:#}")),old);
    }

    #[test]
    fn metadata_source_admission_matches_loader_refusals() {
        let base=Path::new(".");
        let json=br#"{"asset":{"version":"2.0"},"buffers":[{"byteLength":4}]}"#;
        let basic=br#"{"asset":{"version":"2.0"}}"#;
        let valid=container(&[(b"JSON",json),(b"BIN\0",&[1,2,3,4])]);
        assert!(admission(&valid,base).is_ok());
        for bytes in [
            container(&[(b"JSON",basic)]),
            container(&[(b"JSON",basic),(b"BIN\0",&[])]),
            container(&[(b"JSON",json)]), // missing BIN
            container(&[(b"JSON",json),(b"BIN\0",&[])]), // short BIN
            container(&[(b"JSON",json),(b"BIN\0",&[1,2,3])]), // importer padding
            container(&[(b"JSON",json),(b"BIN\0",&[1,2,3,4]),(b"JUNK",&[9])]),
            container(&[(b"BIN\0",basic)]),
            container(&[(b"JUNK",basic)]),
            container(&[(b"JSON",basic),(b"JSON",&[])]),
            container(&[(b"JSON",basic),(b"JUNK",&[])]),
            container(&[]),
            container(&[(b"JSON",b"not JSON")]),
            container(&[(b"JSON",&[b"\xEF\xBB\xBF".as_slice(),basic].concat())]),
            [b"\xEF\xBB\xBF".as_slice(),&valid].concat(),
            [b"\xEF\xBB\xBF".as_slice(),basic].concat(),
            b"Kaydara FBX Binary  \0".to_vec(),b"; FBX 7.4".to_vec(),
        ] {assert_same_admission(&bytes,base);}
        // Exact error precedence, header/chunk truncation and ignored trailing
        // chunk behavior are compared against the unchanged installer parser.
        for end in 0..valid.len() {assert_same_admission(&valid[..end],base);}
        for index in 0..valid.len() {
            for value in [0,255] {let mut bytes=valid.clone();bytes[index]=value;assert_same_admission(&bytes,base);}
        }
        for tail in [vec![1],vec![0;7],vec![255;8]] {
            let mut bytes=valid.clone();bytes.extend(tail);let length=bytes.len() as u32;
            bytes[8..12].copy_from_slice(&length.to_le_bytes());assert_same_admission(&bytes,base);
        }
        // The same normalization and document/range checks still run on JSON.
        for (index,json) in [
            serde_json::json!({"asset":{"version":"2.0"},"materials":[{"normalTexture":{"index":0,"texCoord":-1}}],"textures":[{"source":0}],"images":[{"uri":"missing-normal.png"}]}),
            serde_json::json!({"asset":{"version":"2.0"},"extensionsRequired":["KHR_materials_clearcoat"],"materials":[{"extensions":{"KHR_materials_clearcoat":{"clearcoatFactor":1}}}]}),
            serde_json::json!({"asset":{"version":"2.0"},"nodes":[{"children":[0]}]}),
            serde_json::json!({"asset":{"version":"2.0"},"buffers":[{"byteLength":4}],"bufferViews":[{"buffer":0,"byteLength":8}]}),
            serde_json::json!({"asset":{"version":"2.0"},"buffers":[{"byteLength":4}],"bufferViews":[{"buffer":0,"byteLength":4}],"accessors":[{"bufferView":0,"componentType":5126,"count":2,"type":"SCALAR"}]}),
            serde_json::json!({"asset":{"version":"2.0"},"images":[{}]}),
        ].into_iter().enumerate() {
            let bytes=container(&[(b"JSON",&serde_json::to_vec(&json).unwrap()),(b"BIN\0",&[0;4])]);
            assert_same_admission(&bytes,base);
            assert_eq!(admission(&bytes,base).is_ok(),index<2,"normalization accepts; invalid document ranges refuse");
        }
    }

    #[test]
    fn metadata_source_admission_buffers_extension_chunks() -> Result<()> {
        use std::io::{Read,Seek,SeekFrom};
        struct Counted {bytes:std::io::Cursor<Vec<u8>>,reads:usize,seeks:usize,read_bytes:usize}
        impl Read for Counted {
            fn read(&mut self,out:&mut [u8])->std::io::Result<usize> {
                self.reads+=1;
                let count=self.bytes.read(out)?;self.read_bytes+=count;Ok(count)
            }
        }
        impl Seek for Counted {
            fn seek(&mut self,pos:SeekFrom)->std::io::Result<u64> {
                self.seeks+=1;self.bytes.seek(pos)
            }
        }
        // Zero and nonzero small payloads exercise buffered header/skip paths,
        // including headers straddling buffer boundaries. Large payloads still
        // seek rather than reading the whole extension into memory.
        for size in [0,4,13,16384] {
            let json=br#"{"asset":{"version":"2.0"}}"#;
            let payload=vec![0;size];
            let count=4096;
            let mut chunks=vec![(b"JSON",json.as_slice()),(b"BIN\0",&[][..])];
            chunks.extend(std::iter::repeat_n((b"TEST",payload.as_slice()),count));
            let bytes=container(&chunks);let length=bytes.len() as u64;
            let mut reader=Counted {bytes:std::io::Cursor::new(bytes),reads:0,seeks:0,read_bytes:0};
            let (_,bin_length)=source_document(&mut reader,length)?;
            assert_eq!(bin_length,Some(0));
            if size<8192 {
                assert!(reader.reads<count/100,"{} reads for {count} small chunks",reader.reads);
                assert!(reader.seeks<20,"{} seeks for {count} small chunks",reader.seeks);
            } else {
                assert!(reader.read_bytes<(length as usize)/2,"large extension payloads must be skipped");
            }
            // Even an ignored chunk after many buffer refills must be checked.
            let bytes=reader.bytes.get_mut();let end=bytes.len();
            bytes[end-size-8..end-size-4].copy_from_slice(&u32::MAX.to_le_bytes());
            assert_same_admission(bytes,Path::new("."));
        }
        Ok(())
    }

    #[test]
    fn metadata_source_admission_skips_binary_payload() -> Result<()> {
        use std::io::{Read,Seek,SeekFrom};
        struct Guard {bytes:std::io::Cursor<Vec<u8>>,forbidden:std::ops::Range<u64>}
        impl Read for Guard {
            fn read(&mut self,out:&mut [u8])->std::io::Result<usize> {
                let at=self.bytes.position();let end=(at+out.len() as u64).min(self.bytes.get_ref().len() as u64);
                assert!(end<=self.forbidden.start || at>=self.forbidden.end,"admission read BIN bytes");
                self.bytes.read(out)
            }
        }
        impl Seek for Guard {fn seek(&mut self,pos:SeekFrom)->std::io::Result<u64>{self.bytes.seek(pos)}}
        let json=br#"{"asset":{"version":"2.0"},"buffers":[{"byteLength":32768}]}"#;
        let bytes=container(&[(b"JSON",json),(b"BIN\0",&vec![0;32768]),(b"JUNK",&[1,2,3])]);
        let length=bytes.len() as u64;let bin_start=28+json.len() as u64;
        let mut reader=Guard {bytes:std::io::Cursor::new(bytes),forbidden:bin_start..bin_start+32768};
        let (gltf,bin_length)=source_document(&mut reader,length)?;
        assert_eq!(bin_length,Some(32768));assert!(gltf.blob.is_none());
        validate_sources(&gltf.document,bin_length,Path::new("."))?;
        Ok(())
    }

    #[test]
    fn metadata_source_admission_validates_external_files() -> Result<()> {
        let root=std::env::temp_dir().join(format!("vork-gltf-metadata-source-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;std::fs::write(root.join("short.bin"),[1,2,3])?;
        std::fs::create_dir_all(root.join("directory.png"))?;
        for (uri,length) in [("short.bin",4),("short.bin",8),("missing.bin",4),("https://example.invalid/data",4)] {
            let json=serde_json::to_vec(&serde_json::json!({"asset":{"version":"2.0"},"buffers":[{"uri":uri,"byteLength":length}]}))?;
            assert_same_admission(&container(&[(b"JSON",&json)]),&root);
        }
        for image in ["missing.png","directory.png","https://example.invalid/image"] {
            let json=serde_json::to_vec(&serde_json::json!({"asset":{"version":"2.0"},"images":[{"uri":image}]}))?;
            assert_same_admission(&container(&[(b"JSON",&json)]),&root);
        }
        let json=br#"{"asset":{"version":"2.0"},"buffers":[{"byteLength":4}]}"#;
        let path=root.join("valid.glb");std::fs::write(&path,container(&[(b"JSON",json),(b"BIN\0",&[0;4])]))?;
        assert!(validate_file(&path,&root)?.is_empty());
        std::fs::remove_dir_all(root)?;Ok(())
    }

    #[test]
    fn source_admission_matches_import_uri_and_padding_rules() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-gltf-source-rules-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        std::fs::write(root.join("three bytes.bin"), [1, 2, 3])?;
        let document = from_slice(&serde_json::to_vec(&serde_json::json!({
            "asset":{"version":"2.0"},
            "buffers":[
                {"uri":"three%20bytes.bin","byteLength":4},
                {"uri":format!("file:{}",root.join("three bytes.bin").display()),"byteLength":4},
                {"uri":"data:application/octet-stream;base64,AQID","byteLength":4}
            ],
            "images":[{"uri":"missing-sampled-image.png"}]
        }))?)?;
        assert_eq!(validate_sources(&document.document, document.blob.as_ref().map(|b|b.len() as u64), &root)?, vec![
            root.join("three bytes.bin"), root.join("three bytes.bin"), root.join("missing-sampled-image.png")]);
        // Installation accepts these exact external paths and pads 3 bytes to 4.
        let payloads = buffers(&document.document, Some(&root), None)?;
        assert!(payloads.iter().all(|data| data.len() == 4));
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
