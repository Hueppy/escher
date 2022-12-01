use std::{
    collections::HashMap,
    io::{BufReader, Cursor},
    path::Path,
};

use bytemuck::{Pod, Zeroable};
use nalgebra::{Vector2, Vector3};

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, Zeroable, Pod)]
pub struct Vertex {
    pub coord: Vector3<f32>,
    pub normal: Vector3<f32>,
    pub tex_coord: Vector2<f32>,
}

vulkano::impl_vertex!(Vertex, coord, normal, tex_coord);

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn from_obj(obj: &str, mtls: &HashMap<&Path, &str>) -> Result<Vec<Self>, tobj::LoadError> {
        let cursor = Cursor::new(obj);
        let mut reader = BufReader::new(cursor);

        let (models, _) = tobj::load_obj_buf(
            &mut reader,
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
            |p| {
                mtls.get(p)
                    .map(Cursor::new)
                    .map(BufReader::new)
                    .map(|mut reader| tobj::load_mtl_buf(&mut reader))
                    .unwrap_or(Err(tobj::LoadError::OpenFileFailed))
            },
        )?;

        let meshes = models
            .into_iter()
            .map(|m| {
                let vertices = (0..m.mesh.positions.len() / 3)
                    .map(|p| Vertex {
                        coord: Vector3::from_row_slice(&m.mesh.positions[p * 3..(p + 1) * 3]),
                        tex_coord: Vector2::from_row_slice(&m.mesh.texcoords[p * 2..(p + 1) * 2]),
                        normal: Vector3::from_row_slice(&m.mesh.normals[p * 3..(p + 1) * 3]),
                    })
                    .collect();

                let indices = m.mesh.indices;

                Self { vertices, indices }
            })
            .collect();

        Ok(meshes)
    }
}
