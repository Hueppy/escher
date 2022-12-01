use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use nalgebra::{Matrix4, Point3, Rotation3, Vector3};
use vulkano::{
    buffer::{cpu_pool::CpuBufferPoolSubbuffer, CpuBufferPool},
    device::Device,
    memory::pool::StandardMemoryPool,
};

#[derive(Debug)]
pub struct CameraConfiguration {
    pub position: Vector3<f32>,
    pub angle: Vector3<f32>,

    pub aspect: f32,
    pub fov_y: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Default for CameraConfiguration {
    fn default() -> Self {
        Self {
            position: Vector3::zeros(),
            angle: Vector3::zeros(),

            aspect: 1.0,
            fov_y: 45.0,
            z_near: 0.01,
            z_far: 100.0,
        }
    }
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, Zeroable, Pod)]
pub struct CameraData {
    view: Matrix4<f32>,
    projection: Matrix4<f32>,
}

impl CameraData {
    const BASE_MATRIX: Matrix4<f32> = Matrix4::new(
        1.0,  0.0,  0.0,  0.0,
        0.0,  1.0,  0.0,  0.0,
        0.0,  0.0, -0.5,  0.5,
        0.0,  0.0,  0.0, -1.0,
    );

    pub fn new(configuration: &CameraConfiguration) -> Self {
        let eye = Point3::from(configuration.position);
        let target = Rotation3::new(Vector3::new(0.0, 0.0, configuration.angle.z))
            * Rotation3::new(Vector3::new(0.0, configuration.angle.y, 0.0))
            * Rotation3::new(Vector3::new(configuration.angle.x, 0.0, 0.0))
            * Point3::new(0.0, 0.0, 1.0)
            + configuration.position;
        let up = Vector3::y();

        let view = Matrix4::look_at_lh(&eye, &target, &up);

        let projection = Self::BASE_MATRIX
            * Matrix4::new_perspective(
                configuration.aspect,
                configuration.fov_y,
                configuration.z_near,
                configuration.z_far,
            );

        Self { view, projection }
    }
}

pub struct Camera {
    configuration: CameraConfiguration,

    buffer_pool: CpuBufferPool<CameraData>,
    subbuffer: Option<Arc<CpuBufferPoolSubbuffer<CameraData, Arc<StandardMemoryPool>>>>,
}

impl Camera {
    pub fn new(device: Arc<Device>) -> Self {
        let configuration = Default::default();
        let buffer_pool = CpuBufferPool::uniform_buffer(device);
        let subbuffer = None;

        Self {
            configuration,
            buffer_pool,
            subbuffer,
        }
    }

    pub fn subbuffer(
        &mut self,
    ) -> Arc<CpuBufferPoolSubbuffer<CameraData, Arc<StandardMemoryPool>>> {
        if self.subbuffer.is_none() {
            let data = CameraData::new(&self.configuration);
            self.subbuffer = Some(self.buffer_pool.from_data(data).unwrap());
        }

        self.subbuffer.clone().unwrap()
    }

    pub fn update<F: FnOnce(&mut CameraConfiguration)>(&mut self, f: F) {
        self.invalidate();
        f(&mut self.configuration);
    }

    pub fn invalidate(&mut self) {
        self.subbuffer = None;
    }
}
