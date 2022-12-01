use std::{collections::HashMap, sync::Arc};

use nalgebra::{Matrix4, Vector3};
use vulkano::{
    buffer::{
        cpu_pool::CpuBufferPoolSubbuffer, BufferUsage, CpuAccessibleBuffer, CpuBufferPool,
        TypedBufferAccess,
    },
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferUsage,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SecondaryAutoCommandBuffer, SubpassContents,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{Device, Queue},
    memory::pool::StandardMemoryPool,
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            rasterization::{CullMode, FrontFace, RasterizationState},
            render_pass::PipelineRenderPassType,
            vertex_input::{BuffersDefinition, VertexMember, VertexMemberTy},
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint, StateMode,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    shader::ShaderModule, image::{view::ImageView, ImmutableImage},
};

use crate::{
    camera::{Camera, CameraData},
    mesh::{Mesh, Vertex},
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Model {
    pub model: Matrix4<f32>,
}

impl Default for Model {
    fn default() -> Self {
        Self {
            model: Matrix4::identity(),
        }
    }
}

unsafe impl VertexMember for Model {
    fn format() -> (VertexMemberTy, usize) {
        let (ty, sz) = f32::format();
        (ty, sz * 4 * 4)
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Instance {
    pub position: Vector3<f32>,
    pub angle: Vector3<f32>,
    pub scale: f32,

    model: Model,
}

vulkano::impl_vertex!(Instance, model);

impl Instance {
    pub fn update<F: FnOnce(&mut Instance)>(&mut self, f: F) {
        f(self);

        self.model = Model {
            model: Matrix4::new_translation(&self.position)
                * Matrix4::new_rotation(Vector3::new(self.angle.x, 0.0, 0.0))
                * Matrix4::new_rotation(Vector3::new(0.0, self.angle.y, 0.0))
                * Matrix4::new_rotation(Vector3::new(0.0, 0.0, self.angle.z))
                * Matrix4::new_scaling(self.scale),
        }
    }
}

impl Default for Instance {
    fn default() -> Self {
        let position = Vector3::new(0.0, 0.0, 0.0);
        let angle = Vector3::new(0.0, 0.0, 0.0);
        let scale = 1.0;

        let model = Default::default();

        Self {
            position,
            angle,
            scale,

            model,
        }
    }
}

pub struct Object {
    instances: HashMap<String, Instance>,

    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    index_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
    instance_buffer: CpuBufferPool<Instance>,
//    texture_image: ImageView<ImmutableImage>,

    command_buffer: Option<Arc<SecondaryAutoCommandBuffer>>,
}

impl Object {
    pub fn new(mesh: Mesh, device: Arc<Device>) -> Self {
        let instances = HashMap::new();

        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            mesh.vertices,
        )
        .unwrap();

        let index_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage {
                index_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            mesh.indices,
        )
        .unwrap();

        let instance_buffer = CpuBufferPool::vertex_buffer(device);

        let command_buffer = None;

        Self {
            instances,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            command_buffer,
        }
    }

    pub fn command_buffer(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        pipeline: Arc<GraphicsPipeline>,
        camera_buffer: Arc<CpuBufferPoolSubbuffer<CameraData, Arc<StandardMemoryPool>>>,
    ) -> Arc<SecondaryAutoCommandBuffer> {
        if self.command_buffer.is_none() {
            let instance_subbuffer = self
                .instance_buffer
                .from_iter(self.instances.clone().into_values())
                .unwrap();

            let mut builder = AutoCommandBufferBuilder::secondary(
                device,
                queue.queue_family_index(),
                CommandBufferUsage::SimultaneousUse,
                CommandBufferInheritanceInfo {
                    render_pass: Some(match pipeline.render_pass() {
                        PipelineRenderPassType::BeginRenderPass(render_pass) => {
                            render_pass.clone().into()
                        }
                        PipelineRenderPassType::BeginRendering(_) => panic!(),
                    }),
                    ..Default::default()
                },
            )
            .unwrap();

            let layout = pipeline.layout().set_layouts().get(0).unwrap();
            let set = PersistentDescriptorSet::new(
                layout.clone(),
                [WriteDescriptorSet::buffer(0, camera_buffer)],
            )
            .unwrap();

            builder
                .bind_pipeline_graphics(pipeline.clone())
                .bind_vertex_buffers(0, self.vertex_buffer.clone())
                .bind_vertex_buffers(1, instance_subbuffer)
                .bind_index_buffer(self.index_buffer.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline.layout().clone(),
                    0,
                    set,
                )
                .draw_indexed(
                    self.index_buffer.len() as u32,
                    self.instances.len() as u32,
                    0,
                    0,
                    0,
                )
                .unwrap();

            self.command_buffer = Some(Arc::new(builder.build().unwrap()));
        }

        self.command_buffer.clone().unwrap()
    }

    pub fn create_instance(&mut self, id: &str) -> &mut Instance {
        self.instances.insert(String::from(id), Instance::default());
        self.get_instance(id)
    }

    pub fn get_instance(&mut self, id: &str) -> &mut Instance {
        self.invalidate();
        self.instances.get_mut(id).unwrap()
    }

    fn invalidate(&mut self) {
        self.command_buffer = None;
    }
}

pub struct Group {
    objects: HashMap<String, Object>,
    pipeline: Arc<GraphicsPipeline>,
    vertex_shader: Arc<ShaderModule>,
    fragment_shader: Arc<ShaderModule>,
}

impl Group {
    pub fn new(
        device: Arc<Device>,
        vertex_shader: Arc<ShaderModule>,
        fragment_shader: Arc<ShaderModule>,
        render_pass: Arc<RenderPass>,
        viewport: Viewport,
    ) -> Self {
        let objects = HashMap::new();
        let pipeline = Self::create_pipeline(
            device,
            vertex_shader.clone(),
            fragment_shader.clone(),
            render_pass,
            viewport,
        );

        Self {
            objects,
            pipeline,
            vertex_shader,
            fragment_shader,
        }
    }

    pub fn recreate_pipeline(
        &mut self,
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        viewport: Viewport,
    ) {
        self.pipeline = Self::create_pipeline(
            device,
            self.vertex_shader.clone(),
            self.fragment_shader.clone(),
            render_pass,
            viewport,
        );
    }

    fn create_pipeline(
        device: Arc<Device>,
        vertex_shader: Arc<ShaderModule>,
        fragment_shader: Arc<ShaderModule>,
        render_pass: Arc<RenderPass>,
        viewport: Viewport,
    ) -> Arc<GraphicsPipeline> {
        GraphicsPipeline::start()
            .vertex_input_state(
                BuffersDefinition::new()
                    .vertex::<Vertex>()
                    .instance::<Instance>(),
            )
            .vertex_shader(vertex_shader.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
            .fragment_shader(fragment_shader.entry_point("main").unwrap(), ())
            .render_pass(Subpass::from(render_pass, 0).unwrap())
            .rasterization_state(RasterizationState {
                cull_mode: StateMode::Fixed(CullMode::Back),
                front_face: StateMode::Fixed(FrontFace::CounterClockwise),
                ..Default::default()
            })
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .build(device)
            .unwrap()
    }

    pub fn command_buffers(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        camera_buffer: Arc<CpuBufferPoolSubbuffer<CameraData, Arc<StandardMemoryPool>>>,
    ) -> Vec<Arc<SecondaryAutoCommandBuffer>> {
        self.objects
            .iter_mut()
            .map(|(_, object)| {
                object.command_buffer(
                    device.clone(),
                    queue.clone(),
                    self.pipeline.clone(),
                    camera_buffer.clone(),
                )
            })
            .collect()
    }

    pub fn create_object(&mut self, id: &str, mesh: Mesh, device: Arc<Device>) -> &mut Object {
        self.objects
            .insert(String::from(id), Object::new(mesh, device));
        self.get_object(id)
    }

    pub fn get_object(&mut self, id: &str) -> &mut Object {
        self.objects.get_mut(id).unwrap()
    }

    fn invalidate(&mut self) {
        for (_, object) in self.objects.iter_mut() {
            object.invalidate();
        }
    }
}

pub struct Scene {
    groups: HashMap<String, Group>,
    camera: Camera,

    command_buffers: Option<Vec<Arc<PrimaryAutoCommandBuffer>>>,
}

impl Scene {
    pub fn new(device: Arc<Device>) -> Self {
        let groups = HashMap::new();
        let camera = Camera::new(device);
        let command_buffers = None;

        Self {
            groups,
            camera,
            command_buffers,
        }
    }

    pub fn recreate_pipeline(
        &mut self,
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        viewport: Viewport,
    ) {
        for (_, group) in self.groups.iter_mut() {
            group.recreate_pipeline(device.clone(), render_pass.clone(), viewport.clone());
        }
    }

    pub fn command_buffers(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        framebuffers: &[Arc<Framebuffer>],
    ) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
        if self.command_buffers.is_none() {
            let camera_buffer = self.camera.subbuffer();

            self.command_buffers = Some(
                framebuffers
                    .iter()
                    .map(|framebuffer| {
                        let mut builder = AutoCommandBufferBuilder::primary(
                            device.clone(),
                            queue.queue_family_index(),
                            CommandBufferUsage::SimultaneousUse,
                        )
                        .unwrap();

                        builder
                            .begin_render_pass(
                                RenderPassBeginInfo {
                                    clear_values: vec![
                                        Some([0.3, 0.3, 0.3, 1.0].into()),
                                        Some(1.0.into()),
                                    ],
                                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                                },
                                SubpassContents::SecondaryCommandBuffers,
                            )
                            .unwrap();

                        for (_, group) in self.groups.iter_mut() {
                            builder
                                .execute_commands_from_vec(group.command_buffers(
                                    device.clone(),
                                    queue.clone(),
                                    camera_buffer.clone(),
                                ))
                                .unwrap();
                        }

                        builder.end_render_pass().unwrap();

                        Arc::new(builder.build().unwrap())
                    })
                    .collect(),
            )
        }

        self.command_buffers.clone().unwrap()
    }

    pub fn create_group(
        &mut self,
        id: &str,
        device: Arc<Device>,
        vertex_shader: Arc<ShaderModule>,
        fragment_shader: Arc<ShaderModule>,
        render_pass: Arc<RenderPass>,
        viewport: Viewport,
    ) -> &mut Group {
        self.groups.insert(
            String::from(id),
            Group::new(
                device,
                vertex_shader,
                fragment_shader,
                render_pass,
                viewport,
            ),
        );
        self.get_group(id)
    }

    pub fn get_group(&mut self, id: &str) -> &mut Group {
        self.invalidate();
        self.groups.get_mut(id).unwrap()
    }

    pub fn get_camera(&mut self) -> &mut Camera {
        self.invalidate_all();
        &mut self.camera
    }

    fn invalidate(&mut self) {
        self.command_buffers = None;
    }

    fn invalidate_all(&mut self) {
        self.invalidate();
        self.camera.invalidate();
        for (_, group) in self.groups.iter_mut() {
            group.invalidate()
        }
    }
}
