use std::{sync::Arc};



use vulkano::{
    device::{
        physical::{PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
    },
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, ImageUsage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
    pipeline::{
        graphics::{
            viewport::{Viewport},
        },
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    swapchain::{
        self, AcquireError, PresentInfo, PresentMode, Surface, Swapchain, SwapchainCreateInfo,
        SwapchainCreationError,
    },
    sync::{self, FenceSignalFuture, FlushError, GpuFuture},
    VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder, CursorGrabMode, Fullscreen}, dpi::PhysicalSize,
};

use crate::{
    scene::{Scene},
};

pub struct Engine {
    pub device: Arc<Device>,
    pub scene: Scene,
    pub render_pass: Arc<RenderPass>,
    pub viewport: Viewport,
    
    event_loop: EventLoop<()>,
    surface: Arc<Surface<Window>>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain<Window>>,
    images: Vec<Arc<SwapchainImage<Window>>>,
    framebuffers: Vec<Arc<Framebuffer>>,
    depth_buffer: Arc<ImageView<AttachmentImage>>,
}

impl Engine {
    fn create_frambuffers(
        images: &[Arc<SwapchainImage<Window>>],
        render_pass: Arc<RenderPass>,
        depth_buffer: Arc<ImageView<AttachmentImage>>,
    ) -> Vec<Arc<Framebuffer>> {
        images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view, depth_buffer.clone()],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect()
    }

    pub fn run<H: FnMut(&Event<()>, &mut Scene) + 'static>(mut self, mut update_handler: H) {
        let mut window_resized = false;
        let mut recreate_swapchain = false;

        let frames_in_flight = self.images.len();
        let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
        let mut previous_fence_i = 0;

        self.event_loop
            .run(move |event, _, control_flow| {
                update_handler(&event, &mut self.scene);
                
                match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => *control_flow = ControlFlow::Exit,
                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => window_resized = true,
                Event::MainEventsCleared => {
                    if window_resized || recreate_swapchain {
                        recreate_swapchain = false;
                        let new_dimensions = self.surface.window().inner_size();

                        let (new_swapchain, new_images) =
                            match self.swapchain.recreate(SwapchainCreateInfo {
                                image_extent: new_dimensions.into(),
                                ..self.swapchain.create_info()
                            }) {
                                Ok(r) => r,
                                Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => {
                                    return
                                }
                                Err(e) => panic!("{:?}", e),
                            };

                        self.swapchain = new_swapchain;
                        self.framebuffers = Self::create_frambuffers(
                            &new_images,
                            self.render_pass.clone(),
                            self.depth_buffer.clone(),
                        );

                        if window_resized {
                            window_resized = false;

                            self.viewport.dimensions = new_dimensions.into();

                            let [width, height] = self.viewport.dimensions;
                            self.scene.get_camera().update(|configuration| configuration.aspect = width / height);

                            self.scene.recreate_pipeline(
                                self.device.clone(),
                                self.render_pass.clone(),
                                self.viewport.clone(),
                            );
                        }
                    }

                    let (image_i, suboptimal, acquire_future) =
                        match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                            Ok(r) => r,
                            Err(AcquireError::OutOfDate) => {
                                recreate_swapchain = true;
                                return;
                            }
                            Err(e) => panic!("{:?}", e),
                        };

                    if suboptimal {
                        recreate_swapchain = true;
                    }

                    if let Some(image_fence) = &fences[image_i] {
                        image_fence.wait(None).unwrap();
                    }

                    let previous_future = match fences[previous_fence_i].clone() {
                        None => {
                            let mut now = sync::now(self.device.clone());
                            now.cleanup_finished();
                            now.boxed()
                        }
                        Some(fence) => fence.boxed(),
                    };

                    let future = previous_future
                        .join(acquire_future)
                        .then_execute(
                            self.queue.clone(),
                            self.scene.command_buffers(
                                self.device.clone(),
                                self.queue.clone(),
                                &self.framebuffers,
                            )[image_i]
                                .clone(),
                        )
                        .unwrap()
                        .then_swapchain_present(
                            self.queue.clone(),
                            PresentInfo {
                                index: image_i,
                                ..PresentInfo::swapchain(self.swapchain.clone())
                            },
                        )
                        .then_signal_fence_and_flush();

                    fences[image_i] = match future {
                        Ok(future) => Some(Arc::new(future)),
                        Err(FlushError::OutOfDate) => {
                            recreate_swapchain = true;
                            None
                        }
                        Err(e) => {
                            println!("{:?}", e);
                            None
                        }
                    };

                    previous_fence_i = image_i;
                }
                _ => (),
            }});
    }
}

pub struct EngineBuilder {
    library: Arc<VulkanLibrary>,
    instance_create_info: InstanceCreateInfo,
}

impl EngineBuilder {
    pub fn new() -> Self {
        let library = VulkanLibrary::new().unwrap();
        let instance_create_info = Default::default();
        
        Self {
            library,
            instance_create_info,
        }
    }

    pub fn instance_with_required_extensions(mut self) -> Self {
        let extensions = vulkano_win::required_extensions(&self.library);
        self.instance_create_info.enabled_extensions = extensions;
        self
    }

    pub fn build(self) -> Engine {
        let instance = Instance::new(self.library, self.instance_create_info).unwrap();
        let event_loop = EventLoop::new();

        let surface = WindowBuilder::new()
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap();
        
        surface.window()
            .set_cursor_grab(CursorGrabMode::Confined)
            .unwrap();
        
        let physical_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let (physical, family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&physical_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.graphics
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();

        let (device, mut queues) = Device::new(
            physical,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: family_index,
                    ..Default::default()
                }],
                enabled_extensions: physical_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let caps = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let dimensions = surface.window().inner_size();
        let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let image_format = Some(
            device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count + 1,
                image_format,
                image_extent: dimensions.into(),
                image_usage: ImageUsage {
                    color_attachment: true,
                    ..ImageUsage::empty()
                },
                composite_alpha,
                present_mode: PresentMode::Fifo,
                ..Default::default()
            },
        )
        .unwrap();

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.image_format(),
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16_UNORM,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        )
        .unwrap();

        let depth_buffer = ImageView::new_default(
            AttachmentImage::with_usage(
                device.clone(),
                images[0].dimensions().width_height(),
                Format::D16_UNORM,
                ImageUsage {
                    depth_stencil_attachment: true,
                    transient_attachment: true,
                    ..ImageUsage::empty()
                },
            )
            .unwrap(),
        )
        .unwrap();

        let framebuffers: Vec<_> =
            Engine::create_frambuffers(&images, render_pass.clone(), depth_buffer.clone());


        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [1024.0, 1024.0],
            depth_range: 0.0..1.0,
        };

        /*
        let mesh = Mesh {
            vertices: vec![
                Vertex {
                    coord: Vector3::new(0.0, -0.5, 0.5),
                    normal: Vector3::new(1.0, 0.0, 0.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                },
                Vertex {
                    coord: Vector3::new(-0.5, 0.5, 0.5),
                    normal: Vector3::new(0.0, 1.0, 0.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                },
                Vertex {
                    coord: Vector3::new(0.5, 0.5, 0.5),
                    normal: Vector3::new(0.0, 0.0, 1.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                },
                /*
                Vertex {
                    coord: Vector3::new(0.5, 0.5, 0.5),
                    normal: Vector3::new(1.0, 0.0, 0.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                },
                */
            ],
            indices: vec![0u32, 1u32, 2u32 /*, 2u32, 1u32, 3u32*/],
        };
        let mesh = Mesh {
            vertices: vec![
                Vertex {
                    coord: Vector3::new(-1.0, -1.0, -1.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                    normal: Vector3::new(0.0, 0.0, 0.0),
                }, //  0: back-bottom-left   (back-face)
                Vertex {
                    coord: Vector3::new(-1.0, -1.0, 1.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                    normal: Vector3::new(0.0, 0.0, 1.0),
                }, //  1: front-bottom-left  (front-face)
                Vertex {
                    coord: Vector3::new(-1.0, 1.0, -1.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                    normal: Vector3::new(0.0, 1.0, 0.0),
                }, //  2: back-top-left      (back-face)
                Vertex {
                    coord: Vector3::new(-1.0, 1.0, 1.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                    normal: Vector3::new(0.0, 1.0, 1.0),
                }, //  3: front-top-left     (front-face)
                Vertex {
                    coord: Vector3::new(1.0, -1.0, -1.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                    normal: Vector3::new(1.0, 0.0, 0.0),
                }, //  4: back-bottom-right  (back-face)
                Vertex {
                    coord: Vector3::new(1.0, -1.0, 1.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                    normal: Vector3::new(1.0, 0.0, 1.0),
                }, //  5: front-bottom-right (front-face)
                Vertex {
                    coord: Vector3::new(1.0, 1.0, -1.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                    normal: Vector3::new(1.0, 1.0, 0.0),
                }, //  6: back-top-right     (back-face)
                Vertex {
                    coord: Vector3::new(1.0, 1.0, 1.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                    normal: Vector3::new(1.0, 1.0, 1.0),
                }, //  7: front-top-right    (front-face)
                Vertex {
                    coord: Vector3::new(-1.0, -1.0, -1.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                    normal: Vector3::new(0.0, 0.0, 0.0),
                }, //  8: back-bottom-left   (bottom-face)
                Vertex {
                    coord: Vector3::new(-1.0, 1.0, -1.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                    normal: Vector3::new(0.0, 1.0, 0.0),
                }, //  9: back-top-left      (top-face)
                Vertex {
                    coord: Vector3::new(1.0, -1.0, -1.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                    normal: Vector3::new(1.0, 0.0, 0.0),
                }, // 10: back-bottom-right  (bottom-face)
                Vertex {
                    coord: Vector3::new(1.0, 1.0, -1.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                    normal: Vector3::new(1.0, 1.0, 0.0),
                }, // 11: back-top-right     (top-face)
                Vertex {
                    coord: Vector3::new(-1.0, -1.0, -1.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                    normal: Vector3::new(0.0, 0.0, 0.0),
                }, // 12: back-bottom-left   (left-face)
                Vertex {
                    coord: Vector3::new(-1.0, 1.0, -1.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                    normal: Vector3::new(0.0, 1.0, 0.0),
                }, // 13: back-top-left      (left-face)
            ],
            indices: vec![
                1, 7, 3, 1, 5, 7, // front
                4, 2, 6, 4, 0, 2, // back
                5, 6, 7, 5, 4, 6, // right
                12, 3, 13, 12, 1, 3, // left
                3, 11, 9, 3, 7, 11, // top
                8, 5, 1, 8, 10, 5, // bottom
            ],
        };
            */

        let scene = Scene::new(
            device.clone(),
        );

        Engine {
            event_loop,
            surface,
            device,
            queue,
            render_pass,
            swapchain,
            images,
            viewport,
            framebuffers,
            scene,
            depth_buffer,
        }
    }
}
