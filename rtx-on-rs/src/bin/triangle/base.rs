use super::rtx::*;
use rtx_on_rs::camera::*;
use rtx_on_rs::config::*;

use ash::extensions::nv::RayTracing;
use ash::{version::DeviceV1_0, vk, Device};
use cgmath::{Deg, Matrix4};
use std::sync::Arc;
use vulkan::*;
use winit::{dpi::LogicalSize, Event, EventsLoop, Window, WindowBuilder, WindowEvent};

const MAX_FRAMES_IN_FLIGHT: u32 = 2;

pub struct BaseApp {
    events_loop: EventsLoop,
    _window: Window,
    resize_dimensions: Option<[u32; 2]>,

    context: Arc<Context>,
    swapchain_properties: SwapchainProperties,
    depth_format: vk::Format,
    msaa_samples: vk::SampleCountFlags,
    render_pass: RenderPass,
    swapchain: Swapchain,

    in_flight_frames: InFlightFrames,

    rtx_data: RTXData,
}

impl BaseApp {
    pub fn new() -> Self {
        log::info!("Creating application.");

        let events_loop = EventsLoop::new();
        let window = WindowBuilder::new()
            .with_title("RTX On rs")
            .with_dimensions(LogicalSize::new(
                f64::from(RESOLUTION[0]),
                f64::from(RESOLUTION[1]),
            ))
            .build(&events_loop)
            .unwrap();

        let context = Arc::new(Context::new(&window));

        let swapchain_support_details = SwapchainSupportDetails::new(
            context.physical_device(),
            context.surface(),
            context.surface_khr(),
        );
        let swapchain_properties =
            swapchain_support_details.get_ideal_swapchain_properties(RESOLUTION, VSYNC);
        let depth_format = Self::find_depth_format(&context);
        let msaa_samples = context.get_max_usable_sample_count(MSAA);

        let render_pass = RenderPass::create(
            Arc::clone(&context),
            swapchain_properties.extent,
            swapchain_properties.format.format,
            depth_format,
            msaa_samples,
        );

        let swapchain = Swapchain::create(
            Arc::clone(&context),
            swapchain_support_details,
            RESOLUTION,
            VSYNC,
            &render_pass,
        );

        let in_flight_frames = Self::create_sync_objects(context.device());

        let camera = Self::create_camera(swapchain_properties);

        let rt_props =
            unsafe { RayTracing::get_properties(context.instance(), context.physical_device()) };
        log::debug!("Ray tracing props: {:#?}", rt_props);

        let rtx_data = RTXData::new(&context, &swapchain, camera);

        Self {
            events_loop,
            _window: window,
            resize_dimensions: None,
            context,
            swapchain_properties,
            render_pass,
            swapchain,
            depth_format,
            msaa_samples,
            in_flight_frames,
            rtx_data,
        }
    }

    fn find_depth_format(context: &Context) -> vk::Format {
        let candidates = vec![
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];
        context
            .find_supported_format(
                &candidates,
                vk::ImageTiling::OPTIMAL,
                vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
            )
            .expect("Failed to find a supported depth format")
    }

    fn create_sync_objects(device: &Device) -> InFlightFrames {
        let mut sync_objects_vec = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let image_available_semaphore = {
                let semaphore_info = vk::SemaphoreCreateInfo::builder();
                unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
            };

            let render_finished_semaphore = {
                let semaphore_info = vk::SemaphoreCreateInfo::builder();
                unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
            };

            let in_flight_fence = {
                let fence_info =
                    vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
                unsafe { device.create_fence(&fence_info, None).unwrap() }
            };

            let sync_objects = SyncObjects {
                image_available_semaphore,
                render_finished_semaphore,
                fence: in_flight_fence,
            };
            sync_objects_vec.push(sync_objects)
        }

        InFlightFrames::new(sync_objects_vec)
    }

    fn create_camera(swapchain_properties: SwapchainProperties) -> Camera {
        let view = Matrix4::look_at(
            [0.0, 0.0, -2.0].into(),
            [0.0, 0.0, 1.0].into(),
            [0.0, 1.0, 0.0].into(),
        );

        let aspect =
            swapchain_properties.extent.width as f32 / swapchain_properties.extent.height as f32;
        let projection = math::perspective(Deg(60.0), aspect, 0.1, 10.0);

        Camera::new(view, projection)
    }

    pub fn run(&mut self) {
        log::info!("Running application.");
        loop {
            if self.process_event() {
                break;
            }

            self.draw_frame();
        }
        unsafe { self.context.device().device_wait_idle().unwrap() };
    }

    /// Process the events from the `EventsLoop` and return whether the
    /// main loop should stop.
    fn process_event(&mut self) -> bool {
        let mut should_stop = false;
        let mut resize_dimensions = None;

        self.events_loop.poll_events(|event| {
            if let Event::WindowEvent { event, .. } = event {
                match event {
                    WindowEvent::CloseRequested => should_stop = true,
                    WindowEvent::Resized(LogicalSize { width, height }) => {
                        resize_dimensions = Some([width as u32, height as u32]);
                    }
                    _ => {}
                }
            }
        });

        self.resize_dimensions = resize_dimensions;
        should_stop
    }

    fn draw_frame(&mut self) {
        log::trace!("Drawing frame.");
        let sync_objects = self.in_flight_frames.next().unwrap();
        let image_available_semaphore = sync_objects.image_available_semaphore;
        let render_finished_semaphore = sync_objects.render_finished_semaphore;
        let in_flight_fence = sync_objects.fence;
        let wait_fences = [in_flight_fence];

        unsafe {
            self.context
                .device()
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .unwrap()
        };

        let result = self
            .swapchain
            .acquire_next_image(None, Some(image_available_semaphore), None);
        let image_index = match result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain();
                return;
            }
            Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
        };

        unsafe { self.context.device().reset_fences(&wait_fences).unwrap() };

        let device = self.context.device();
        let wait_semaphores = [image_available_semaphore];
        let signal_semaphores = [render_finished_semaphore];

        // Submit command buffer
        {
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [self.rtx_data.get_command_buffer(image_index as _)];
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores)
                .build();
            let submit_infos = [submit_info];
            unsafe {
                device
                    .queue_submit(
                        self.context.graphics_queue(),
                        &submit_infos,
                        in_flight_fence,
                    )
                    .unwrap()
            };
        }

        let swapchains = [self.swapchain.swapchain_khr()];
        let images_indices = [image_index];

        {
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&images_indices);
            let result = self.swapchain.present(&present_info);
            match result {
                Ok(is_suboptimal) if is_suboptimal => {
                    self.recreate_swapchain();
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain();
                }
                Err(error) => panic!("Failed to present queue. Cause: {}", error),
                _ => {}
            }

            if self.resize_dimensions.is_some() {
                self.recreate_swapchain();
            }
        }
    }

    /// Recreates the swapchain.
    ///
    /// If the window has been resized, then the new size is used
    /// otherwise, the size of the current swapchain is used.
    ///
    /// If the window has been minimized, then the functions block until
    /// the window is maximized. This is because a width or height of 0
    /// is not legal.
    fn recreate_swapchain(&mut self) {
        log::debug!("Recreating swapchain.");

        if self.has_window_been_minimized() {
            while !self.has_window_been_maximized() {
                self.process_event();
            }
        }

        unsafe { self.context.device().device_wait_idle().unwrap() };

        self.cleanup_swapchain();

        let dimensions = self.resize_dimensions.unwrap_or([
            self.swapchain.properties().extent.width,
            self.swapchain.properties().extent.height,
        ]);

        let swapchain_support_details = SwapchainSupportDetails::new(
            self.context.physical_device(),
            self.context.surface(),
            self.context.surface_khr(),
        );
        let swapchain_properties =
            swapchain_support_details.get_ideal_swapchain_properties(dimensions, VSYNC);

        let render_pass = RenderPass::create(
            Arc::clone(&self.context),
            swapchain_properties.extent,
            swapchain_properties.format.format,
            self.depth_format,
            self.msaa_samples,
        );

        let swapchain = Swapchain::create(
            Arc::clone(&self.context),
            swapchain_support_details,
            dimensions,
            VSYNC,
            &render_pass,
        );

        let camera = Self::create_camera(swapchain_properties);

        let rtx_data = RTXData::new(&self.context, &swapchain, camera);

        self.swapchain = swapchain;
        self.swapchain_properties = swapchain_properties;
        self.render_pass = render_pass;
        self.rtx_data = rtx_data;
    }

    fn has_window_been_minimized(&self) -> bool {
        match self.resize_dimensions {
            Some([x, y]) if x == 0 || y == 0 => true,
            _ => false,
        }
    }

    fn has_window_been_maximized(&self) -> bool {
        match self.resize_dimensions {
            Some([x, y]) if x > 0 && y > 0 => true,
            _ => false,
        }
    }

    /// Clean up the swapchain and all resources that depends on it.
    fn cleanup_swapchain(&mut self) {
        self.swapchain.destroy();
    }
}

impl Drop for BaseApp {
    fn drop(&mut self) {
        log::debug!("Dropping application.");
        self.cleanup_swapchain();
        let device = self.context.device();
        self.in_flight_frames.destroy(device);
    }
}

#[derive(Clone, Copy)]
struct SyncObjects {
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    fence: vk::Fence,
}

impl SyncObjects {
    fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.image_available_semaphore, None);
            device.destroy_semaphore(self.render_finished_semaphore, None);
            device.destroy_fence(self.fence, None);
        }
    }
}

struct InFlightFrames {
    sync_objects: Vec<SyncObjects>,
    current_frame: usize,
}

impl InFlightFrames {
    fn new(sync_objects: Vec<SyncObjects>) -> Self {
        Self {
            sync_objects,
            current_frame: 0,
        }
    }

    fn destroy(&self, device: &Device) {
        self.sync_objects.iter().for_each(|o| o.destroy(&device));
    }
}

impl Iterator for InFlightFrames {
    type Item = SyncObjects;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.sync_objects[self.current_frame];

        self.current_frame = (self.current_frame + 1) % self.sync_objects.len();

        Some(next)
    }
}