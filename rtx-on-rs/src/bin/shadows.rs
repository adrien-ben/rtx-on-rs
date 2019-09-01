use rtx::*;
use rtx_on_rs::base::*;
use rtx_on_rs::camera::*;

use cgmath::{Deg, Matrix4};
use vulkan::SwapchainProperties;

fn main() {
    env_logger::init();
    ShadowsApp::new().run();
}

struct ShadowsApp {
    base: BaseApp,
    rtx_data: RTXData,
}

impl ShadowsApp {
    fn new() -> Self {
        let base = BaseApp::new(Default::default());

        let swapchain = base.swapchain();
        let camera = Self::create_camera(swapchain.properties());
        let rtx_data = RTXData::new(base.context(), swapchain, camera);

        ShadowsApp { base, rtx_data }
    }

    fn create_camera(swapchain_properties: SwapchainProperties) -> Camera {
        let view = Matrix4::look_at(
            [-0.3, 0.3, -0.5].into(),
            [0.3, -0.3, 0.5].into(),
            [0.0, 1.0, 0.0].into(),
        );

        let aspect =
            swapchain_properties.extent.width as f32 / swapchain_properties.extent.height as f32;
        let projection = math::perspective(Deg(60.0), aspect, 0.1, 10.0);

        Camera::new(view, projection)
    }

    fn run(&mut self) {
        self.base.prepare_run();
        loop {
            if self.base.process_event() {
                break;
            }
            self.draw_frame();
        }
        self.base.finish_run();
    }

    fn draw_frame(&mut self) {
        let swapchain_properties = self.base.draw_frame(self.rtx_data.get_command_buffers());
        if let Some(swapchain_properties) = swapchain_properties {
            let swapchain = self.base.swapchain();
            let camera = Self::create_camera(swapchain_properties);
            self.rtx_data = RTXData::new(self.base.context(), swapchain, camera);
        }
    }
}

mod rtx {
    use ash::extensions::nv::RayTracing;
    use ash::version::DeviceV1_0;
    use ash::vk;
    use cgmath::{Matrix4, SquareMatrix};
    use raytracing::acceleration_structure::*;
    use raytracing::geometry_instance::*;
    use rtx_on_rs::camera::*;
    use std::ffi::CString;
    use std::mem::size_of;
    use std::path::Path;
    use std::sync::Arc;
    use tobj::*;
    use vulkan::*;

    const SHADER_FOLDER: &str = "shaders/shadows";

    #[derive(Copy, Clone, Debug)]
    #[repr(C)]
    struct Vertex {
        position: [f32; 4],
        normal: [f32; 4],
    }

    #[allow(dead_code)]
    pub struct RTXData {
        context: Arc<Context>,
        output_texture: Texture,
        uniform_buffer: Buffer,
        vertices: Buffer,
        indices: Buffer,
        bottom_as: AccelerationStructure,
        top_as: AccelerationStructure,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        descriptors: Descriptors,
        shader_binding_table_buffer: Buffer,
        command_buffers: Vec<vk::CommandBuffer>,
    }

    impl RTXData {
        pub fn get_command_buffers(&self) -> &[vk::CommandBuffer] {
            &self.command_buffers
        }
    }

    impl RTXData {
        pub fn new(context: &Arc<Context>, swapchain: &Swapchain, camera: Camera) -> Self {
            let output_texture = create_output_texture(context, swapchain);

            let uniform_buffer = create_uniform_buffer(context, camera);

            let (bottom_as, top_as, vertices, indices) = build_acceleration_structures(context);

            let (pipeline, pipeline_layout, descriptor_set_layout) = create_pipeline(context);

            let descriptors = create_descriptors(
                context,
                descriptor_set_layout,
                &top_as,
                &output_texture,
                &uniform_buffer,
                &vertices,
                &indices,
            );

            let rt_props = unsafe {
                RayTracing::get_properties(context.instance(), context.physical_device())
            };

            let shader_binding_table_buffer =
                create_shader_binding_table(context, pipeline, rt_props);

            let mut rtx = Self {
                context: Arc::clone(context),
                output_texture,
                uniform_buffer,
                vertices,
                indices,
                bottom_as,
                top_as,
                pipeline,
                pipeline_layout,
                descriptors,
                shader_binding_table_buffer,
                command_buffers: Vec::new(),
            };
            rtx.create_and_record_command_buffers(rt_props, swapchain);

            rtx
        }

        fn create_and_record_command_buffers(
            &mut self,
            rt_properties: vk::PhysicalDeviceRayTracingPropertiesNV,
            swapchain: &Swapchain,
        ) {
            let device = self.context.device();
            let image_count = swapchain.image_count();

            {
                let allocate_info = vk::CommandBufferAllocateInfo::builder()
                    .command_pool(self.context.general_command_pool())
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(image_count as _);

                let buffers = unsafe {
                    device
                        .allocate_command_buffers(&allocate_info)
                        .expect("Failed to allocate command buffers")
                };
                self.command_buffers.extend_from_slice(&buffers);
            };

            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

            self.command_buffers
                .iter()
                .enumerate()
                .for_each(|(index, buffer)| {
                    let buffer = *buffer;
                    let swapchain_image = &swapchain.images()[index];

                    // begin command buffer
                    unsafe {
                        device
                            .begin_command_buffer(buffer, &command_buffer_begin_info)
                            .expect("Failed to begin command buffer")
                    };

                    // Bind pipeline
                    unsafe {
                        device.cmd_bind_pipeline(
                            buffer,
                            vk::PipelineBindPoint::RAY_TRACING_NV,
                            self.pipeline,
                        )
                    };

                    // Bind descriptor set
                    unsafe {
                        device.cmd_bind_descriptor_sets(
                            buffer,
                            vk::PipelineBindPoint::RAY_TRACING_NV,
                            self.pipeline_layout,
                            0,
                            &self.descriptors.sets(),
                            &[],
                        );
                    };

                    let swapchain_props = swapchain.properties();

                    // Trace rays
                    let shader_group_handle_size = rt_properties.shader_group_handle_size;
                    let raygen_offset = 0;
                    let miss_offset = shader_group_handle_size;
                    let hit_offset = 3 * shader_group_handle_size;

                    unsafe {
                        let sbt_buffer = self.shader_binding_table_buffer.buffer;
                        self.context.ray_tracing().cmd_trace_rays(
                            buffer,
                            sbt_buffer,
                            raygen_offset,
                            sbt_buffer,
                            miss_offset.into(),
                            shader_group_handle_size.into(),
                            sbt_buffer,
                            hit_offset.into(),
                            shader_group_handle_size.into(),
                            vk::Buffer::null(),
                            0,
                            0,
                            swapchain_props.extent.width,
                            swapchain_props.extent.height,
                            1,
                        );
                    };

                    // Copy output image to swapchain
                    {
                        // transition layouts
                        swapchain_image.cmd_transition_image_layout(
                            buffer,
                            vk::ImageLayout::UNDEFINED,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        );
                        self.output_texture.image.cmd_transition_image_layout(
                            buffer,
                            vk::ImageLayout::GENERAL,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        );

                        // Copy image
                        let image_copy_info = [vk::ImageCopy::builder()
                            .src_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: 0,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .src_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                            .dst_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: 0,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .dst_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                            .extent(vk::Extent3D {
                                width: swapchain_props.extent.width,
                                height: swapchain_props.extent.height,
                                depth: 1,
                            })
                            .build()];

                        unsafe {
                            device.cmd_copy_image(
                                buffer,
                                self.output_texture.image.image,
                                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                                swapchain_image.image,
                                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                                &image_copy_info,
                            );
                        };

                        // Transition layout
                        swapchain_image.cmd_transition_image_layout(
                            buffer,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            vk::ImageLayout::PRESENT_SRC_KHR,
                        );
                        self.output_texture.image.cmd_transition_image_layout(
                            buffer,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            vk::ImageLayout::GENERAL,
                        );
                    }

                    // End command buffer
                    unsafe {
                        device
                            .end_command_buffer(buffer)
                            .expect("Failed to end command buffer")
                    };
                });
        }
    }

    fn create_output_texture(context: &Arc<Context>, swapchain: &Swapchain) -> Texture {
        let swapchain_props = swapchain.properties();

        let params = ImageParameters {
            mem_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent: swapchain_props.extent,
            format: swapchain_props.format.format,
            usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::STORAGE,
            ..Default::default()
        };
        let image = Image::create(Arc::clone(context), params);
        let view = image.create_view(vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::COLOR);
        image.transition_image_layout(vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);

        Texture::new(Arc::clone(context), image, view, None)
    }

    fn create_uniform_buffer(context: &Arc<Context>, camera: Camera) -> Buffer {
        #[repr(C)]
        struct UniformBufferData {
            view_inverse: Matrix4<f32>,
            projection_inverse: Matrix4<f32>,
        };

        let data = UniformBufferData {
            view_inverse: camera.view().invert().unwrap(),
            projection_inverse: camera.projection().invert().unwrap(),
        };
        let data = unsafe { any_as_u8_slice(&data) };
        create_host_visible_buffer(context, vk::BufferUsageFlags::UNIFORM_BUFFER, data)
    }

    fn build_acceleration_structures(
        context: &Arc<Context>,
    ) -> (AccelerationStructure, AccelerationStructure, Buffer, Buffer) {
        let (vertices, indices) = load_model("assets/models/dog/dog.obj");

        // Vertex data buffers
        let vertex_count = vertices.len();
        let vertices = create_device_local_buffer_with_data::<u8, _>(
            context,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
            &vertices,
        );

        let index_count = indices.len();
        let indices = create_device_local_buffer_with_data::<u8, _>(
            context,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
            &indices,
        );

        // Bottom acceleration structure
        let geometries = [vk::GeometryNV::builder()
            .geometry_type(vk::GeometryTypeNV::TRIANGLES)
            .geometry(
                vk::GeometryDataNV::builder()
                    .triangles(
                        vk::GeometryTrianglesNV::builder()
                            .vertex_data(vertices.buffer)
                            .vertex_count(vertex_count as _)
                            .vertex_offset(0)
                            .vertex_stride(size_of::<Vertex>() as _)
                            .vertex_format(vk::Format::R32G32B32A32_SFLOAT)
                            .index_data(indices.buffer)
                            .index_count(index_count as _)
                            .index_offset(0)
                            .index_type(vk::IndexType::UINT32)
                            .transform_data(vk::Buffer::null())
                            .transform_offset(0)
                            .build(),
                    )
                    .build(),
            )
            .flags(vk::GeometryFlagsNV::OPAQUE)
            .build()];

        let bottom_as = AccelerationStructure::create_bottom(Arc::clone(context), &geometries);

        // Top acceleration structure
        let t = Matrix4::identity();
        let transform = [
            t.x.x, t.x.y, t.x.z, t.x.w, t.y.x, t.y.y, t.y.z, t.y.w, t.z.x, t.z.y, t.z.z, t.z.w,
        ];

        let instance_buffer = {
            let geometry_instance = GeometryInstance {
                transform,
                instance_custom_index: 0,
                mask: 0xff,
                instance_offset: 0,
                flags: vk::GeometryInstanceFlagsNV::TRIANGLE_CULL_DISABLE,
                acceleration_structure_handle: bottom_as.handle,
            };
            let geometry_instance = geometry_instance.get_data();
            unsafe {
                create_device_local_buffer_with_data::<u8, _>(
                    context,
                    vk::BufferUsageFlags::RAY_TRACING_NV,
                    any_as_u8_slice(&geometry_instance),
                )
            }
        };
        let top_as = AccelerationStructure::create_top(Arc::clone(context), 1);

        // Build acceleration structure
        let bottom_mem_requirements = bottom_as.get_memory_requirements(
            vk::AccelerationStructureMemoryRequirementsTypeNV::BUILD_SCRATCH,
        );
        let top_mem_requirements = top_as.get_memory_requirements(
            vk::AccelerationStructureMemoryRequirementsTypeNV::BUILD_SCRATCH,
        );

        let scratch_buffer_size = bottom_mem_requirements
            .memory_requirements
            .size
            .max(top_mem_requirements.memory_requirements.size);
        let scratch_buffer = Buffer::create(
            Arc::clone(context),
            scratch_buffer_size,
            vk::BufferUsageFlags::RAY_TRACING_NV,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        context.execute_one_time_commands(|command_buffer| {
            // Build bottom AS
            bottom_as.cmd_build(command_buffer, &scratch_buffer, None);

            let memory_barrier = [vk::MemoryBarrier::builder()
                .src_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_READ_NV
                        | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_NV,
                )
                .dst_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_READ_NV
                        | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_NV,
                )
                .build()];
            unsafe {
                context.device().cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
                    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
                    vk::DependencyFlags::empty(),
                    &memory_barrier,
                    &[],
                    &[],
                )
            };

            // Build top AS
            top_as.cmd_build(command_buffer, &scratch_buffer, Some(&instance_buffer));

            unsafe {
                context.device().cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
                    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
                    vk::DependencyFlags::empty(),
                    &memory_barrier,
                    &[],
                    &[],
                )
            };
        });

        (bottom_as, top_as, vertices, indices)
    }

    /// Load a model, scale it so its biggest dimension is 1 center it around (0, 0, 0)
    fn load_model<P: AsRef<Path>>(path: P) -> (Vec<Vertex>, Vec<u32>) {
        let (models, _) = load_obj(path.as_ref()).expect("Failed to load obj model");

        // Compute model bounds
        let mut min_x = std::f32::MAX;
        let mut min_y = std::f32::MAX;
        let mut min_z = std::f32::MAX;

        let mut max_x = std::f32::MIN;
        let mut max_y = std::f32::MIN;
        let mut max_z = std::f32::MIN;

        for model in &models {
            let mesh = &model.mesh;

            for index in 0..mesh.positions.len() / 3 {
                let x = mesh.positions[index * 3];
                let y = mesh.positions[index * 3 + 1];
                let z = mesh.positions[index * 3 + 2];

                if x < min_x {
                    min_x = x;
                }
                if x > max_x {
                    max_x = x;
                }

                if y < min_y {
                    min_y = y;
                }
                if y > max_y {
                    max_y = y;
                }

                if z < min_z {
                    min_z = z;
                }
                if z > max_z {
                    max_z = z;
                }
            }
        }

        let width = max_x - min_x;
        let height = max_y - min_y;
        let depth = max_z - min_z;

        let x_offset = min_x + (width * 0.5);
        let y_offset = min_y + (height * 0.5);
        let z_offset = min_z + (depth * 0.5);

        fn max(v0: f32, v1: f32) -> f32 {
            if let Some(std::cmp::Ordering::Less) = v0.partial_cmp(&v1) {
                v1
            } else {
                v0
            }
        };

        let biggest_dimension = max(width, max(height, depth));

        // Build model
        let mut vertices = Vec::new();
        let mut indices = Vec::<u32>::new();
        for model in &models {

            let index_offset = vertices.len() as u32;
            let mesh = &model.mesh;

            for index in 0..mesh.positions.len() / 3 {
                let x = (mesh.positions[index * 3] - x_offset) / biggest_dimension;
                let y = (mesh.positions[index * 3 + 1] - y_offset) / biggest_dimension;
                let z = (mesh.positions[index * 3 + 2] - z_offset) / biggest_dimension;

                let nx = mesh.normals[index * 3];
                let ny = mesh.normals[index * 3 + 1];
                let nz = mesh.normals[index * 3 + 2];

                let v = Vertex {
                    position: [x, y, z, 0.0],
                    normal: [nx, ny, nz, 0.0],
                };

                vertices.push(v);
            }


            for index in &mesh.indices {
                indices.push(*index + index_offset);
            }

        }

        (vertices, indices)
    }

    fn create_pipeline(
        context: &Arc<Context>,
    ) -> (vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout) {
        let layout_bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_NV)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_NV | vk::ShaderStageFlags::CLOSEST_HIT_NV)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_NV)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_NV)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_NV)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_NV)
                .build(),
        ];

        let descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&layout_bindings);
        let descriptor_set_layout = unsafe {
            context
                .device()
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                .expect("Failed to create descriptor set layout")
        };
        let descriptor_set_layouts = [descriptor_set_layout];

        let pipeline_layout_create_info =
            vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts);
        let pipeline_layout = unsafe {
            context
                .device()
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .expect("Failed to create pipeline layout")
        };

        let entry_point_name = CString::new("main").unwrap();
        let raygen_shader_module = ShaderModule::new(
            Arc::clone(context),
            format!("{}/raygen.rgen.spv", SHADER_FOLDER),
        );
        let miss_shader_module = ShaderModule::new(
            Arc::clone(context),
            format!("{}/miss.rmiss.spv", SHADER_FOLDER),
        );
        let shadowmiss_shader_module = ShaderModule::new(
            Arc::clone(context),
            format!("{}/shadowmiss.rmiss.spv", SHADER_FOLDER),
        );
        let closesthit_shader_module = ShaderModule::new(
            Arc::clone(context),
            format!("{}/closesthit.rchit.spv", SHADER_FOLDER),
        );

        let shader_stages_create_info = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::RAYGEN_NV)
                .module(raygen_shader_module.module())
                .name(&entry_point_name)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::MISS_NV)
                .module(miss_shader_module.module())
                .name(&entry_point_name)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::MISS_NV)
                .module(shadowmiss_shader_module.module())
                .name(&entry_point_name)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::CLOSEST_HIT_NV)
                .module(closesthit_shader_module.module())
                .name(&entry_point_name)
                .build(),
        ];

        let shader_groups_create_info = [
            // Raygen
            vk::RayTracingShaderGroupCreateInfoNV::builder()
                .ty(vk::RayTracingShaderGroupTypeNV::GENERAL)
                .general_shader(0)
                .closest_hit_shader(vk::SHADER_UNUSED_NV)
                .any_hit_shader(vk::SHADER_UNUSED_NV)
                .intersection_shader(vk::SHADER_UNUSED_NV)
                .build(),
            // Miss
            vk::RayTracingShaderGroupCreateInfoNV::builder()
                .ty(vk::RayTracingShaderGroupTypeNV::GENERAL)
                .general_shader(1)
                .closest_hit_shader(vk::SHADER_UNUSED_NV)
                .any_hit_shader(vk::SHADER_UNUSED_NV)
                .intersection_shader(vk::SHADER_UNUSED_NV)
                .build(),
            // Shadow miss
            vk::RayTracingShaderGroupCreateInfoNV::builder()
                .ty(vk::RayTracingShaderGroupTypeNV::GENERAL)
                .general_shader(2)
                .closest_hit_shader(vk::SHADER_UNUSED_NV)
                .any_hit_shader(vk::SHADER_UNUSED_NV)
                .intersection_shader(vk::SHADER_UNUSED_NV)
                .build(),
            // Closest hit
            vk::RayTracingShaderGroupCreateInfoNV::builder()
                .ty(vk::RayTracingShaderGroupTypeNV::TRIANGLES_HIT_GROUP)
                .closest_hit_shader(3)
                .general_shader(vk::SHADER_UNUSED_NV)
                .any_hit_shader(vk::SHADER_UNUSED_NV)
                .intersection_shader(vk::SHADER_UNUSED_NV)
                .build(),
            // Shadow hit, will be skipped
            vk::RayTracingShaderGroupCreateInfoNV::builder()
                .ty(vk::RayTracingShaderGroupTypeNV::TRIANGLES_HIT_GROUP)
                .closest_hit_shader(3)
                .general_shader(vk::SHADER_UNUSED_NV)
                .any_hit_shader(vk::SHADER_UNUSED_NV)
                .intersection_shader(vk::SHADER_UNUSED_NV)
                .build(),
        ];

        let pipeline_create_info = [vk::RayTracingPipelineCreateInfoNV::builder()
            .stages(&shader_stages_create_info)
            .groups(&shader_groups_create_info)
            .max_recursion_depth(2)
            .layout(pipeline_layout)
            .build()];
        let pipeline = unsafe {
            context
                .ray_tracing()
                .create_ray_tracing_pipelines(
                    vk::PipelineCache::null(),
                    &pipeline_create_info,
                    None,
                )
                .expect("Failed to create pipeline")[0]
        };
        (pipeline, pipeline_layout, descriptor_set_layout)
    }

    fn create_shader_binding_table(
        context: &Arc<Context>,
        pipeline: vk::Pipeline,
        rt_properties: vk::PhysicalDeviceRayTracingPropertiesNV,
    ) -> Buffer {
        let shader_group_handle_size = rt_properties.shader_group_handle_size;
        let stb_size = shader_group_handle_size * 5;

        let mut shader_handles = Vec::new();
        shader_handles.resize(stb_size as _, 0u8);
        unsafe {
            context
                .ray_tracing()
                .get_ray_tracing_shader_group_handles(pipeline, 0, 5, &mut shader_handles)
                .expect("Failed to get rt shader group handles")
        };

        create_device_local_buffer_with_data::<u8, _>(
            context,
            vk::BufferUsageFlags::RAY_TRACING_NV,
            &shader_handles,
        )
    }

    fn create_descriptors(
        context: &Arc<Context>,
        descriptor_set_layout: vk::DescriptorSetLayout,
        top_as: &AccelerationStructure,
        output_texture: &Texture,
        uniform_buffer: &Buffer,
        vertex_buffer: &Buffer,
        index_buffer: &Buffer,
    ) -> Descriptors {
        let device = context.device();

        let descriptor_pool = {
            let pool_sizes = [
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_NV)
                    .descriptor_count(1)
                    .build(),
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .build(),
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .build(),
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(2)
                    .build(),
            ];
            let pool_create_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            unsafe {
                device
                    .create_descriptor_pool(&pool_create_info, None)
                    .expect("Failed to create descriptor pool")
            }
        };

        let descriptor_sets = {
            let set_layouts = [descriptor_set_layout];
            let allocate_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts);
            let sets = unsafe {
                device
                    .allocate_descriptor_sets(&allocate_info)
                    .expect("Failed to allocate descriptor set")
            };
            let acceleration_structures = [top_as.acceleration_structure];
            let mut as_set_info = vk::WriteDescriptorSetAccelerationStructureNV::builder()
                .acceleration_structures(&acceleration_structures)
                .build();
            let mut as_write_info = vk::WriteDescriptorSet::builder()
                .push_next(&mut as_set_info)
                .dst_set(sets[0])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_NV)
                .build();
            as_write_info.descriptor_count = 1;

            let image_set_info = [vk::DescriptorImageInfo::builder()
                .image_view(output_texture.view)
                .image_layout(vk::ImageLayout::GENERAL)
                .build()];
            let image_write_info = vk::WriteDescriptorSet::builder()
                .image_info(&image_set_info)
                .dst_set(sets[0])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .build();

            let uniform_set_info = [vk::DescriptorBufferInfo::builder()
                .buffer(uniform_buffer.buffer)
                .range(vk::WHOLE_SIZE)
                .build()];
            let uniform_write_info = vk::WriteDescriptorSet::builder()
                .buffer_info(&uniform_set_info)
                .dst_set(sets[0])
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .build();

            let vertex_set_info = [vk::DescriptorBufferInfo::builder()
                .buffer(vertex_buffer.buffer)
                .range(vk::WHOLE_SIZE)
                .build()];
            let vertex_write_info = vk::WriteDescriptorSet::builder()
                .buffer_info(&vertex_set_info)
                .dst_set(sets[0])
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .build();

            let index_set_info = [vk::DescriptorBufferInfo::builder()
                .buffer(index_buffer.buffer)
                .range(vk::WHOLE_SIZE)
                .build()];
            let index_write_info = vk::WriteDescriptorSet::builder()
                .buffer_info(&index_set_info)
                .dst_set(sets[0])
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .build();

            let write_descriptor_sets = [
                as_write_info,
                image_write_info,
                uniform_write_info,
                vertex_write_info,
                index_write_info,
            ];

            unsafe { device.update_descriptor_sets(&write_descriptor_sets, &[]) };

            sets
        };

        Descriptors::new(
            Arc::clone(context),
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
        )
    }

    unsafe fn any_as_u8_slice<T: Sized>(any: &T) -> &[u8] {
        let ptr = (any as *const T) as *const u8;
        std::slice::from_raw_parts(ptr, std::mem::size_of::<T>())
    }

    impl Drop for RTXData {
        fn drop(&mut self) {
            let device = self.context.device();
            unsafe {
                device.free_command_buffers(
                    self.context.general_command_pool(),
                    &self.command_buffers,
                );
                device.destroy_pipeline(self.pipeline, None);
                device.destroy_pipeline_layout(self.pipeline_layout, None);
            }
        }
    }

}
