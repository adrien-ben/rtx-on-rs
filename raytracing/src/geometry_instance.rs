use ash::vk;

#[derive(Copy, Clone)]
pub struct GeometryInstance {
    pub transform: [f32; 12],
    pub instance_custom_index: u32,
    pub mask: u32,
    pub instance_offset: u32,
    pub flags: vk::GeometryInstanceFlagsNV,
    pub acceleration_structure_handle: u64,
}

impl GeometryInstance {
    pub fn get_data(&self) -> GeometryInstanceData {
        let instance_custom_indexand_mask =
            (self.mask << 24) | (self.instance_custom_index & 0x00_ff_ff_ff);
        let instance_offset_and_flags =
            (self.flags.as_raw() << 24) | (self.instance_offset & 0x00_ff_ff_ff);
        GeometryInstanceData {
            transform: self.transform,
            instance_custom_indexand_mask,
            instance_offset_and_flags,
            acceleration_structure_handle: self.acceleration_structure_handle,
        }
    }
}

#[derive(Copy, Clone)]
#[allow(dead_code)]
#[repr(C)]
pub struct GeometryInstanceData {
    transform: [f32; 12],
    instance_custom_indexand_mask: u32,
    instance_offset_and_flags: u32,
    acceleration_structure_handle: u64,
}
