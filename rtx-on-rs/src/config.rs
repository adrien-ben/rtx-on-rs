#[derive(Copy, Clone, Debug)]
pub struct Config {
    pub resolution: [u32; 2],
    pub msaa: u32,
    pub vsync: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            resolution: [800, 600],
            msaa: 1,
            vsync: true,
        }
    }
}
