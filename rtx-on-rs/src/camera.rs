use cgmath::Matrix4;

pub struct Camera {
    view: Matrix4<f32>,
    projection: Matrix4<f32>,
}

impl Camera {
    pub fn new(view: Matrix4<f32>, projection: Matrix4<f32>) -> Self {
        Self { view, projection }
    }
}

impl Camera {
    pub fn view(&self) -> Matrix4<f32> {
        self.view
    }

    pub fn projection(&self) -> Matrix4<f32> {
        self.projection
    }
}
