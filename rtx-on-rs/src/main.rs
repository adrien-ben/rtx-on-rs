mod base;

use self::base::BaseApp;
use std::{error::Error, result::Result};

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    BaseApp::new().run();
    Ok(())
}
