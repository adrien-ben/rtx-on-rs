# RTX-On with Rust

Vulkan and RTX samples written in Rust using [Ash][2].

## Samples

- [x] Triangle
- [x] Shadows
- [ ] Reflections
- [ ] Refractions
- [ ] Global illumination
- [ ] More...

## Run it

```sh
# Compile shaders
./compile_shaders.sh

# Run app
RUST_LOG=rtx_on_rs=debug,vulkan=warn cargo run --bin <sample_name>
```

## Tested on

Windows 10 on a GTX1070.

## Credits

[Sascha Willem's Vulkan repository][0]

[NVidia's tutorial][1]

[0]: https://github.com/SaschaWillems/Vulkan
[1]: https://developer.nvidia.com/rtx/raytracing/vkray
[2]: https://github.com/MaikKlein/ash
