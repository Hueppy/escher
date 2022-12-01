pub mod simple {
    pub mod vertex {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
#version 450

layout(location = 0) in vec3 coord;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_coord;

layout(location = 4) in mat4 model;

layout(set = 0, binding = 0) uniform Camera {
    mat4 view;
    mat4 projection;
} camera;

layout(location = 0) out vec3 fragColor;

void main() {
    mat4 viewModel = camera.view * model;
    gl_Position = camera.projection * viewModel * vec4(coord, 1.0);
    fragColor = abs(/*viewModel */ vec4(normal, 1.0)).xyz;
}
"
        }
    }

    pub mod fragment {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(fragColor, 1.0);
}
"
        }
    }
}
