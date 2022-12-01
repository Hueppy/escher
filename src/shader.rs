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


pub mod raymarch {
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

layout(location = 0) out vec2 uv;

void main() {
    mat4 viewModel = camera.view * model;
    gl_Position = camera.projection * viewModel * vec4(coord, 1.0);
    uv = coord.xy;
}
"
        }
    }

    pub mod fragment {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
#version 450

#define MAX_STEPS 128
#define MAX_DIST 500.0
#define EPSILON 0.01

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 f_color;

vec2 sphere(vec3 p, float r, float id) {
    return vec2(length(p) - r, id);
}

vec2 plane(vec3 p, vec3 n, float d, float id) {
    return vec2(dot(n, p) + d, id);
}

vec2 union_object(vec2 a, vec2 b) {
    if (a.x < b.x) {
        return a;
    }
    return b;
}

vec2 map(vec3 p) {
    return union_object(
        sphere(p, 0.5, 1.0),
        plane(p, vec3(0.0, 0.0, 1.0), 0.0, 0.0)
    );
}

vec2 march_ray(vec3 ro, vec3 rd) {
    vec2 hit = vec2(0.0);
    vec2 object = vec2(0,0);

    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + object.x * rd;
        hit = map(p);
        object.x += hit.x;
        object.y = hit.y;
        if (abs(hit.x) < EPSILON || object.x > MAX_DIST) {
            break;
        }
    }

    return object;
}

vec4 render(vec2 uv) {
    vec2 object = march_ray(vec3(uv, 0.0), vec3(0.0));
    return vec4(vec3(object.y), 1.0);
}

void main() {
    f_color = render(uv);
}
"
        }
    }
}
