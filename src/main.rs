#![feature(trait_alias)]

use std::{collections::HashMap, path::Path, time::Instant, f32::consts::PI};

use engine::EngineBuilder;
use mesh::Mesh;
use nalgebra::{Matrix4, Vector3, Vector4, clamp};
use rand::Rng;
use winit::event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};

mod camera;
mod engine;
mod mesh;
mod scene;
mod shader;
mod texture;

fn main() {
    let mut engine = EngineBuilder::new()
        .instance_with_required_extensions()
        .build();

    let mesh = Mesh::from_obj(
        include_str!("cube.obj"),
        &HashMap::from([(Path::new("cube.mtl"), include_str!("cube.mtl"))]),
    )
    .unwrap()
    .first()
    .unwrap()
    .clone();

    engine
        .scene
        .create_group(
            "basic",
            engine.device.clone(),
            crate::shader::simple::vertex::load(engine.device.clone()).unwrap(),
            crate::shader::simple::fragment::load(engine.device.clone()).unwrap(),
            engine.render_pass.clone(),
            engine.viewport.clone(),
        )
        .create_object("cube", mesh, engine.device.clone())
        .create_instance("0").update(|instance| {
            instance.position = Vector3::new(-1.0, 0.5, 0.5);
            instance.scale = 0.5;
        });
    engine
        .scene
        .get_camera()
        .update(|configuration| configuration.position.z -= 2.0);

    let mut ids = vec![];

    let mut rng = rand::thread_rng();

    let mut last_frame_time = Instant::now();

    let mut linear_velocity = Vector4::zeros();

    engine.run(move |event, scene| match event {
        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state,
                            virtual_keycode: Some(key),
                            ..
                        },
                    ..
                },
            ..
        } => match (state, key) {
            (ElementState::Pressed, VirtualKeyCode::W) => linear_velocity.z = 1.0,
            (ElementState::Released, VirtualKeyCode::W) => linear_velocity.z = 0.0,
            (ElementState::Pressed, VirtualKeyCode::S) => linear_velocity.z = -1.0,
            (ElementState::Released, VirtualKeyCode::S) => linear_velocity.z = 0.0,
            (ElementState::Pressed, VirtualKeyCode::A) => linear_velocity.x = -1.0,
            (ElementState::Released, VirtualKeyCode::A) => linear_velocity.x = 0.0,
            (ElementState::Pressed, VirtualKeyCode::D) => linear_velocity.x = 1.0,
            (ElementState::Released, VirtualKeyCode::D) => linear_velocity.x = 0.0,
            (ElementState::Pressed, VirtualKeyCode::Space) => linear_velocity.y = -1.0,
            (ElementState::Released, VirtualKeyCode::Space) => linear_velocity.y = 0.0,
            (ElementState::Pressed, VirtualKeyCode::LControl) => linear_velocity.y = 1.0,
            (ElementState::Released, VirtualKeyCode::LControl) => linear_velocity.y = 0.0,
            (ElementState::Pressed, VirtualKeyCode::E) => {
                let id = rng.gen::<u32>().to_string();

                scene
                    .get_group("basic")
                    .get_object("cube")
                    .create_instance(&id)
                    .update(|instance| {
                        instance.scale = 0.01;
                        instance.angle = rng.gen::<[f32; 3]>().into();
                        instance.position = rng.gen::<[f32; 3]>().into();
                    });

                ids.push(id);
                println!("{:?}", ids.len());
            }
            _ => (),
        },
        Event::DeviceEvent {
            event: DeviceEvent::MouseMotion { delta: (x, y) },
            ..
        } => {
            scene.get_camera().update(|configuration| {
                configuration.angle += Vector3::new(-*y as f32 / 500.0, *x as f32 / 500.0, 0.0);
                configuration.angle.x = clamp(configuration.angle.x, -PI / 2.01, PI / 2.01);
            });
        }
        Event::MainEventsCleared => {
            let frame_time = Instant::now();
            let ticks = frame_time - last_frame_time;

            scene.get_camera().update(|configuration| {
                configuration.position +=
                    (Matrix4::new_rotation(Vector3::new(0.0, 0.0, configuration.angle.z))
                        * Matrix4::new_rotation(Vector3::new(0.0, configuration.angle.y, 0.0))
                        * Matrix4::new_rotation(Vector3::new(configuration.angle.x, 0.0, 0.0))
                        * linear_velocity
                        * ticks.as_secs_f32())
                    .xyz()
            });

            let object = scene.get_group("basic").get_object("cube");

            for id in &ids {
                object.get_instance(id).update(|instance| {
                    instance.angle +=
                        Vector3::new(1.0 * ticks.as_secs_f32(), 1.0 * ticks.as_secs_f32(), 0.0)
                });
            }

            last_frame_time = frame_time;
        }
        _ => (),
    });
}
