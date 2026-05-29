import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import cv2 as cv
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config

from image_publisher.image_publisher import DepthImagePublisher


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

# ELASTIC BAND for humanoid robot
if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1":
        band_attached_link = mj_model.body("pelvis").id
    elif config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data)

if config.ENABLE_DEPTH_CAMERA:
    camera_name = config.CAMERA_SENSOR_NAME
    original_width = config.CAMERA_ORIGINAL_WIDTH
    original_height = config.CAMERA_ORIGINAL_HEIGHT
    downsampled_width = config.CAMERA_DOWNSAMPLED_WIDTH
    downsampled_height = config.CAMERA_DOWNSAMPLED_HEIGHT
    depth_image = np.zeros((downsampled_height, downsampled_width), dtype=np.float32)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)

def SimulationThread():
    global mj_data, mj_model, depth_image

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)
    depth_image_publisher = DepthImagePublisher(depth_image, config.DEPTH_PUBLISH_DT)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        mujoco.mj_step(mj_model, mj_data)
        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    global camera_name, original_width, original_height, depth_image
    # Create the Renderer and scene in this thread to avoid X BadAccess errors.
    renderer = mujoco.Renderer(mj_model, original_height, original_width)
    
    while viewer.is_running():
        step_start = time.perf_counter()
        locker.acquire()
        # sync viewer with simulation state
        viewer.sync()

        # If a renderer exists, render the depth image from the viewer thread
        if renderer is not None:
            try:
                renderer.update_scene(mj_data, camera=camera_name)
                renderer.enable_depth_rendering()
                raw_depth_image = renderer.render()
                renderer.disable_depth_rendering()
                
                # downsample the raw depth image
                depth_image = cv.resize(raw_depth_image, 
                                        dsize=(downsampled_width, downsampled_height),
                                        dst=depth_image,
                                        interpolation=cv.INTER_LINEAR)

                # normalize and convert to 8-bit pixels safely
                depth_image[:] = np.clip(depth_image, config.NEAR_CLIP, config.FAR_CLIP)[:]
                depth_image[:] = (depth_image[:] - config.NEAR_CLIP) / (config.FAR_CLIP - config.NEAR_CLIP)

                # Show the depth image from the viewer thread (safe for X/GL).
                try:
                    cv.imshow("Depth Image", (255 * depth_image).astype(np.uint8))
                    cv.waitKey(1)
                except Exception as e:
                    # If cv windowing fails, fall back to saving an image.
                    print("Depth display error:", e)
                    try:
                        cv.imwrite("/tmp/depth_image.png", depth_image.astype(np.uint8))
                    except Exception as ee:
                        print("Failed to write depth image:", ee)
            except Exception as e:
                print("Depth render error (viewer thread):", e)

        locker.release()
        time_until_next_step = config.VIEWER_DT - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# Depth rendering is handled from the viewer thread above to avoid cross-
# thread GL/X access which can trigger BadAccess errors.


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)
    # Depth rendering runs inside the viewer thread now (no separate thread).
    viewer_thread.start()
    sim_thread.start()
