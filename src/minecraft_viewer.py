"""Companion OpenGL viewer: same anatomical BrainMap as the arena."""
from pathlib import Path
import numpy as np
import time


def run_viewer(session, graph):
    import glfw
    import moderngl
    from src.brain_view import BrainMap, BrainRenderer
    from src.minecraft_panel import MinecraftPanel
    from src.minecraft_eye_panel import EyePanel
    if not glfw.init():
        raise RuntimeError('GLFW could not initialize')
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(1400, 960, 'FlyBrain — Minecraft', None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError('Could not create brain viewer window')
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    ctx = moderngl.create_context(require=330)
    brain = BrainMap(graph.ids)
    fibers = brain.load_fibers(sorted(Path('runs').glob('*/skeleton_cache')))
    renderer = BrainRenderer(ctx, brain, fibers)
    panel = MinecraftPanel(ctx)
    eyes = EyePanel(ctx)
    eye_height = lambda height: min(250,max(1,height//3))
    camera = brain.camera_preset('front', (1400-panel.width)/(960-eye_height(960)))
    camera['active_only'] = True
    current = [None]
    panel_updated = 0.
    drag = [None]
    def key(window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        if key == glfw.KEY_H:
            camera['active_only'] = not camera.get('active_only', False)
        if key == glfw.KEY_L:
            camera['fibers'] = not camera.get('fibers', True)
        if key == glfw.KEY_C:
            w,h = glfw.get_framebuffer_size(window)
            camera.update(brain.camera_preset('front', max(w-panel.width,1)/max(h-eye_height(h),1)))
    def button(window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_RIGHT:
            drag[0] = glfw.get_cursor_pos(window) if action == glfw.PRESS else None
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            x,y = glfw.get_cursor_pos(window)
            w,h = glfw.get_framebuffer_size(window)
            ww,wh = glfw.get_window_size(window)
            px=x*w/max(ww,1)
            if px < w-panel.width and h-y*h/max(wh,1)>eye_height(h) and current[0] is not None:
                brain.selected = None
                brain.pick(px,h-y*h/max(wh,1),renderer.mvp,(0,eye_height(h),max(w-panel.width,1),max(h-eye_height(h),1)),
                           current[0]['brain_activity'] if camera['active_only'] else None)
    def cursor(window, x, y):
        if drag[0] is not None:
            px,py = drag[0]
            camera['yaw'] += (x-px)*.006
            camera['pitch'] = float(np.clip(camera['pitch']+(y-py)*.006,-1.5,1.5))
            drag[0] = (x,y)
    def scroll(window, x, y):
        camera['distance'] = float(np.clip(camera['distance']*np.exp(-y*.12),.2,30))
    glfw.set_key_callback(window,key)
    glfw.set_mouse_button_callback(window,button)
    glfw.set_cursor_pos_callback(window,cursor)
    glfw.set_scroll_callback(window,scroll)
    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            w,h = glfw.get_framebuffer_size(window)
            if w == 0 or h == 0:
                glfw.wait_events_timeout(.1)
                continue
            pose = session.snapshot()
            # Exact most-recent observation batch; no old activity trail.
            pose['brain_activity'] = (pose['brain_recent']>0).astype('f4')
            if brain.selected is not None and camera['active_only'] and not pose['brain_activity'][brain.selected]:
                brain.selected=None
            current[0] = pose
            ctx.clear(.015,.018,.028)
            ctx.enable(moderngl.BLEND)
            renderer.draw(pose,(0,eye_height(h),max(w-panel.width,1),max(h-eye_height(h),1)),camera)
            now=time.monotonic()
            if now-panel_updated>.1 or panel.height!=h:
                panel.update(pose,h,brain,camera['active_only'])
                eyes.update(pose,max(w-panel.width,1),eye_height(h))
                panel_updated=now
            panel.draw(w,h)
            eyes.draw()
            selected = '' if brain.selected is None else f' | neuron {graph.ids[brain.selected]} spikes {pose["brain_counts"][brain.selected]}'
            glfw.set_window_title(window, f'FlyBrain | {pose["action"]} | health {pose["health"]:.0f} food {pose["food"]:.0f} | deaths {pose.get("deaths",0)} updates {pose.get("updates",0)} | active {pose["active"]}{selected}')
            glfw.swap_buffers(window)
    finally:
        renderer.release()
        panel.release()
        eyes.release()
        ctx.release()
        glfw.destroy_window(window)
        glfw.terminate()
