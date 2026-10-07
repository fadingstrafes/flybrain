"""ModernGL arena scene, articulated procedural fly, and readable on-screen controls."""

import math
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.fly_brain_opengl import look_at, perspective, write_mat4
from src.arena_model import leg_hip, solve_knee
from src.viewer_camera import project_points


def sphere_vertices():
    points = []
    for y in range(10):
        for x in range(16):
            corners = []
            for u, v in [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]:
                a, b = u * 2 * math.pi / 16, v * math.pi / 10
                corners.append([math.sin(b) * math.cos(a), math.sin(b) * math.sin(a), math.cos(b)])
            points.extend([corners[i] for i in (0, 1, 2, 0, 2, 3)])
    return np.asarray(points, dtype="f4")


class ArenaRenderer:
    def __init__(self, ctx, arena):
        self.ctx, self.arena = ctx, arena
        self.program = ctx.program(vertex_shader="""
            #version 330
            in vec3 in_pos;
            in vec3 axis_x; in vec3 axis_y; in vec3 axis_z;
            in vec3 center; in vec4 color;
            uniform mat4 mvp;
            out vec3 normal; out vec4 shade;
            void main() {
                vec3 world = center + mat3(axis_x, axis_y, axis_z) * in_pos;
                normal = normalize(axis_x * in_pos.x / dot(axis_x,axis_x)
                                 + axis_y * in_pos.y / dot(axis_y,axis_y)
                                 + axis_z * in_pos.z / dot(axis_z,axis_z));
                shade = color;
                gl_Position = mvp * vec4(world, 1.0);
            }
        """, fragment_shader="""
            #version 330
            in vec3 normal; in vec4 shade; out vec4 fragColor;
            void main() {
                float light = 0.30 + 0.70 * max(0.0, dot(normalize(normal), normalize(vec3(-0.4,-0.5,1.0))));
                fragColor = vec4(shade.rgb * light, shade.a);
            }
        """)
        self.sphere = ctx.buffer(sphere_vertices().tobytes())
        self.instances = ctx.buffer(reserve=256 * 16 * 4, dynamic=True)
        self.vao = ctx.vertex_array(self.program, [(self.sphere, "3f", "in_pos"),
                                   (self.instances, "3f 3f 3f 3f 4f /i", "axis_x", "axis_y", "axis_z", "center", "color")])
        self.line_program = ctx.program(vertex_shader="""
            #version 330
            in vec3 in_pos; uniform mat4 mvp;
            void main() { gl_Position = mvp * vec4(in_pos,1.0); }
        """, fragment_shader="""
            #version 330
            uniform vec4 color; out vec4 fragColor;
            void main() { fragColor = color; }
        """)
        w, h = arena.half_width, arena.half_height
        self.grids = []
        for spacing in (1, 5, 20, 100):
            grid = []
            for x in np.arange(math.ceil(-w/spacing)*spacing, w+.1, spacing):
                grid.extend([[x, -h, 0], [x, h, 0]])
            for y in np.arange(math.ceil(-h/spacing)*spacing, h+.1, spacing):
                grid.extend([[-w, y, 0], [w, y, 0]])
            buffer = ctx.buffer(np.array(grid, dtype="f4").tobytes())
            self.grids.append((buffer, ctx.vertex_array(self.line_program, [(buffer, "3f", "in_pos")])))
        self.path = ctx.buffer(reserve=2400 * 3 * 4, dynamic=True)
        self.path_vao = ctx.vertex_array(self.line_program, [(self.path, "3f", "in_pos")])
        walls = []
        for a,b in [((-w,-h),(w,-h)),((w,-h),(w,h)),((w,h),(-w,h)),((-w,h),(-w,-h))]:
            corners = [[*a,0],[*b,0],[*b,6],[*a,6]]
            walls.extend(corners[i] for i in (0,1,2,0,2,3))
        self.walls = ctx.buffer(np.asarray(walls,dtype="f4").tobytes())
        self.wall_vao = ctx.vertex_array(self.line_program,[(self.walls,"3f","in_pos")])

    def draw(self, pose, viewport, camera):
        self.ctx.viewport = viewport
        target = np.array([*pose["position"], pose["altitude"] + 0.25]) if camera["follow"] else np.asarray(camera.get("target", np.zeros(3)))
        yaw, pitch, distance = camera["yaw"], camera["pitch"], camera["distance"]
        eye = target + distance * np.array([math.cos(pitch) * math.sin(yaw),
                                           math.cos(pitch) * math.cos(yaw), math.sin(pitch)])
        mvp = perspective(45, viewport[2] / viewport[3], .05,
                          max(100,distance+4*max(self.arena.half_width,self.arena.half_height))) @ look_at(eye, target, [0, 0, 1])
        self.mvp = mvp
        write_mat4(self.program["mvp"], mvp)
        write_mat4(self.line_program["mvp"], mvp)
        previous_depth_mask = self.ctx.fbo.depth_mask
        self.ctx.fbo.depth_mask = False
        self.line_program["color"].value = (.23,.34,.40,.16)
        self.wall_vao.render(moderngl.TRIANGLES)
        self.ctx.fbo.depth_mask = previous_depth_mask
        self.line_program["color"].value = (0.11, 0.23, 0.25, 1)
        level = 0 if distance < 35 else 1 if distance < 160 else 2 if distance < 800 else 3
        self.grids[level][1].render(moderngl.LINES)
        path = pose["path"]
        if len(path) > 1:
            points = np.column_stack([path, np.full(len(path), 0.015)]).astype("f4")
            self.path.write(points.tobytes())
            self.line_program["color"].value = (0.20, 0.70, 0.60, 0.65)
            self.path_vao.render(moderngl.LINE_STRIP, vertices=len(points))
        objects = []
        def ellipsoid(center, scales, color, rotation=None):
            axes = np.eye(3) if rotation is None else rotation
            axes = axes @ np.diag(scales)
            objects.append([*axes[:, 0], *axes[:, 1], *axes[:, 2], *center, *color])
        w, h = self.arena.half_width, self.arena.half_height
        for x in (-w, w):
            ellipsoid([x, 0, 0.12], [.06, h, .12], [.25, .43, .45, 1])
        for y in (-h, h):
            ellipsoid([0, y, 0.12], [w, .06, .12], [.25, .43, .45, 1])
        post_spacing = max(4, max(w,h)/30)
        for x in np.arange(-w,w+.01,post_spacing):
            for y in (-h,h):
                ellipsoid([x,y,3],[.035,.035,3],[.27,.41,.48,1])
        for y in np.arange(-h+post_spacing,h,post_spacing):
            for x in (-w,w):
                ellipsoid([x,y,3],[.035,.035,3],[.27,.41,.48,1])
        for x, y, radius in self.arena.obstacles:
            ellipsoid([x, y, .32], [radius, radius, .55], [.27, .39, .44, 1])
        c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        origin = np.array([*pose["position"], pose["altitude"]])
        def local(center, scale, color):
            ellipsoid(origin + R @ np.asarray(center), scale, color, R)
        local([0, -.32, .48], [.26, .49, .23], [.42, .29, .13, 1])
        for y in (-.13, -.29, -.45, -.59):
            local([0, y, .485], [.264 - abs(y + .3) * .14, .025, .235 - abs(y + .3) * .10], [.18, .13, .085, 1])
        local([0, .20, .52], [.30, .33, .28], [.43, .31, .15, 1])
        local([0, .61, .54], [.28, .22, .22], [.25, .19, .13, 1])
        for side in (-1, 1):
            local([side * .22, .66, .58], [.13, .15, .15], [.80, .09, .045, 1])
            local([side * .11, .86, .56], [.025, .10, .027], [.35, .25, .14, 1])
        for leg in range(6):
            side = -1 if leg < 3 else 1
            amount = float(pose["activation"][leg])
            anchor = origin + R @ leg_hip(leg)
            foot = pose["feet"][leg]
            bend = [side * .3, -.2, -1.0] if pose["altitude"] > .08 else [side, .1, .5]
            knee = solve_knee(anchor, foot, R @ bend)
            toe = foot + R @ np.array([0, .09, -.008])
            for a, b in [(anchor, knee), (knee, foot), (foot, toe)]:
                middle = (a + b) / 2
                direction = b - a
                length = np.linalg.norm(direction)
                z = direction / length
                x = np.cross(z, [0, 0, 1])
                x /= np.linalg.norm(x)
                y = np.cross(z, x)
                ellipsoid(middle, [.028, .028, length / 2], [.30 + .45 * amount, .23 + .20 * amount, .12, 1],
                          np.column_stack([x, y, z]))
            ellipsoid(knee, [.04, .04, .04], [.25, .55 + .35 * amount, .45, 1])
        flying = pose["flight_requested"] or pose["altitude"] > .08
        for side in (-1, 1):
            if flying:
                power = float(pose["wing_activation"][0 if side < 0 else 1])
                # Slow visible wing strokes are illustrative, not real wingbeat frequency.
                angle = side * (.2 + .85 * power * math.sin(pose["elapsed"] * 2 * math.pi * 12))
                ca, sa = math.cos(angle), math.sin(angle)
                flap = np.array([[ca,0,sa], [0,1,0], [-sa,0,ca]])
                center = origin + R @ (np.array([side*.18, .15, .73]) + flap @ [side*.63, -.05, 0])
                ellipsoid(center, [.70, .22, .012], [.68, .78, .76, 1], R @ flap)
            else:
                local([side * .27, -.18, .79], [.22, .63, .012], [.68, .78, .76, 1])
        data = np.asarray(objects, dtype="f4")
        if data.nbytes > self.instances.size:
            self.instances.orphan(data.nbytes*2)
        self.instances.write(data.tobytes())
        self.vao.render(moderngl.TRIANGLES, instances=len(data))

    def release(self):
        for buffer, vao in self.grids:
            vao.release()
            buffer.release()
        for resource in (self.vao, self.sphere, self.instances, self.program,
                         self.path_vao, self.path, self.wall_vao,self.walls,self.line_program):
            resource.release()


class Overlay:
    width = 350

    def __init__(self, ctx):
        self.ctx = ctx
        self.program = ctx.program(vertex_shader="""
            #version 330
            in vec2 in_pos; out vec2 uv;
            void main() { gl_Position=vec4(in_pos,0,1); uv=vec2(in_pos.x*.5+.5,.5-in_pos.y*.5); }
        """, fragment_shader="""
            #version 330
            in vec2 uv; uniform sampler2D image; out vec4 fragColor;
            void main() { fragColor=texture(image,uv); }
        """)
        self.vbo = ctx.buffer(np.array([[-1,-1], [1,-1], [-1,1], [1,1]], dtype="f4").tobytes())
        self.vao = ctx.vertex_array(self.program, [(self.vbo, "2f", "in_pos")])
        self.texture = None
        self.height = 0
        self.fonts = {}

    def font(self, size):
        if size not in self.fonts:
            path = Path("/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf")
            self.fonts[size] = ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)
        return self.fonts[size]

    def update(self, pose, height, fps, neurons, mode="arena", brain=None):
        # Keep logical coordinates fixed; scale the panel to the actual window.
        image = Image.new("RGBA", (350, 900), (11, 20, 29, 245))
        d = ImageDraw.Draw(image)
        def text(x, y, value, size=14, color="#91a6b4"):
            d.text((x, y), value, font=self.font(size), fill=color)
        text(24, 22, "FLYBRAIN / " + ("CNS" if mode == "brain" else "ARENA"), 22, "#e6efed")
        text(24, 55, "male-cns:v1.0  ·  live neural activity", 13)
        status = "PAUSED" if pose["paused"] else "EXPLORING" if pose["exploration"] else "SENSORY INPUT ONLY"
        if pose["halted"]: status = "ACTIVITY GUARD — PRESS R"
        if pose["error"]: status = "SIMULATION ERROR — SEE TERMINAL"
        text(24, 120, status, 11, "#67d3ad" if not pose["halted"] else "#f4b879")
        d.line((24, 119, 326, 119), fill="#243747")
        text(24, 138, f"{pose['elapsed']:6.1f} s", 26, "#e6efed")
        text(195, 145, f"{fps:.0f} FPS", 18, "#e6efed")
        text(24, 179, f"Distance {pose['distance']:.2f}  ·  Speed {pose['speed']:.2f} units/s")
        text(24, 204, f"Step {pose['step']:,}  ·  {pose['active']:,} neurons firing")
        text(24, 229, f"{pose['motor_spikes']:,} leg motor spikes  ·  {pose['brain_ms']:.1f} ms/step")
        text(24, 251, f"{'AIRBORNE' if pose['altitude'] > .08 else 'GROUNDED'}  ·  Altitude {pose['altitude']:.2f}  ·  Wing output {np.mean(pose['wing_activation']):.2f}", 12, "#d5ba87")
        text(24, 272, "LEG MOTOR OUTPUT", 13, "#d5e4e4")
        names = ["L front", "L middle", "L hind", "R front", "R middle", "R hind"]
        for i, name in enumerate(names):
            y = 302 + i * 28
            text(24, y, name, 13)
            d.rounded_rectangle((100, y + 2, 300, y + 13), radius=4, fill="#20333e")
            level = float(pose["activation"][i])
            if level > .01:
                d.rounded_rectangle((100, y + 2, 100 + 200 * level, y + 13), radius=3, fill="#6dccad")
            if pose["touch"][i] > .1:
                d.ellipse((310, y + 1, 322, y + 13), fill="#f1a75b")
        text(24, 477, "LEFT EYE                       RIGHT EYE", 12, "#d5e4e4")
        vision = pose.get("vision")
        if vision is not None:
            for eye,x in enumerate((24,181)):
                eye_image = Image.fromarray(vision.images[eye]).resize((145,58),Image.Resampling.NEAREST)
                image.paste(eye_image,(x,500))
            left,front,right = vision.clearances
            text(24,566,f"Clearance  L {left:.1f}  /  Front {front:.1f}  /  R {right:.1f}",12)
            assist = "TURNING / BRAKING" if pose["avoidance_active"] else "ON" if pose["avoidance_enabled"] else "OFF"
            text(24,589,f"Vision assist: {assist}   ·   V toggle",12,"#e4b27a")
        else:
            text(24,515,"Vision disabled",14)
        # Transparent cutout for the separate real-morphology viewport.
        d.rectangle((8, 615, 342, 715), fill=(0, 0, 0, 0))
        text(24, 730, "Space pause   R reset   E exploration", 13, "#c0d3d8")
        text(24, 755, "1 / 2 touch left / right   C follow camera", 13)
        text(24, 780, "F takeoff / land   G auto flight   Esc save", 13)
        learning = pose.get("learning")
        if learning is not None:
            text(24, 813, f"{'LEARNING' if learning['training'] else 'EVALUATING'}  /  {learning['action']}", 13, "#e4b27a")
            text(24, 838, f"Reward {learning['reward']:+.1f}  ·  {learning['tiles']} tiles  ·  {learning['updates']} updates", 12)
            text(24, 861, "Learned sensory policy; fixed connectome weights", 11)
        else:
            text(24, 824, "Exploration drive + kinematics: toy model", 12, "#e4b27a")
            text(24, 847, "Fixed sensory drive; no learning controller", 11)
        if not pose["recording_complete"]:
            text(24, 872, "Event log capped; totals continue recording", 11, "#e4b27a")
        if mode == "brain" and brain is not None:
            d.rectangle((0,118,350,900),fill=(11,20,29,245))
            text(24,139,"WHOLE CNS",24,"#e6efed")
            text(24,179,f"{len(brain.indices):,} anatomically positioned cells",14)
            text(24,204,f"{neurons-len(brain.indices):,} retained cells lack coordinates",12)
            text(24,231,"Cell-group colors · yellow = recent spikes",12)
            text(24,258,"RECENTLY ACTIVE GROUPS",13,"#d5e4e4")
            active = np.flatnonzero(pose["brain_activity"] > .05)
            names,counts = np.unique(brain.classes[active],return_counts=True)
            order = np.argsort(counts)[::-1][:8]
            for i,index in enumerate(order):
                color = tuple(int(c*255) for c in brain.color(names[index]))
                d.ellipse((24,290+i*26,34,300+i*26),fill=color)
                text(43,284+i*26,str(names[index])[:24],12)
                text(282,284+i*26,str(counts[index]),13,"#e6efed")
            details = brain.details(pose["brain_counts"])
            text(24,514,"SELECTED NEURON",13,"#d5e4e4")
            if details:
                text(24,544,str(details["bodyId"]),22,"#e6efed")
                text(24,581,details["instance"][:35],14)
                text(24,609,details["class"][:35],13)
                text(24,637,f"{details['spikes']:,} spikes in this run",14,"#67d3ad")
            else:
                text(24,548,"Click a neuron to inspect it",14)
            text(24,688,"3 front  4 side  5 top  6 oblique",13)
            text(24,716,"B whole / head / cord   / search",13)
            text(24,744,"Z focus   C reset   H active only",13)
            text(24,772,"D draw   T color   [ / ] brush   X clear",12)
            text(24,798,"Shift erase   Ctrl+Z undo   L branches",12)
            text(24,824,"O media   K pause   J replay   Y display",12)
            text(24,851,"Cached branches + soma/root positions",11,"#e4b27a")
            text(24,872,"I neural input   - / + strength",11,"#e4b27a")
        for label,x,tab in [("ARENA",18,"arena"),("WHOLE CNS",178,"brain")]:
            d.rounded_rectangle((x,82,x+154,111),radius=5,fill="#255445" if mode==tab else "#20333e")
            text(x+18,88,label,12,"#e6efed")
        image = image.resize((self.width, height), Image.Resampling.LANCZOS)
        if self.texture is None or self.height != height:
            if self.texture: self.texture.release()
            self.texture = self.ctx.texture((self.width, height), 4)
            self.height = height
        self.texture.write(image.tobytes())

    def draw(self, fbw, fbh):
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.viewport = (fbw - self.width, 0, self.width, fbh)
        self.texture.use(location=0)
        self.program["image"].value = 0
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def release(self):
        for resource in (self.vao, self.vbo, self.program, self.texture):
            if resource is not None: resource.release()


class SceneHUD(Overlay):
    """Viewer-only orientation aids. The map never feeds the neural controller."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.map_rect = None
        self.map_bounds = None

    def map_position(self, x, y):
        if self.map_rect is None:
            return None
        x0, y0, x1, y1 = self.map_rect
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return None
        low, high = self.map_bounds
        return low + np.array([(x-x0)/(x1-x0), 1-(y-y0)/(y1-y0)])*(high-low)

    def update_scene(self, pose, width, height, mode, camera, scene, brain=None, mvp=None):
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        d = ImageDraw.Draw(image)
        def text(x, y, value, size=13, color="#bbd0d7"):
            d.text((x, y), value, font=self.font(size), fill=color)
        def panel(box):
            d.rounded_rectangle(box, radius=9, fill=(11, 24, 33, 225), outline="#294550")
        compact = width < 800
        header_width = min(width-32, 340 if compact else 450)
        panel((16, 16, 16+header_width, 111))
        if mode == "arena":
            w, h = pose["arena_size"]
            text(30, 28, f"OPEN ARENA   /   {w:g} × {h:g}", 18, "#e4eeea")
            label = "FOLLOW FLY" if camera["follow"] else "FREE CAMERA"
            text(30, 57, f"{label}   ·   M local / full map", 12)
            text(30, 82, f"X {pose['position'][0]:+.1f}   Y {pose['position'][1]:+.1f}   Contacts {pose['contacts']}", 12)
        else:
            text(30, 28, f"{camera.get('preset', 'orbit').upper()}   /   {camera.get('region', 'all').upper()}", 18, "#e4eeea")
            text(30, 57, f"{scene.get('fiber_count',0)} cached neuron arbors · L {'hide' if camera.get('fibers',True) else 'show'} branches", 12)
            text(30, 82, f"DRAW · brush {scene.get('brush_radius',14)} · T color" if scene.get("drawing") else
                 "Drag orbit · D draw · Right-drag pan", 12)
            if scene.get("fly_preview") and width >= 850:
                text(width-292,25,"LIVE MOTOR RESPONSE · U hide",12,"#c5e8de")
                text(width-292,238,f"Speed {pose['speed']:.2f} · {pose['motor_spikes']:,} leg motor spikes",11)
        self.map_rect = self.map_bounds = None
        if mode == "arena":
            size = min(230, width-52, height-250)
            x0, y0 = width-size-26, 144 if compact else 52
            panel((x0-10, y0-32, x0+size+10, y0+size+43))
            local = scene.get("local_map", False)
            text(x0, y0-24, "LOCAL MAP · 40 units" if local else "ARENA MAP", 12, "#e4eeea")
            half = np.array(pose["arena_size"])/2
            if local:
                low, high = pose["position"]-20, pose["position"]+20
            else:
                low, high = -half, half
            span = high-low
            map_w, map_h = size*span/max(span)
            x0 += (size-map_w)/2
            y0 += (size-map_h)/2
            rect = (x0, y0, x0+map_w, y0+map_h)
            self.map_rect, self.map_bounds = rect, (low, high)
            d.rectangle(rect, fill="#101f27", outline="#46616b")
            def xy(point):
                uv = (np.asarray(point)-low)/span
                return x0+uv[0]*map_w, y0+(1-uv[1])*map_h
            def inside(point):
                return bool((np.asarray(point) >= low).all() and (np.asarray(point) <= high).all())
            # Draw into a clipped layer so local paths/obstacles cannot spill
            # into the controls outside the map.
            layer = Image.new("RGBA", image.size)
            ld = ImageDraw.Draw(layer)
            for ox, oy, radius in pose["obstacles"]:
                px, py = xy([ox, oy])
                r = max(1.5, radius*map_w/span[0])
                ld.ellipse((px-r, py-r, px+r, py+r), fill="#6c8590")
            route = pose.get("map_path", pose["path"])
            if len(route) > 1:
                ld.line([xy(p) for p in route], fill="#50aa97", width=2)
            spawn = xy([0, -3.5])
            ld.ellipse((spawn[0]-3, spawn[1]-3, spawn[0]+3, spawn[1]+3), outline="#e8c882", width=1)
            px, py = xy(pose["position"])
            forward = np.array([-math.sin(pose["yaw"]), -math.cos(pose["yaw"])])
            right = np.array([-forward[1], forward[0]])
            center = np.array([px, py])
            ld.polygon([tuple(center+forward*8), tuple(center-forward*5+right*5),
                        tuple(center-forward*5-right*5)], fill="#f7e8a1")
            crop = tuple(map(int, (x0+1, y0+1, x0+map_w, y0+map_h)))
            image.alpha_composite(layer.crop(crop), (crop[0], crop[1]))
            text(width-size-26, (144 if compact else 52)+size+9, "Click map to inspect · C follow", 11)
            if mvp is not None and not camera["follow"]:
                screen, visible = project_points([[*pose["position"], pose["altitude"]+.5]], mvp, (0, 0, width, height))
                if visible[0]:
                    x, y = screen[0, 0], height-screen[0, 1]
                    d.ellipse((x-8, y-8, x+8, y+8), outline="#f7e8a1", width=2)
                    text(x+12, y-7, "FLY", 12, "#f7e8a1")
        elif brain is not None and mvp is not None and not camera.get("active_only"):
            groups = [("OPTIC LOBE", brain.head_mask & (brain.points[:, 0] < 0)),
                      ("OPTIC LOBE", brain.head_mask & (brain.points[:, 0] >= 0)),
                      ("VENTRAL NERVE CORD", brain.region_masks["cord"])]
            for label, mask in groups:
                mask = mask & brain.visible_mask(camera.get("region", "all"))
                if not mask.any():
                    continue
                point = np.median(brain.points[mask], axis=0)
                screen, visible = project_points([point], mvp, (0, 0, width, height))
                if visible[0]:
                    x, y = screen[0, 0], height-screen[0, 1]
                    if 130 < y < height-100:
                        text(max(20, min(width-190, x-55)), y, label, 11, "#c3dce1")
        bottom = height-(145 if mode == "brain" else 72)
        panel((16, bottom, width-16, height-16))
        if scene.get("searching"):
            text(30, bottom+8, "Find ID / name: " + scene.get("query", "") + "▏", 15, "#f7e8a1")
            text(30, bottom+31, "Enter find / next match   ·   Esc close search", 11)
        else:
            text(30, bottom+8, scene.get("notice", "") or ("C follow / overview   3 chase   4 overhead" if mode == "arena" else "/ find neuron   B whole / head / cord   3 / 4 / 5 / 6 views"), 12)
            text(30, bottom+31, "Space pause   N single step   P screenshot   S save checkpoint", 11)
            if mode == "brain":
                text(30, bottom+54, scene.get("media_status", "")[:max(25, (width-60)//7)], 12, "#f7e8a1")
                text(30, bottom+78, f"{scene.get('media_mode','neurons').upper()} · O open · K pause · J replay · Y display · Backspace stop", 11)
                stimulus = pose.get("art_input",{})
                label = "HALTED · R reset" if pose.get("halted") else "SIM PAUSED" if pose.get("paused") else "INPUT ON" if stimulus.get("enabled") else "INPUT OFF"
                text(30, bottom+103, f"{label} · {stimulus.get('targets',0):,}/{stimulus.get('mapped',0):,} cells · strength {stimulus.get('strength',0):.1f} · yellow = spikes", 11,"#ffc876")
        if self.texture is None or (self.width, self.height) != (width, height):
            if self.texture:
                self.texture.release()
            self.texture = self.ctx.texture((width, height), 4)
            self.width, self.height = width, height
        self.texture.write(image.tobytes())

    def draw(self, fbw, fbh):
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.texture.use(location=0)
        self.program["image"].value = 0
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST)
