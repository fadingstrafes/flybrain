"""Interactive whole-CNS soma map using anatomical positions from local metadata."""

from pathlib import Path

import moderngl
import numpy as np
import pandas as pd

from src.fly_brain_opengl import look_at, perspective, write_mat4
from src.viewer_camera import fit_camera, project_points


class BrainMap:
    def __init__(self, ids, cache=Path("data/cache")):
        metadata = pd.read_parquet(Path(cache)/"metadata.parquet",
            columns=["bodyId", "instance", "type", "superclass", "somaLocation", "tosomaLocation"])
        metadata = metadata.drop_duplicates("bodyId").set_index("bodyId").reindex(ids)
        points, indices = [], []
        for index, row in enumerate(metadata[["somaLocation", "tosomaLocation"]].itertuples(index=False)):
            for value in row:
                if isinstance(value, (list, tuple, np.ndarray)) and np.asarray(value).shape == (3,) and np.isfinite(value).all():
                    points.append(value)
                    indices.append(index)
                    break
        if not points:
            raise RuntimeError("No anatomical neuron positions are available in the local metadata")
        points = np.asarray(points, dtype=np.float32)
        # A rigid 180-degree rotation about specimen X places the optic lobes
        # above the VNC. No neuron coordinates are invented or reshaped.
        self.rotation = np.array([1, -1, -1], dtype=np.float32)
        points *= self.rotation
        low, high = np.percentile(points, [1,99], axis=0)
        self.center = (low+high)*.5
        self.scale = 3.0/max(float(np.max(high-low)), 1.0)
        self.points = ((points-self.center) * self.scale).astype("f4")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.ids = ids
        self.metadata = metadata
        self.classes = metadata.superclass.fillna("unclassified").to_numpy()
        self.colors = np.array([self.color(name) for name in self.classes[self.indices]], dtype="f4")
        self.selected = None
        self.id_lookup = {int(body): int(index) for index, body in enumerate(ids)}
        self.point_lookup = {int(index): i for i, index in enumerate(self.indices)}
        plotted_classes = self.classes[self.indices].astype(str)
        self.head_mask = np.char.startswith(plotted_classes, "ol_") | np.char.startswith(plotted_classes, "cb_")
        self.region_masks = {"all": np.ones(len(self.points), dtype=bool),
                             "head": self.head_mask,
                             "cord": np.char.startswith(plotted_classes, "vnc_")}
        self.search_names = (metadata.instance.fillna("")+" "+metadata.type.fillna("")).str.lower().to_numpy()

    def transform(self, coordinates):
        """Apply exactly the soma map's rigid transform to raw skeleton voxels."""
        return ((np.asarray(coordinates, dtype="f4")*self.rotation-self.center)*self.scale).astype("f4")

    def stimulus_from_image(self, image, mvp, viewport, region="all"):
        """Map visible soma/root positions to image intensity; return sparse input."""
        pixels = np.asarray(image, dtype=np.float32)/255
        screen, visible = project_points(self.points, mvp, viewport)
        visible &= self.visible_mask(region)
        u = np.clip((screen[:,0]-viewport[0])/viewport[2], 0, 1)
        v = np.clip(1-(screen[:,1]-viewport[1])/viewport[3], 0, 1)
        x = np.minimum((u*pixels.shape[1]).astype(int), pixels.shape[1]-1)
        y = np.minimum((v*pixels.shape[0]).astype(int), pixels.shape[0]-1)
        sampled = pixels[y,x]
        # Color-independent brightness also supports colored ink and spectrograms.
        intensity = sampled[:,:3].max(axis=1)*sampled[:,3]
        valid = visible & (intensity > .15)
        return self.indices[valid], intensity[valid]

    def load_fibers(self, directories, limit=80):
        candidates = {}
        for directory in directories:
            for path in sorted(Path(directory).glob("*.npz")):
                if path.stem.isdigit() and int(path.stem) in self.id_lookup:
                    candidates.setdefault(int(path.stem), path)
        # Long ascending/descending arbors make the head-to-cord pathways visible.
        order = sorted(candidates, key=lambda body: (
            self.classes[self.id_lookup[body]] not in ("descending_neuron", "ascending_neuron"), body))
        fibers = []
        for body in order[:limit]:
            with np.load(candidates[body], allow_pickle=False) as data:
                segments = data["segments"]
            if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all():
                raise ValueError(f"Invalid skeleton segments in {candidates[body]}")
            if len(segments):
                fibers.append((self.id_lookup[body], self.transform(segments.reshape(-1, 3))))
        return fibers

    def camera_preset(self, name="front", aspect=1.4, region="all"):
        import math
        yaw, pitch = {"front": (0, 0), "side": (math.pi/2, 0),
                      "top": (0, 1.45), "oblique": (-.55, .3)}[name]
        points = self.points[self.region_masks[region]]
        if not len(points):
            points = self.points
        return dict(fit_camera(points, yaw, pitch, aspect, padding=1.4), preset=name, region=region)

    def visible_mask(self, region="all"):
        mask = self.region_masks.get(region, self.region_masks["all"]).copy()
        if self.selected in self.point_lookup:
            mask[self.point_lookup[self.selected]] = True
        return mask

    def search(self, query):
        """Exact body ID first, then a bounded list of literal annotation matches."""
        query = query.strip().lower()
        if not query:
            return []
        if query.isdigit():
            index = self.id_lookup.get(int(query))
            return [] if index is None else [index]
        return [i for i, name in enumerate(self.search_names) if query in name][:50]

    @staticmethod
    def color(name):
        if str(name).startswith("ol_"): return (.27,.59,.88)
        if str(name).startswith("cb_"): return (.67,.45,.82)
        if str(name).startswith("vnc_"): return (.27,.76,.57)
        if "descending" in str(name): return (.95,.65,.25)
        if "ascending" in str(name): return (.84,.42,.40)
        if "visual" in str(name): return (.35,.78,.87)
        return (.55,.61,.65)

    def pick(self, x, y, mvp, viewport, activity=None, region="all"):
        screen, visible = project_points(self.points, mvp, viewport)
        distance = np.sum((screen-[x,y])**2,axis=1)
        distance[~visible | ~self.visible_mask(region)] = np.inf
        if activity is not None:
            hidden = activity[self.indices] <= .05
            if self.selected is not None:
                hidden &= self.indices != self.selected
            distance[hidden] = np.inf
        index = int(np.argmin(distance))
        self.selected = int(self.indices[index]) if distance[index] < 18**2 else None
        return self.selected

    def details(self, counts=None):
        if self.selected is None: return None
        row = self.metadata.iloc[self.selected]
        return {"bodyId": int(self.ids[self.selected]),
                "instance": str(row["instance"]), "type": str(row["type"]),
                "class": str(row["superclass"]),
                "spikes": int(counts[self.selected]) if counts is not None else 0}


class BrainRenderer:
    def __init__(self, ctx, brain, fibers=()):
        self.ctx, self.brain = ctx, brain
        self.program = ctx.program(vertex_shader="""
            #version 330
            in vec3 position; in vec3 color; in float neuron; in float region;
            uniform mat4 mvp; uniform sampler2D activity;
            uniform float point_size; uniform int selected; uniform int region_filter;
            uniform bool is_fiber; uniform sampler2D artwork; uniform bool art_enabled;
            uniform vec2 art_scale;
            out vec4 shade; out float neural_active;
            void main() {
                int index=int(neuron);
                float level=texelFetch(activity,ivec2(index%512,index/512),0).r;
                bool chosen=index==selected;
                neural_active=(level>.05 || chosen) ? 1.0 : 0.0;
                shade=level>.05 ? vec4(mix(color,vec3(1.0,.87,.32),.8),.95) : vec4(color,.22);
                if(chosen) shade=vec4(1,1,1,1);
                gl_PointSize=chosen ? 10.0 : point_size+(level>.05 ? (art_enabled ? .4 : 2.3):0.0);
                gl_Position=mvp*vec4(position,1);
                if(is_fiber) shade.a *= .55;
                if(art_enabled) {
                    vec2 uv=gl_Position.xy/gl_Position.w*.5+.5;
                    uv.y=1.0-uv.y;
                    uv=(uv-.5)*art_scale+.5;
                    if(all(greaterThanEqual(uv,vec2(0))) && all(lessThanEqual(uv,vec2(1)))) {
                        vec4 art=texture(artwork,uv);
                        shade=mix(shade,vec4(art.rgb, is_fiber ? .65 : .95),art.a);
                    }
                }
                if(art_enabled && level>.05) shade=vec4(mix(shade.rgb,vec3(1,.72,.15),.8),is_fiber ? .6 : 1.0);
                if(chosen) shade=vec4(1,1,1,1);
                if(region_filter!=0 && int(region)!=region_filter && !chosen)
                    gl_Position=vec4(2,2,2,1);
            }
        """, fragment_shader="""
            #version 330
            in vec4 shade; in float neural_active; uniform bool active_only; uniform bool is_fiber;
            uniform bool art_enabled; out vec4 fragColor;
            void main() {
                if(active_only && neural_active<.5) discard;
                if(!is_fiber && length(gl_PointCoord-vec2(.5))>.5) discard;
                fragColor=shade;
            }
        """)
        regions = np.zeros(len(brain.points))
        regions[brain.region_masks["head"]] = 1
        regions[brain.region_masks["cord"]] = 2
        vertices = np.column_stack([brain.points, brain.colors, brain.indices, regions]).astype("f4")
        self.vbo = ctx.buffer(vertices.tobytes())
        self.vao = ctx.vertex_array(self.program,[(self.vbo,"3f 3f 1f 1f","position","color","neuron","region")])
        self.fiber_count = len(fibers)
        self.fiber_vbo = self.fiber_vao = None
        if fibers:
            batches = []
            for index, points in fibers:
                name = str(brain.classes[index])
                region = 1 if name.startswith(("ol_", "cb_")) else 2 if name.startswith("vnc_") else 0
                batches.append(np.column_stack([points, np.tile(brain.color(name), (len(points), 1)),
                                                np.full(len(points), index), np.full(len(points), region)]).astype("f4"))
            self.fiber_vbo = ctx.buffer(np.concatenate(batches).tobytes())
            self.fiber_vao = ctx.vertex_array(self.program, [(self.fiber_vbo, "3f 3f 1f 1f", "position", "color", "neuron", "region")])
        self.levels = np.zeros((int(np.ceil(len(brain.ids)/512)),512),dtype="f4")
        self.texture = ctx.texture((512,len(self.levels)),1,dtype="f4")
        self.texture.filter = (moderngl.NEAREST,moderngl.NEAREST)
        self.program["activity"].value = 0
        self.program["artwork"].value = 1
        self.empty_art = ctx.texture((1, 1), 4, bytes(4))
        self.mvp = np.eye(4,dtype="f4")
        self.viewport = (0,0,1,1)

    def draw(self, pose, viewport, camera, artwork=None):
        self.ctx.viewport = viewport
        import math
        yaw,pitch,dist = camera["yaw"],camera["pitch"],camera["distance"]
        target = np.asarray(camera.get("target",[0,0,0]))
        eye = target+dist*np.array([math.cos(pitch)*math.sin(yaw),math.cos(pitch)*math.cos(yaw),math.sin(pitch)])
        self.mvp = perspective(42,viewport[2]/viewport[3],.02,100) @ look_at(eye,target,[0,0,1])
        self.viewport = viewport
        write_mat4(self.program["mvp"],self.mvp)
        self.levels.flat[:len(self.brain.ids)] = pose["brain_activity"]
        self.texture.write(self.levels.tobytes())
        self.texture.use(location=0)
        (artwork if artwork is not None else self.empty_art).use(location=1)
        self.program["art_enabled"].value = artwork is not None
        self.program["art_scale"].value = (1., 1.)
        self.program["selected"].value = self.brain.selected if self.brain.selected is not None else -1
        self.program["point_size"].value = float(np.clip(6/dist,1,1.6 if artwork is not None else 5))
        self.program["active_only"].value = camera.get("active_only",False)
        self.program["region_filter"].value = {"all":0,"head":1,"cord":2}[camera.get("region","all")]
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.blend_func = (moderngl.SRC_ALPHA,moderngl.ONE_MINUS_SRC_ALPHA if artwork is not None else moderngl.ONE)
        if self.fiber_vao is not None and camera.get("fibers", True):
            self.program["is_fiber"].value = True
            self.fiber_vao.render(moderngl.LINES)
        self.program["is_fiber"].value = False
        self.vao.render(moderngl.POINTS)
        self.ctx.blend_func = (moderngl.SRC_ALPHA,moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def release(self):
        for resource in (self.fiber_vao, self.fiber_vbo, self.empty_art):
            if resource is not None: resource.release()
        for resource in (self.vao,self.vbo,self.texture,self.program): resource.release()
