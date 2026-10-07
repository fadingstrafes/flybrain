"""Live RGB eyes and the exact compound-eye samples used for neural stimulation."""
from PIL import Image, ImageDraw
import numpy as np
import moderngl
from src.arena_render import Overlay


class EyePanel(Overlay):
    def update(self,pose,width,height=250):
        image=Image.new('RGBA',(1020,250),'#101d28');draw=ImageDraw.Draw(image)
        headings=['LEFT EYE / rendered RGB','RIGHT EYE / rendered RGB','COMPOUND SAMPLES / L + R','IMAGE CHANGE / L + R']
        for column,title in enumerate(headings):
            draw.text((15+column*255,12),title,font=self.font(12),fill='#7ddcc0')
        rgb=pose.get('eye_rgb');facets=pose.get('eye_facets');motion=pose.get('eye_motion')
        if rgb is None:
            draw.text((20,75),'Waiting for fresh eye images from Minecraft',font=self.font(22),fill='#edbd79')
            draw.text((20,118),'No image is being substituted with depth or the observer camera.',font=self.font(14),fill='#b7cad8')
        else:
            for side in range(2):
                eye=Image.fromarray(rgb[side]).resize((235,157),Image.Resampling.NEAREST)
                image.paste(eye,(10+side*255,38))
            sampled=np.concatenate([facets[0],facets[1]],axis=1)
            image.paste(Image.fromarray((sampled*255).astype('uint8')).resize((235,88),Image.Resampling.NEAREST),(520,48))
            change=np.concatenate([motion[0],motion[1]],axis=1)
            image.paste(Image.fromarray((np.clip(change*3,0,1)*255).astype('uint8')).convert('RGB').resize((235,88),Image.Resampling.NEAREST),(775,48))
            draw.text((520,151),'768 samples / eye',font=self.font(14),fill='#b7cad8')
            draw.text((775,151),'Brightness difference x3',font=self.font(14),fill='#b7cad8')
        draw.text((15,209),f'Frame {pose.get("eye_frame", "--")}  |  Visual inputs: {pose.get("visual_input_firing",0):,} firing / {pose.get("visual_input_cells",0):,} mapped cells  |  {pose.get("visual_input_spikes",0):,} spikes',font=self.font(15),fill='#edf7fb')
        draw.text((15,231),'R1-R6 / R7 / R8 + synthetic L2/L3 relay input; facet map and RGB tuning are assumptions. Capture: up to 4 Hz.',font=self.font(12),fill='#a3b5c5')
        if self.texture is None or (self.width,self.height)!=(width,height):
            if self.texture:self.texture.release()
            self.texture=self.ctx.texture((width,height),4)
        self.width,self.height=width,height
        self.texture.write(image.resize((width,height),Image.Resampling.LANCZOS).tobytes())

    def draw(self):
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.viewport=(0,0,self.width,self.height)
        self.texture.use(location=0);self.program['image'].value=0
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST)
