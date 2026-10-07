"""Readable Minecraft observer dashboard rendered through the existing GL overlay."""
from PIL import Image, ImageDraw
import numpy as np
from src.arena_render import Overlay


class MinecraftPanel(Overlay):
    width = 380

    def update(self, pose, height, brain, active_only=True):
        image = Image.new('RGBA', (380, 960), '#0c1620')
        draw = ImageDraw.Draw(image)
        def text(y, value, size=15, color='#b7cad8', x=20):
            draw.text((x,y), str(value), font=self.font(size), fill=color)
        def bar(y, title, value, color):
            value = float(np.clip(value,0,1))
            text(y, f'{title}  {value*100:.0f}%', 14)
            draw.rounded_rectangle((20,y+23,360,y+30),3,fill='#223342')
            if value > 0:
                draw.rectangle((20,y+23,20+340*value,y+30),fill=color)
        text(18,'FLYBRAIN / LIVE OBSERVER',20,'#ecf5fa')
        age = pose.get('observation_age')
        status = 'WAITING FOR MINECRAFT' if age is None else (
            f'NO NEW INPUT / {age:.0f}s' if age>5 else 'RECEIVING GAME OBSERVATIONS')
        text(51,status,13,'#e6b969' if age is None or age>5 else '#79d6b6')
        text(77,'Latest 6 neural steps' if active_only else 'All mapped neurons',16,'#ecf5fa')
        recent = pose['brain_recent']
        active = recent>0
        text(102,f'{np.count_nonzero(active):,} firing / {len(recent):,} retained',14)
        text(124,f'{np.count_nonzero(active[brain.indices]):,} have mapped positions',13)
        text(155,'BODY STATE / estimated, not emotion',15,'#79d6b6')
        text(180,pose.get('body_state','Waiting for body signals'),18,'#ecf5fa')
        text(209,f'Health {pose["health"]:.0f}/20    Food {pose["food"]:.0f}/20')
        bar(237,'Injury / pain proxy',pose.get('pain_proxy',0),'#eb866c')
        bar(280,'Stress proxy',pose.get('stress_proxy',0),'#e6b969')
        bar(323,'Hunger',pose.get('hunger',0),'#79d6b6')
        text(364,'ACTIVE DRIVE / estimated, not measured mood',12)
        text(384,pose.get('drive_state','Waiting'),16,'#ecf5fa')
        text(406,'Game-derived scores; not a verified pain circuit.',12)
        text(431,'TRANSMITTER-ANNOTATED CELLS',14,'#79d6b6')
        text(455,'Firing cells / retained cells    |    spikes',12)
        for row,(name,data) in enumerate(pose.get('transmitters',{}).items()):
            short={'acetylcholine':'ACh','gaba':'GABA'}.get(name,name.capitalize())
            text(478+row*23,f'{short:<13} {data["active"]:,}/{data["total"]:,}  |  {data["spikes"]:,}',13)
        text(622,'Activity counts, not chemical concentrations.',12,'#e6b969')
        text(639,'DA / 5-HT / OA signaling is not modeled.',12,'#e6b969')
        text(668,f'Action: {pose["action"]}  (x{pose.get("action_repeat",0)})',15,'#ecf5fa')
        text(693,f'Inventory: {"open - movement exits" if pose.get("inventory_open") else "closed"}',13)
        text(716,f'Deaths {pose.get("deaths",0)}   Updates {pose.get("updates",0)}   Reward {pose["reward"]:+.2f}',13)
        text(740,f'Dopamine-like reward pulse: +{pose.get("reward_pulse",0):.3f}',13,'#79d6b6')
        text(763,'DEPTH RAYS / bright = nearer surface',12)
        for i,value in enumerate(pose.get('depth_rays',[0]*9)):
            x=20+(i%3)*32; y=785+(i//3)*22
            level=int(30+210*value)
            draw.rectangle((x,y,x+27,y+17),fill=(level,level,min(255,level+10)))
        text(790,'8-block range',12,x=135)
        text(811,'Not camera pixels',12,x=135)
        details=brain.details(pose['brain_counts'])
        if details:
            text(855,f'Neuron {details["bodyId"]}: {recent[brain.selected]} recent spikes',13)
            text(875,str(details['type'])[:43],12)
        else:
            text(855,'Click a firing neuron for its ID and spike count.',12)
        text(902,'H all / active    L branches    C reset view',12)
        text(923,'Right-drag orbit / wheel zoom / Esc save + exit',12)
        image=image.resize((self.width,height),Image.Resampling.LANCZOS)
        if self.texture is None or self.height!=height:
            if self.texture: self.texture.release()
            self.texture=self.ctx.texture((self.width,height),4)
            self.height=height
        self.texture.write(image.tobytes())
