"""Explicit survival reward shaping; a modeling assumption, not fly physiology."""
import numpy as np


def validate_world(world):
    if world is None:
        return
    if not isinstance(world,dict):
        raise ValueError('Expected world telemetry object')
    for key in ('food_visible','food_left','food_right','shelter_exit','shelter_light','shelter_hazard'):
        value=world.get(key,0)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not 0<=value<=1:
            raise ValueError('Invalid world '+key)
    for key, maximum in [('food_stock',100000),('materials',100000),('placed',2**31),
                          ('broken',2**31),('ticks',2**31),('threat',1),('cover',1),('danger',1)]:
        value=world.get(key)
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not np.isfinite(value) or not 0<=value<=maximum:
            raise ValueError('Invalid world '+key)
    for key,size,maximum in [('position',3,32000000),('hotbar',9,1)]:
        values=np.asarray(world.get(key),dtype=float)
        if values.shape!=(size,) or not np.isfinite(values).all() or np.any(np.abs(values)>maximum):
            raise ValueError('Invalid world '+key)
        if key=='hotbar' and np.any(values<0):
            raise ValueError('Invalid hotbar')


def state_features(neural, observation):
    world=observation.get('world') or {}
    extra=[min(world.get('food_stock',0)/64,1), min(world.get('materials',0)/64,1),
           world.get('threat',0),world.get('cover',0),world.get('danger',0),
           min(world.get('placed',0)/32,1),min(world.get('broken',0)/32,1),float(bool(world))]
    foraging=[world.get(k,0) for k in ('food_visible','food_left','food_right','shelter_exit','shelter_light','shelter_hazard')]
    return np.r_[neural,observation['senses'],extra,world.get('hotbar',[1]*9),foraging]


class SurvivalDrives:
    def __init__(self): self.reset()

    def reset(self):
        self.food_high=0
        self.material_high=0
        self.cover_best=0.
        self.idle_ticks=0
        self.build_credit=0
        self.visited=set()

    def baseline(self,o):
        w=o.get('world') or {}
        self.food_high=w.get('food_stock',0)
        self.material_high=w.get('materials',0)
        self.cover_best=w.get('cover',0)
        if 'position' in w: self.visited.add(self.cell(w))

    @staticmethod
    def cell(w): return tuple(int(np.floor(v/4)) for v in w['position'])

    def transition(self,old,new):
        parts={'eating':0. if new['dead'] else .1*min(max(new['food']-old['food'],0),max(20-old['food'],0)),
               'injury':-.15*max(old['health']-new['health'],0), 'death':-2. if new['dead'] else 0.}
        if new['dead']: return parts
        a,b=old.get('world'),new.get('world')
        if not a or not b: return parts
        hunger=max(0,1-old['food']/20)
        # High-water marks prevent drop/pickup or repeated cover-entry reward loops.
        fresh_food=max(0,min(b['food_stock'],16)-min(self.food_high,16))
        fresh_material=max(0,min(b['materials'],32)-min(self.material_high,32))
        parts['food_acquired']=fresh_food*(.03+.07*hunger)
        parts['gathering']=fresh_material*.015
        self.food_high=max(self.food_high,b['food_stock'])
        self.material_high=max(self.material_high,b['materials'])
        new_cover=max(0,b['cover']-self.cover_best)
        parts['shelter']=new_cover*(.1+.2*max(a['danger'],b['danger']))
        placed=max(0,b['placed']-a['placed'])
        # Construction reward requires improved actual cover and available lifetime budget.
        parts['shelter_building']=min(placed,8-self.build_credit)*.04 if new_cover>0 else 0.
        if new_cover>0: self.build_credit=min(8,self.build_credit+placed)
        self.cover_best=max(self.cover_best,b['cover'])
        parts['threat_exposure']=-.01*b['threat']
        moved=float(np.linalg.norm(np.asarray(b['position'])-a['position']))>.15
        changed=(b['food_stock']!=a['food_stock'] or b['materials']!=a['materials']
                 or b['broken']!=a['broken'] or placed>0 or new_cover>0 or new['food']>old['food'])
        elapsed=max(0,min(b['ticks']-a['ticks'],100))
        self.idle_ticks=0 if moved or changed else self.idle_ticks+elapsed
        rest=new['health']<20 and new['food']>=18 and b['threat']==0
        parts['inactivity']=-.01 if self.idle_ticks>200 and not rest else 0.
        cell=self.cell(b)
        parts['exploration']=.005 if moved and cell not in self.visited and b['threat']==0 and len(self.visited)<4096 else 0.
        if len(self.visited)<4096: self.visited.add(cell)
        return parts


def estimated_drive(o,parts,idle_ticks):
    w=o.get('world') or {}
    if o['dead']: return 'Dead'
    if parts.get('injury',0)<0 or w.get('threat',0)>.2: return 'Threatened'
    if o['food']<14: return 'Food-seeking drive'
    if o['health']<20 and o['food']>=18: return 'Recovery / rest'
    if w.get('danger',0)>.4 and w.get('cover',0)<.7: return 'Shelter-seeking drive'
    if sum(parts.values())>.02: return 'Rewarded / recent progress'
    if idle_ticks>200: return 'Inactive / stalled'
    return 'Exploration drive'
