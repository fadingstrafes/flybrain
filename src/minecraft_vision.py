"""RGB compound-eye approximation onto annotated photoreceptors.

Left/right identity and cell classes are data-derived. Facet assignments and
RGB spectral response are synthetic: no retinal coordinates or UV are available.
"""
import base64
from pathlib import Path
import numpy as np

WIDTH, HEIGHT = 96, 64


def decode_vision(data):
    if data is None: return None
    if not isinstance(data,dict) or data.get('width')!=WIDTH or data.get('height')!=HEIGHT:
        raise ValueError('Expected two 96x64 RGB eyes')
    if type(data.get('frame')) is not int or data['frame']<0:
        raise ValueError('Invalid eye frame')
    if not isinstance(data.get('rgb'),str) or len(data['rgb'])!=2*WIDTH*HEIGHT*4:
        raise ValueError('Invalid eye payload size')
    raw=base64.b64decode(data['rgb'],validate=True)
    if len(raw)!=2*WIDTH*HEIGHT*3: raise ValueError('Invalid RGB bytes')
    return np.frombuffer(raw,dtype=np.uint8).reshape(2,HEIGHT,WIDTH,3).copy()


def sample_facets(rgb):
    """768 staggered angular samples per eye, constrained to rendered field."""
    row,col=np.indices((24,32))
    az=np.deg2rad(((col+.5+.5*(row%2))/32-.5)*112)
    el=np.deg2rad((.5-(row+.5)/24)*80)
    tan_v=np.tan(np.deg2rad(50));tan_h=tan_v*WIDTH/HEIGHT
    # Sample camera projection for a spherical angular grid.
    u=(np.tan(az)/tan_h+1)*.5
    v=(1-np.tan(el)/(np.cos(az)*tan_v))*.5
    x=np.clip((u*WIDTH).astype(int),0,WIDTH-1)
    y=np.clip((v*HEIGHT).astype(int),0,HEIGHT-1)
    return rgb[:,y,x].astype(np.float32)/255


class CompoundEyes:
    def __init__(self,ids,cache=Path('data/cache')):
        import pandas as pd
        metadata=pd.read_parquet(Path(cache)/'metadata.parquet',columns=['bodyId','type','instance'])
        metadata=metadata.drop_duplicates('bodyId').set_index('bodyId').reindex(ids)
        types=metadata.type.fillna('').to_numpy(dtype=str)
        names=metadata.instance.fillna('').to_numpy(dtype=str)
        indices=[]; eyes=[]; families=[];facets=[]
        rng=np.random.default_rng(2603)
        for eye,side in enumerate(('_L','_R')):
            for family,prefix in enumerate(('R1-R6','R7','R8','L2','L3')):
                selected=np.flatnonzero(np.char.startswith(types,prefix)&np.char.endswith(names,side))
                rng.shuffle(selected)
                indices.extend(selected);eyes.extend([eye]*len(selected));families.extend([family]*len(selected))
                facets.extend(np.arange(len(selected))%768)
        self.indices=np.asarray(indices,dtype=np.int64)
        self.eyes=np.asarray(eyes);self.families=np.asarray(families);self.facets=np.asarray(facets)
        if not len(self.indices): raise ValueError('No annotated left/right photoreceptors found')
        self.readout_bins=self.eyes*16+self.facets//48
        self.photoreceptors=self.indices[self.families<3]
        self.relays=self.indices[self.families>=3]
        self.reset()

    def reset(self):
        self.rgb=None;self.faceted=None;self.previous=None;self.frame=None
        self.motion=None;self.currents=np.zeros(len(self.indices),dtype=np.float32)
        self.base_currents=self.currents.copy()

    def update(self,rgb,frame):
        if rgb is None:
            self.reset();return
        if frame==self.frame:
            self.motion.fill(0)
            self.currents=self.base_currents.copy()
            return
        self.frame=frame;self.rgb=rgb.copy();self.faceted=sample_facets(rgb)
        luminance=self.faceted @ np.array([.2126,.7152,.0722],dtype=np.float32)
        self.motion=np.zeros_like(luminance) if self.previous is None else np.abs(luminance-self.previous)
        self.previous=luminance.copy()
        flat=self.faceted.reshape(2,768,3)
        lum=luminance.reshape(2,768);change=self.motion.reshape(2,768)
        signal=lum[self.eyes,self.facets]
        # RGB surrogate for chromatic populations, not biological spectral tuning.
        r7=self.families==1;r8=self.families==2
        signal[r7]=flat[self.eyes[r7],self.facets[r7],2]
        signal[r8]=flat[self.eyes[r8],self.facets[r8],1]
        self.base_currents=(1.05*signal).astype(np.float32)
        # Photoreceptor transmitters are neutral in the project's NT model.
        # Explicit synthetic lamina input enables downstream activity without
        # modifying anatomical weights or claiming histaminergic transmission.
        self.base_currents[self.families>=3]*=.65/1.05
        self.currents=self.base_currents+.45*change[self.eyes,self.facets]

    def readout(self,counts):
        return 1-np.exp(-np.bincount(self.readout_bins,weights=counts[self.indices],minlength=32)/40)

    def snapshot(self,counts):
        return dict(eye_rgb=None if self.rgb is None else self.rgb.copy(),
                    eye_facets=None if self.faceted is None else self.faceted.copy(),
                    eye_motion=None if self.motion is None else self.motion.copy(),
                    visual_input_cells=len(self.indices),
                    photoreceptor_cells=len(self.photoreceptors),relay_cells=len(self.relays),
                    visual_input_firing=int(np.count_nonzero(counts[self.indices])),
                    visual_input_spikes=int(counts[self.indices].sum()),eye_frame=self.frame)
