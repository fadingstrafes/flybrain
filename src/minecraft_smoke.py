"""Real sparse-brain synthetic-input smoke check; does not claim Minecraft gameplay."""
import argparse
from pathlib import Path
import tempfile
import json
import base64
import numpy as np
from src.live_brain import load_live_graph
from src.minecraft_brain import NeuralAdapter, MinecraftSession
from src.minecraft_learning import MinecraftLearner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='auto')
    p.add_argument('--output',type=Path,default=Path('runs/minecraft_smoke'))
    p.add_argument('--eyes',type=Path,help='192x64 PNG containing two real Minecraft eye captures')
    args = p.parse_args()
    graph = load_live_graph()
    adapter = NeuralAdapter(graph,device=args.device)
    # With all body inputs off, images alone must drive annotated visual cells
    # and propagate beyond them through the existing sparse connectome.
    eyes=np.zeros((2,64,96,3),dtype=np.uint8)
    def visual(frame): return dict(width=96,height=64,frame=frame,rgb=base64.b64encode(eyes.tobytes()).decode())
    adapter.set_vision(visual(0));adapter.encode(np.zeros(32))
    dark_spikes=int(adapter.recent_counts.sum())
    adapter.reset();eyes[:]=180
    adapter.set_vision(visual(1));adapter.encode(np.zeros(32))
    visual_spikes=int(adapter.recent_counts[adapter.eyes.indices].sum())
    downstream=int(adapter.recent_counts.sum())-visual_spikes
    assert visual_spikes>0 and downstream>0 and dark_spikes==0
    adapter.reset()
    captured=None
    if args.eyes:
        from PIL import Image
        pixels=np.asarray(Image.open(args.eyes).convert('RGB'))
        if pixels.shape!=(64,192,3): raise ValueError('Expected 192x64 eye capture')
        captured=np.stack((pixels[:,:96],pixels[:,96:]))
    args.output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        policy = Path(d)/'policy.json'
        learner = MinecraftLearner(policy)
        session = MinecraftSession(adapter,learner,args.output/'events.jsonl')
        for seq in range(8):
            senses = np.zeros(32); senses[[0,2,3,16+seq%9]] = 1
            eyes[:,32:]=[55,95,35]
            eyes[:,:32]=[115,160,215]
            eyes[0,20:36,10+seq*4:18+seq*4]=[190,30,20]
            if captured is not None: eyes[:]=captured
            session.step(dict(version=1,session='synthetic-smoke',life=0,seq=seq,dead=False,
                              health=20,food=10+(2 if seq>3 else 0),senses=senses.tolist(),vision=visual(seq+2)))
        spikes = int(adapter.counts.sum())
        assert spikes > 0, 'No neural response'
        pose = session.snapshot()
        session.step(dict(version=1,session='synthetic-smoke',life=0,seq=8,dead=True,
                          health=0,food=12,senses=[0.]*32))
        reloaded = MinecraftLearner(policy)
        np.testing.assert_array_equal(learner.weights,reloaded.weights)
        assert reloaded.deaths == 1
        report = dict(device=str(adapter.brain.device),neurons=len(graph.ids),spikes=spikes,
                      updates=learner.updates,deaths=learner.deaths,checkpoint_restored=True,
                      dark_spikes=dark_spikes,visual_input_spikes=visual_spikes,downstream_visual_spikes=downstream,
                      visual_input_cells=len(adapter.eyes.indices))
        report['eye_source']=str(args.eyes) if args.eyes else 'synthetic validation pattern'
        try:
            import moderngl
            from PIL import Image
            from src.brain_view import BrainMap,BrainRenderer
            from src.minecraft_panel import MinecraftPanel
            from src.minecraft_eye_panel import EyePanel
            ctx=moderngl.create_standalone_context(backend='egl')
            brain=BrainMap(graph.ids)
            renderer=BrainRenderer(ctx,brain,brain.load_fibers(sorted(Path('runs').glob('*/skeleton_cache'))))
            target=ctx.simple_framebuffer((1400,960)); target.use(); target.clear(.015,.018,.028)
            ctx.enable(moderngl.BLEND)
            pose['brain_activity']=(pose['brain_recent']>0).astype('f4')
            renderer.draw(pose,(0,250,1020,710),dict(brain.camera_preset('front',1020/710),active_only=True))
            panel=MinecraftPanel(ctx); panel.update(pose,960,brain); panel.draw(1400,960)
            eye_panel=EyePanel(ctx);eye_panel.update(pose,1020,250);eye_panel.draw()
            Image.frombytes('RGB',(1400,960),target.read(components=3)).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(args.output/'brain.png')
            report['viewer_rendered']=True
            eye_panel.release(); panel.release(); renderer.release(); target.release(); ctx.release()
        except ImportError:
            report['viewer_rendered']=False
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))


if __name__=='__main__': main()
