package org.flybrain;

import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.pipeline.TextureTarget;
import com.mojang.renderpearl.api.GpuFormat;
import net.minecraft.client.*;
import org.flybrain.mixin.*;
import org.joml.Matrix4f;
import java.util.concurrent.atomic.AtomicInteger;

/** Two actual RGB world renders from the fly, independent of observer camera. */
public final class FlyEyes {
    public static boolean capturing;
    private static final int W=96,H=64;
    private static RenderTarget target;
    private static Camera eye;
    private static long last,sequence;
    private static volatile boolean pending;
    public static String error;

    public static void capture() {
        Minecraft mc=Minecraft.getInstance();
        if(capturing||mc.level==null||mc.player==null||mc.isPaused()) return;
        var fly=mc.level.getPlayerByUUID(FlyBrainMod.ID);
        if(fly==null) {EyeFrames.clear();return;}
        long now=System.nanoTime();
        if(pending||now-last<250_000_000L||error!=null) return;
        last=now;capturing=true;pending=true;
        var renderer=mc.gameRenderer;
        var access=(EyeRendererAccess)renderer;
        Camera original=renderer.mainCamera(); RenderTarget originalTarget=renderer.mainRenderTarget();
        CameraType originalType=mc.options.getCameraType();
        var state=renderer.gameRenderState();
        int width=state.windowRenderState.width,height=state.windowRenderState.height;
        boolean bob=state.optionsRenderState.bobView;
        byte[] pixels=new byte[2*W*H*3]; AtomicInteger complete=new AtomicInteger();
        long frame=++sequence; int entity=fly.getId();
        try {
            if(target==null) target=new TextureTarget("FlyBrain eyes",W,H,GpuFormat.RGBA8_UNORM,GpuFormat.D32_FLOAT);
            if(eye==null) eye=new Camera();
            eye.setLevel(mc.level);eye.setEntity(fly);
            access.flybrain$camera(eye);access.flybrain$target(target);
            if(mc.levelRenderer.skyRenderer()!=null) ((EyeSkyAccess)mc.levelRenderer.skyRenderer()).flybrain$target(target);
            mc.options.setCameraType(CameraType.FIRST_PERSON);
            state.windowRenderState.width=W;state.windowRenderState.height=H;
            state.optionsRenderState.bobView=false;
            for(int side=0;side<2;side++) {
                eye.update(mc.getDeltaTracker());
                var c=(EyeCameraAccess)eye;
                var sideways=net.minecraft.world.phys.Vec3.directionFromRotation(0,fly.getYRot()+90);
                c.flybrain$position(fly.getEyePosition().add(sideways.scale(side==0?-.12:.12)));
                eye.attributeProbe().tick(mc.level,eye.position());
                c.flybrain$rotation(fly.getYRot()+(side==0?-55:55),fly.getXRot());
                c.flybrain$projection(.05f,256,100,W,H);
                c.flybrain$frustum(eye.getViewRotationMatrix(new Matrix4f()),
                    new Matrix4f().perspective((float)Math.toRadians(100),W/(float)H,.05f,256),eye.position());
                access.flybrain$extractCamera(mc.getDeltaTracker(),1);
                mc.levelExtractor.extract(mc.getDeltaTracker(),eye,1);
                com.mojang.blaze3d.systems.RenderSystem.getDevice().createCommandEncoder().clearColorAndDepthTextures(
                    target.getColorTexture(),new org.joml.Vector4f(0,0,0,1),target.getDepthTexture(),0);
                access.flybrain$uniforms().update(W,H,state.optionsRenderState.glintStrength,
                    state.levelRenderState.gameTime,1,0,eye.position(),false);
                renderer.renderLevel();
                final int offset=side*W*H*3;
                Screenshot.takeScreenshot(target,image -> {
                    try {
                        for(int y=0;y<H;y++) for(int x=0;x<W;x++) {
                            int color=image.getPixel(x,y),i=offset+(y*W+x)*3;
                            pixels[i]=(byte)(color>>16);pixels[i+1]=(byte)(color>>8);pixels[i+2]=(byte)color;
                        }
                        if(complete.incrementAndGet()==2) {EyeFrames.publish(entity,frame,pixels);pending=false;}
                    } finally {image.close();}
                });
                mc.levelRenderer.endFrame();
            }
        } catch(Exception failure) {
            error=failure.toString(); pending=false;
            org.slf4j.LoggerFactory.getLogger("FlyBrainEyes").error("Eye capture stopped",failure);
        } finally {
            access.flybrain$camera(original);access.flybrain$target(originalTarget);
            if(mc.levelRenderer.skyRenderer()!=null) ((EyeSkyAccess)mc.levelRenderer.skyRenderer()).flybrain$target(originalTarget);
            mc.options.setCameraType(originalType);
            state.windowRenderState.width=width;state.windowRenderState.height=height;
            state.optionsRenderState.bobView=bob;
            capturing=false;
        }
    }
    public static void close() {
        if(target!=null) target.destroyBuffers();target=null;eye=null;pending=false;error=null;EyeFrames.clear();
    }
}
