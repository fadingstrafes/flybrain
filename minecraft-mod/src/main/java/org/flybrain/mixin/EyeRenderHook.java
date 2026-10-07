package org.flybrain.mixin;
import net.minecraft.client.renderer.GameRenderer;
import org.flybrain.FlyEyes;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
@Mixin(GameRenderer.class)
public abstract class EyeRenderHook {
    @Inject(method="render",at=@At("TAIL"))
    private void flybrain$capture(CallbackInfo ci) { FlyEyes.capture(); }
    @Inject(method="render3dHud",at=@At("HEAD"),cancellable=true)
    private void flybrain$noHands(CallbackInfo ci) { if(FlyEyes.capturing) ci.cancel(); }
    @Inject(method="close",at=@At("HEAD"))
    private void flybrain$close(CallbackInfo ci) { FlyEyes.close(); }
}
