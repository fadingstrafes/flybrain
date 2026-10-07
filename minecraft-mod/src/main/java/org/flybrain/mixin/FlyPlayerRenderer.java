package org.flybrain.mixin;

import net.minecraft.client.renderer.entity.EntityRenderDispatcher;
import net.minecraft.client.renderer.culling.Frustum;
import net.minecraft.world.entity.Entity;
import org.flybrain.FlyBrainMod;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/** Suppress only the agent's human avatar, independent of synced visibility. */
@Mixin(EntityRenderDispatcher.class)
public abstract class FlyPlayerRenderer {
    @Inject(method="shouldRender", at=@At("HEAD"), cancellable=true)
    private void flybrain$hideAvatar(Entity entity, Frustum frustum, double x, double y,
                                    double z, float partialTick, CallbackInfoReturnable<Boolean> ci) {
        if (entity.getUUID().equals(FlyBrainMod.ID) || (org.flybrain.FlyEyes.capturing && org.flybrain.FlyModel.VISUAL_IDS.contains(entity.getUUID()))) ci.setReturnValue(false);
    }
}
