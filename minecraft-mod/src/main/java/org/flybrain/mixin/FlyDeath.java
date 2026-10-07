package org.flybrain.mixin;
import carpet.patches.EntityPlayerMPFake;
import net.minecraft.world.damagesource.DamageSource;
import org.flybrain.FlyBrainMod;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
@Mixin(EntityPlayerMPFake.class)
public abstract class FlyDeath {
    @Inject(method="die",at=@At("HEAD"))
    private void flybrain$death(DamageSource source, CallbackInfo ci) {
        if (((EntityPlayerMPFake)(Object)this).getUUID().equals(FlyBrainMod.ID)) FlyBrainMod.died = true;
    }
}
