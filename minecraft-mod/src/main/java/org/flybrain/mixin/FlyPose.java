package org.flybrain.mixin;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.entity.Pose;
import org.flybrain.FlyBrainMod;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
@Mixin(Player.class)
public abstract class FlyPose {
    @Inject(method="updatePlayerPose",at=@At("HEAD"),cancellable=true)
    private void flybrain$pose(CallbackInfo ci) {
        Player player = (Player)(Object)this;
        if (!player.level().isClientSide() && player.getUUID().equals(FlyBrainMod.ID) && FlyBrainMod.crawling) {
            player.setPose(Pose.SWIMMING); ci.cancel();
        }
    }
}
