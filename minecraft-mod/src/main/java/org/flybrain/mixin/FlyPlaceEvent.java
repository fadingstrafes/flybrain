package org.flybrain.mixin;

import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.context.BlockPlaceContext;
import net.minecraft.world.InteractionResult;
import org.flybrain.FlyBrainMod;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(BlockItem.class)
public abstract class FlyPlaceEvent {
    @Inject(method="place", at=@At("RETURN"))
    private void flybrain$placed(BlockPlaceContext context, CallbackInfoReturnable<InteractionResult> ci) {
        if (!context.getLevel().isClientSide() && context.getPlayer()!=null
                && context.getPlayer().getUUID().equals(FlyBrainMod.ID) && ci.getReturnValue().consumesAction())
            FlyBrainMod.placed++;
    }
}
