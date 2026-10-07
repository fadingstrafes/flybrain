package org.flybrain.mixin;
import com.mojang.blaze3d.pipeline.RenderTarget;
import net.minecraft.client.renderer.SkyRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Mutable;
import org.spongepowered.asm.mixin.gen.Accessor;
@Mixin(SkyRenderer.class)
public interface EyeSkyAccess {
    @Mutable @Accessor("renderTarget") void flybrain$target(RenderTarget target);
}
