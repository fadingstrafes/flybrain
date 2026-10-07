package org.flybrain.mixin;
import com.mojang.blaze3d.pipeline.RenderTarget;
import net.minecraft.client.Camera;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.renderer.GameRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Mutable;
import org.spongepowered.asm.mixin.gen.Accessor;
import org.spongepowered.asm.mixin.gen.Invoker;
@Mixin(GameRenderer.class)
public interface EyeRendererAccess {
    @Accessor("globalSettingsUniform") net.minecraft.client.renderer.GlobalSettingsUniform flybrain$uniforms();
    @Mutable @Accessor("mainCamera") void flybrain$camera(Camera value);
    @Mutable @Accessor("mainRenderTarget") void flybrain$target(RenderTarget value);
    @Invoker("extractCamera") void flybrain$extractCamera(DeltaTracker delta,float partial);
}
