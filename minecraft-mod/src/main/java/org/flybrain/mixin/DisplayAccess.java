package org.flybrain.mixin;
import com.mojang.math.Transformation;
import net.minecraft.world.entity.Display;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Invoker;
@Mixin(Display.class)
public interface DisplayAccess {
    @Invoker("setTransformation") void flybrain$transform(Transformation value);
    @Invoker("setPosRotInterpolationDuration") void flybrain$interpolate(int ticks);
}
