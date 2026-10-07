package org.flybrain.mixin;
import net.minecraft.client.Camera;
import net.minecraft.world.phys.Vec3;
import org.joml.Matrix4f;
import org.joml.Matrix4fc;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Invoker;
@Mixin(Camera.class)
public interface EyeCameraAccess {
    @Invoker("setRotation") void flybrain$rotation(float yaw,float pitch);
    @Invoker("setPosition") void flybrain$position(Vec3 position);
    @Invoker("setupPerspective") void flybrain$projection(float near,float far,float fov,float width,float height);
    @Invoker("prepareCullFrustum") void flybrain$frustum(Matrix4fc view,Matrix4f projection,Vec3 position);
}
