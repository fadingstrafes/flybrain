package org.flybrain;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import com.mojang.blaze3d.platform.InputConstants;
import net.minecraft.client.CameraType;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;
import net.minecraft.world.entity.Entity;

/** Observer only: never feeds the human player's actions to the brain. */
public final class FlyBrainClient implements ClientModInitializer {
    private boolean f8, f9, following;
    private CameraType original;
    @Override public void onInitializeClient() { ClientTickEvents.START_CLIENT_TICK.register(mc -> { if(following) net.minecraft.client.KeyMapping.releaseAll(); }); ClientTickEvents.END_CLIENT_TICK.register(this::tick); }
    private void tick(Minecraft mc) {
        boolean a=InputConstants.isKeyDown(InputConstants.KEY_F8), b=InputConstants.isKeyDown(InputConstants.KEY_F9);
        if(mc.player==null||mc.level==null) {
            if(following) mc.options.setCameraType(original);
            following=false;f8=a;f9=b;return;
        }
        Entity fly=mc.level.getPlayerByUUID(FlyBrainMod.ID);
        if(a&&!f8&&mc.getConnection()!=null) mc.getConnection().sendCommand("flybrain toggle");
        if(b&&!f9) {
            if(following) {
                mc.setCameraEntity(mc.player);mc.options.setCameraType(original);following=false;
            } else if(fly!=null) {
                original=mc.options.getCameraType();mc.options.setCameraType(CameraType.THIRD_PERSON_BACK);
                mc.setCameraEntity(fly);following=true;
                mc.player.sendSystemMessage(Component.literal("FlyBrain camera: F9 returns to your player. Your body remains in the world."));
            }
        }
        if(following) {
            if(fly==null) {mc.setCameraEntity(mc.player);mc.options.setCameraType(original);following=false;}
            else {
                mc.setCameraEntity(fly);
                mc.options.keyUp.setDown(false);mc.options.keyDown.setDown(false);
                mc.options.keyLeft.setDown(false);mc.options.keyRight.setDown(false);
                mc.options.keyJump.setDown(false);mc.options.keyAttack.setDown(false);mc.options.keyUse.setDown(false);
            }
        }
        f8=a;f9=b;
    }
}
