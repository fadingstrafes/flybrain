package org.flybrain;

import com.google.gson.*;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.screens.inventory.AbstractContainerScreen;
import net.minecraft.client.gui.screens.inventory.InventoryScreen;
import net.minecraft.core.component.DataComponents;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.network.chat.Component;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.inventory.ContainerInput;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.phys.*;
import com.mojang.blaze3d.platform.InputConstants;
import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

/** Transport and primitive player controls only. Decisions live in the brain service. */
public final class PlayerControlPrototype implements ClientModInitializer {
    private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(2)).build();
    private CompletableFuture<HttpResponse<String>> pending;
    private boolean enabled, keyWasDown, wasDead, awaitingRespawn;
    private String session = UUID.randomUUID().toString();
    private int life, seq, ticks, holdUntil, slot;
    private String action = "idle";

    @Override public void onInitializeClient() {
        ClientTickEvents.END_CLIENT_TICK.register(this::tick);
    }

    private void message(Minecraft mc, String text) {
        if (mc.player != null) mc.player.sendSystemMessage(Component.literal("FlyBrain: " + text));
    }

    private void release(Minecraft mc) {
        mc.options.keyUp.setDown(false); mc.options.keyDown.setDown(false);
        mc.options.keyLeft.setDown(false); mc.options.keyRight.setDown(false);
        mc.options.keyJump.setDown(false); mc.options.keyShift.setDown(false);
        if (mc.gameMode != null) {
            mc.gameMode.stopDestroyBlock();
            if (mc.player != null && mc.player.isUsingItem()) mc.gameMode.releaseUsingItem(mc.player);
        }
        action = "idle";
    }

    private void stop(Minecraft mc) {
        enabled = false; release(mc);
        if (pending != null) pending.cancel(true);
        pending = null;
    }

    private void tick(Minecraft mc) {
        ticks++;
        boolean down = InputConstants.isKeyDown(InputConstants.KEY_F8);
        if (down && !keyWasDown) {
            if (enabled) { stop(mc); message(mc, "control OFF"); }
            else if (mc.player != null) {
                enabled = true; session = UUID.randomUUID().toString(); seq = life = slot = 0;
                wasDead = awaitingRespawn = false; holdUntil = ticks;
                message(mc, "control ON — F8 to stop");
            }
        }
        keyWasDown = down;
        if (!enabled) return;
        if (mc.player == null || mc.level == null || mc.gameMode == null) { stop(mc); return; }
        if (mc.isPaused()) { stop(mc); message(mc, "paused; F8 to resume"); return; }
        boolean dead = mc.player.isDeadOrDying();
        if (wasDead && !dead) { life++; awaitingRespawn = false; slot = 0; holdUntil = ticks; }
        wasDead = dead;
        if (dead) release(mc);
        if (pending != null && pending.isDone()) {
            try {
                HttpResponse<String> result = pending.join(); pending = null;
                if (result.statusCode() != 200) throw new IllegalStateException("brain HTTP " + result.statusCode());
                JsonObject reply = JsonParser.parseString(result.body()).getAsJsonObject();
                if (!reply.get("session").getAsString().equals(session) || reply.get("seq").getAsInt() != seq-1
                        || reply.get("life").getAsInt() != life) throw new IllegalStateException("stale reply");
                if (dead && reply.get("respawn").getAsBoolean()) {
                    awaitingRespawn = true; mc.player.respawn(); mc.gui.setScreen(null);
                } else if (!dead) {
                    apply(mc, reply.get("action").getAsString());
                }
            } catch (Exception error) {
                stop(mc); message(mc, "connection stopped; check brain service, then F8"); return;
            }
        }
        if (!dead && ticks < holdUntil) maintain(mc);
        if (ticks >= holdUntil && !action.equals("idle")) release(mc);
        if (pending == null && !awaitingRespawn && (dead || ticks >= holdUntil)) {
            JsonObject observation = observe(mc, dead);
            HttpRequest request = HttpRequest.newBuilder(URI.create("http://127.0.0.1:8765/step"))
                .timeout(Duration.ofSeconds(5)).header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(observation.toString())).build();
            pending = http.sendAsync(request, HttpResponse.BodyHandlers.ofString());
        }
    }

    private void apply(Minecraft mc, String next) {
        release(mc); action = next;
        holdUntil = ticks + ((next.equals("use") || next.equals("attack")) ? 40 : 10);
        if (next.equals("look_left")) mc.player.setYRot(mc.player.getYRot()-15);
        if (next.equals("look_right")) mc.player.setYRot(mc.player.getYRot()+15);
        if (next.equals("look_up")) mc.player.setXRot(Math.max(-90,mc.player.getXRot()-15));
        if (next.equals("look_down")) mc.player.setXRot(Math.min(90,mc.player.getXRot()+15));
        if (next.startsWith("hotbar_")) mc.player.getInventory().setSelectedSlot(Integer.parseInt(next.substring(7)));
        if (next.equals("inventory")) {
            if (mc.gui.screen() instanceof AbstractContainerScreen<?>) mc.player.closeContainer();
            else mc.gui.setScreen(new InventoryScreen(mc.player));
        }
        int size = mc.player.containerMenu.slots.size();
        if (next.equals("slot_next")) slot = (slot+1)%size;
        if (next.equals("slot_previous")) slot = (slot+size-1)%size;
        slot = Math.floorMod(slot,size);
        if (mc.gui.screen() instanceof AbstractContainerScreen<?> && next.startsWith("slot_") && !next.equals("slot_next") && !next.equals("slot_previous")) {
            mc.gameMode.handleContainerInput(mc.player.containerMenu.containerId,slot,
                next.equals("slot_right_click") ? 1 : 0,
                next.equals("slot_quick_move") ? ContainerInput.QUICK_MOVE : ContainerInput.PICKUP,mc.player);
        }
        if (mc.gui.screen() == null && next.equals("use")) {
            if (mc.hitResult instanceof BlockHitResult hit && hit.getType() == HitResult.Type.BLOCK) {
                if (!mc.gameMode.useItemOn(mc.player,InteractionHand.MAIN_HAND,hit).consumesAction())
                    mc.gameMode.useItem(mc.player,InteractionHand.MAIN_HAND);
            } else if (mc.hitResult instanceof EntityHitResult hit) {
                mc.gameMode.interact(mc.player,hit.getEntity(),hit,InteractionHand.MAIN_HAND);
            } else mc.gameMode.useItem(mc.player,InteractionHand.MAIN_HAND);
        }
        if (mc.gui.screen() == null && next.equals("attack")) {
            if (mc.hitResult instanceof EntityHitResult hit) mc.gameMode.attack(mc.player,hit.getEntity());
            else if (mc.hitResult instanceof BlockHitResult hit && hit.getType() == HitResult.Type.BLOCK)
                mc.gameMode.startDestroyBlock(hit.getBlockPos(),hit.getDirection());
        }
        maintain(mc);
    }

    private void maintain(Minecraft mc) {
        if (mc.gui.screen() != null) return;
        mc.options.keyUp.setDown(action.equals("forward") || action.equals("forward_jump"));
        mc.options.keyDown.setDown(action.equals("back"));
        mc.options.keyLeft.setDown(action.equals("left")); mc.options.keyRight.setDown(action.equals("right"));
        mc.options.keyJump.setDown(action.equals("jump") || action.equals("forward_jump"));
        mc.options.keyShift.setDown(action.equals("sneak"));
        if (action.equals("attack") && mc.hitResult instanceof BlockHitResult hit && hit.getType() == HitResult.Type.BLOCK)
            mc.gameMode.continueDestroyBlock(hit.getBlockPos(),hit.getDirection());
    }

    private static double unit(double value) { return Math.max(0,Math.min(1,value)); }
    private static double food(ItemStack stack) { return stack.get(DataComponents.FOOD) == null ? 0 : 1; }
    private JsonObject observe(Minecraft mc, boolean dead) {
        JsonObject o = new JsonObject();
        o.addProperty("version",1); o.addProperty("session",session); o.addProperty("seq",seq++);
        o.addProperty("life",life); o.addProperty("dead",dead);
        o.addProperty("health",Math.max(0,mc.player.getHealth()));
        o.addProperty("food",mc.player.getFoodData().getFoodLevel());
        double[] s = new double[32];
        s[0] = 1; s[1] = 1-unit(mc.player.getHealth()/20); s[2] = 1-mc.player.getFoodData().getFoodLevel()/20.;
        s[3] = mc.player.onGround()?1:0; s[4] = mc.player.isInWater()?1:0;
        s[5] = mc.player.isOnFire()?1:0; s[6] = unit(mc.player.getDeltaMovement().length());
        s[7] = food(mc.player.getMainHandItem()); s[8] = mc.player.getInventory().getSelectedSlot()/8.;
        s[9] = mc.gui.screen() instanceof AbstractContainerScreen<?>?1:0;
        slot = Math.floorMod(slot,mc.player.containerMenu.slots.size());
        s[10] = slot/(double)Math.max(1,mc.player.containerMenu.slots.size()-1);
        ItemStack selected = mc.player.containerMenu.slots.get(slot).getItem();
        s[11] = unit(selected.getCount()/64.); s[12] = food(selected);
        s[13] = unit(mc.player.containerMenu.getCarried().getCount()/64.);
        s[14] = (mc.player.getXRot()+90)/180.; s[15] = mc.player.horizontalCollision?1:0;
        Vec3 start = mc.player.getEyePosition();
        for (int i=0;i<9;i++) {
            Vec3 direction = Vec3.directionFromRotation(mc.player.getXRot()+(i/3-1)*30, mc.player.getYRot()+(i%3-1)*30);
            BlockHitResult hit = mc.level.clip(new ClipContext(start,start.add(direction.scale(8)),ClipContext.Block.COLLIDER,ClipContext.Fluid.ANY,mc.player));
            s[16+i] = hit.getType()==HitResult.Type.MISS?0:1-unit(start.distanceTo(hit.getLocation())/8);
        }
        // Object-category hashes are local observations, not language comprehension.
        String target = "air";
        if (mc.hitResult instanceof BlockHitResult hit && hit.getType()==HitResult.Type.BLOCK)
            target = BuiltInRegistries.BLOCK.getKey(mc.level.getBlockState(hit.getBlockPos()).getBlock()).toString();
        else if (mc.hitResult instanceof EntityHitResult hit)
            target = BuiltInRegistries.ENTITY_TYPE.getKey(hit.getEntity().getType()).toString();
        int hash = target.hashCode() ^ BuiltInRegistries.ITEM.getKey(selected.getItem()).toString().hashCode();
        for (int i=0;i<7;i++) s[25+i] = ((hash >>> (i*4)) & 15)/15.;
        JsonArray senses = new JsonArray(); for (double value:s) senses.add(unit(value)); o.add("senses",senses);
        return o;
    }
}
