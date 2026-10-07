package org.flybrain;

import carpet.patches.EntityPlayerMPFake;
import carpet.patches.FakeClientConnection;
import carpet.fakes.ServerPlayerInterface;
import carpet.helpers.EntityPlayerActionPack;
import com.google.gson.*;
import com.mojang.authlib.GameProfile;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.minecraft.commands.Commands;
import net.minecraft.core.component.DataComponents;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.network.chat.Component;
import net.minecraft.network.protocol.PacketFlow;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.*;
import net.minecraft.server.network.CommonListenerCookie;
import net.minecraft.world.inventory.ContainerInput;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.*;
import net.minecraft.world.phys.*;
import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

public final class FlyBrainMod implements ModInitializer {
    public static final UUID ID = UUID.nameUUIDFromBytes("OfflinePlayer:FlyBrain".getBytes(java.nio.charset.StandardCharsets.UTF_8));
    public static boolean crawling, died;
    public static int placed, broken;
    private static EntityPlayerMPFake bot;
    private static FlyModel model;
    private static ServerLevel spawnWorld;
    private static Vec3 spawnPosition;
    static boolean running;
    private static boolean inventory;
    private static int ticks, until, life, sequence, slot;
    private static String session = UUID.randomUUID().toString(), action = "idle";
    private static final HttpClient HTTP = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(2)).build();
    private static CompletableFuture<HttpResponse<String>> pending;

    @Override public void onInitialize() {
        net.fabricmc.fabric.api.event.player.PlayerBlockBreakEvents.AFTER.register((world,player,pos,state,entity) -> {
            if(player.getUUID().equals(ID)) broken++;
        });
        CommandRegistrationCallback.EVENT.register((dispatcher,access,environment) -> dispatcher.register(
            Commands.literal("flybrain").requires(source -> source.getServer()!=null && source.getServer().isSingleplayer()
                && source.getEntity() instanceof ServerPlayer player && source.getServer().isSingleplayerOwner(player.nameAndId()))
                .then(Commands.literal("spawn").executes(ctx -> {
                    if(bot!=null) { ctx.getSource().sendSuccess(()->Component.literal("FlyBrain already exists"),false); return 0; }
                    ServerPlayer owner = ctx.getSource().getPlayerOrException();
                    spawnWorld=owner.level(); spawnPosition=owner.position().add(2,0,0);
                    session=UUID.randomUUID().toString(); sequence=life=0;
                    spawn(owner.level().getServer()); running=true;
                    Foraging.starter(bot);
                    int plants=Foraging.seedPatch(bot);
                    ctx.getSource().sendSuccess(()->Component.literal("FlyBrain spawned with 4 bread and "+plants+" ripe berry bushes nearby. F8 pause/resume, F9 follow camera."),false); return 1;
                }))
                .then(Commands.literal("forage").executes(ctx -> {
                    if(bot==null) return 0;
                    int plants=Foraging.seedPatch(bot);
                    ctx.getSource().sendSuccess(()->Component.literal("Added "+plants+" berry bushes on suitable empty ground."),false);return plants;
                }))
                .then(Commands.literal("toggle").executes(ctx -> {
                    if(bot==null) { ctx.getSource().sendSuccess(()->Component.literal("Use /flybrain spawn first"),false);return 0; }
                    running=!running; stopAction();
                    if(pending!=null) pending.cancel(true); pending=null;
                    session=UUID.randomUUID().toString(); sequence=0;
                    ctx.getSource().sendSuccess(()->Component.literal("FlyBrain "+(running?"running":"paused")),false);return 1;
                }))
                .then(Commands.literal("remove").executes(ctx -> { cleanup();return 1; }))));
        ServerTickEvents.END_SERVER_TICK.register(FlyBrainMod::tick);
        ServerLifecycleEvents.SERVER_STOPPING.register(server -> cleanup());
    }
    private static EntityPlayerActionPack pack() { return ((ServerPlayerInterface)bot).getActionPack(); }
    private static void spawn(MinecraftServer server) {
        if(model!=null) model.remove();
        GameProfile profile = new GameProfile(ID,"FlyBrain");
        bot=EntityPlayerMPFake.respawnFake(server,spawnWorld,profile,ClientInformation.createDefault());
        bot.fixStartingPosition=()->bot.snapTo(spawnPosition.x,spawnPosition.y,spawnPosition.z,0,0);
        server.getPlayerList().placeNewPlayer(new FakeClientConnection(PacketFlow.SERVERBOUND),bot,
            new CommonListenerCookie(profile,0,bot.clientInformation(),false));
        bot.teleportTo(spawnWorld,spawnPosition.x,spawnPosition.y,spawnPosition.z,java.util.Set.of(),0,0,true);
        bot.connection.handleAcceptPlayerLoad(new net.minecraft.network.protocol.game.ServerboundPlayerLoadedPacket());
        bot.gameMode.changeGameModeForPlayer(GameType.SURVIVAL);
        bot.setHealth(20); bot.getFoodData().setFoodLevel(20);
        bot.getAbilities().mayfly=true; bot.getAbilities().flying=false; bot.onUpdateAbilities();
        crawling=died=inventory=false; slot=0; until=ticks;
        placed=broken=0;
        model=new FlyModel(bot);
    }
    private static void stopAction() {
        if(bot!=null) { pack().stopAll(); bot.stopUsingItem(); }
        action="idle";
    }
    private static void cleanup() {
        running=false; stopAction();
        if(pending!=null) pending.cancel(true); pending=null;
        if(model!=null) model.remove(); model=null;
        if(bot!=null) bot.kill(Component.literal("FlyBrain removed")); bot=null;
        spawnWorld=null; crawling=died=false;
    }
    private static void tick(MinecraftServer server) {
        ticks++;
        if(bot==null) return;
        if(model!=null && !model.isIn(bot.level())) { model.remove(); model=new FlyModel(bot); }
        if(model!=null) model.update(bot,ticks,crawling);
        if(!running) return;
        if(bot.isRemoved() && !died) { running=false;stopAction();return; }
        if(died) stopAction();
        if(pending!=null && pending.isDone()) {
            try {
                HttpResponse<String> result=pending.join(); pending=null;
                if(result.statusCode()!=200) throw new IllegalStateException("brain HTTP "+result.statusCode());
                JsonObject reply=JsonParser.parseString(result.body()).getAsJsonObject();
                if(!reply.get("session").getAsString().equals(session) || reply.get("seq").getAsInt()!=sequence-1
                        || reply.get("life").getAsInt()!=life) throw new IllegalStateException("stale response");
                if(died && reply.get("respawn").getAsBoolean()) {
                    if(server.getPlayerList().getPlayer(ID)!=null) server.getPlayerList().remove(bot);
                    life++; spawn(server);
                } else if(!died) apply(reply.get("action").getAsString());
            } catch(Exception error) {
                running=false; stopAction();
                server.getPlayerList().broadcastSystemMessage(Component.literal("FlyBrain connection stopped: "+error.getMessage()+". Start brain service, then F8."),false);
                return;
            }
        }
        if(!died && bot.getAbilities().flying) {
            double vertical=action.equals("ascend")?.15:action.equals("descend")?-.15:0;
            Vec3 velocity=bot.getDeltaMovement(); bot.setDeltaMovement(velocity.x,vertical,velocity.z);
            bot.causeFoodExhaustion(.03f); // Artificial flight energy cost, not fly physiology.
        }
        if(ticks>=until && !action.equals("idle")) stopAction();
        if(pending==null && (died || ticks>=until)) {
            HttpRequest request=HttpRequest.newBuilder(URI.create("http://127.0.0.1:8765/step"))
                .timeout(Duration.ofSeconds(5)).header("Content-Type","application/json")
                .POST(HttpRequest.BodyPublishers.ofString(observe().toString())).build();
            pending=HTTP.sendAsync(request,HttpResponse.BodyHandlers.ofString());
        }
    }
    static void apply(String next) {
        stopAction(); action=next; until=ticks+((next.equals("attack")||next.equals("use"))?40:10);
        if(next.equals("takeoff")) { crawling=false;bot.getAbilities().flying=true;bot.onUpdateAbilities(); }
        if(next.equals("land")) { bot.getAbilities().flying=false;bot.onUpdateAbilities(); }
        if(next.equals("crawl")) { bot.getAbilities().flying=false;bot.onUpdateAbilities();crawling=true; }
        if(next.equals("stand")) crawling=false;
        pack().setSneaking(crawling||next.equals("sneak"));
        if(next.equals("inventory")) {
            if(bot.containerMenu!=bot.inventoryMenu) { bot.closeContainer(); inventory=false; }
            else inventory=!inventory;
        }
        int size=bot.containerMenu.slots.size(); slot=Math.floorMod(slot,size);
        if(next.equals("slot_next")) slot=(slot+1)%size;
        if(next.equals("slot_previous")) slot=(slot+size-1)%size;
        if(next.startsWith("hotbar_")) bot.getInventory().setSelectedSlot(Integer.parseInt(next.substring(7)));
        if(next.equals("slot_click")||next.equals("slot_right_click")||next.equals("slot_quick_move")) {
            if(inventory||bot.containerMenu!=bot.inventoryMenu) {
                bot.containerMenu.clicked(slot,next.equals("slot_right_click")?1:0,
                    next.equals("slot_quick_move")?ContainerInput.QUICK_MOVE:ContainerInput.PICKUP,bot);
                bot.containerMenu.broadcastChanges();
            }
        }
        if(next.equals("look_left")) pack().turn(-15,0);
        if(next.equals("look_right")) pack().turn(15,0);
        if(next.equals("look_up")) pack().turn(0,-15);
        if(next.equals("look_down")) pack().turn(0,15);
        // Movement exits the logical inventory instead of silently doing nothing.
        if(java.util.Set.of("forward","back","left","right","jump","forward_jump").contains(next)) {
            if(bot.containerMenu!=bot.inventoryMenu) bot.closeContainer();
            inventory=false;
        }
        if(inventory||bot.containerMenu!=bot.inventoryMenu) return;
        if(next.equals("forward")||next.equals("forward_jump")) pack().setForward(1);
        if(next.equals("back")) pack().setForward(-1);
        if(next.equals("left")) pack().setStrafing(1);
        if(next.equals("right")) pack().setStrafing(-1);
        if(next.equals("jump")||next.equals("forward_jump")) pack().start(EntityPlayerActionPack.ActionType.JUMP,EntityPlayerActionPack.Action.continuous());
        if(next.equals("attack")) pack().start(EntityPlayerActionPack.ActionType.ATTACK,EntityPlayerActionPack.Action.continuous());
        if(next.equals("use")) pack().start(EntityPlayerActionPack.ActionType.USE,EntityPlayerActionPack.Action.continuous());
    }
    private static double unit(double x) {return Math.max(0,Math.min(1,x));}
    private static double food(ItemStack stack) {return stack.get(DataComponents.FOOD)==null?0:1;}
    static JsonObject observe() {
        JsonObject o=new JsonObject();
        o.addProperty("version",1);o.addProperty("session",session);o.addProperty("life",life);o.addProperty("seq",sequence++);
        o.addProperty("dead",died);o.addProperty("health",died?0:Math.max(0,bot.getHealth()));o.addProperty("food",bot.getFoodData().getFoodLevel());
        double[] s=new double[32];
        s[0]=1;s[1]=1-unit(bot.getHealth()/20);s[2]=1-bot.getFoodData().getFoodLevel()/20.;
        s[3]=bot.onGround()?1:0;s[4]=bot.isInWater()?1:0;s[5]=bot.isOnFire()?1:0;s[6]=unit(bot.getDeltaMovement().length());
        s[7]=food(bot.getMainHandItem());s[8]=bot.getInventory().getSelectedSlot()/8.;
        s[9]=(inventory||bot.containerMenu!=bot.inventoryMenu)?1:0;
        slot=Math.floorMod(slot,bot.containerMenu.slots.size());
        s[10]=slot/(double)Math.max(1,bot.containerMenu.slots.size()-1);
        ItemStack selected=bot.containerMenu.slots.get(slot).getItem();
        s[11]=unit(selected.getCount()/64.);s[12]=food(selected);s[13]=unit(bot.containerMenu.getCarried().getCount()/64.);
        s[14]=(bot.getXRot()+90)/180.;s[15]=bot.horizontalCollision?1:0;
        Vec3 start=bot.getEyePosition(); String target="air";
        for(int i=0;i<9;i++) {
            Vec3 direction=Vec3.directionFromRotation(bot.getXRot()+(i/3-1)*30,bot.getYRot()+(i%3-1)*30);
            BlockHitResult hit=bot.level().clip(new ClipContext(start,start.add(direction.scale(8)),ClipContext.Block.COLLIDER,ClipContext.Fluid.ANY,bot));
            s[16+i]=hit.getType()==HitResult.Type.MISS?0:1-unit(start.distanceTo(hit.getLocation())/8);
            if(i==4 && hit.getType()!=HitResult.Type.MISS) target=BuiltInRegistries.BLOCK.getKey(bot.level().getBlockState(hit.getBlockPos()).getBlock()).toString();
        }
        int hash=target.hashCode()^BuiltInRegistries.ITEM.getKey(selected.getItem()).toString().hashCode();
        for(int i=0;i<5;i++) s[25+i]=((hash>>>(i*4))&15)/15.;
        s[30]=bot.getAbilities().flying?1:0;s[31]=crawling?1:0;
        JsonArray senses=new JsonArray();for(double value:s)senses.add(unit(value));o.add("senses",senses);
        JsonObject world=new JsonObject();
        int foodStock=0,materials=0;
        for(int i=0;i<bot.getInventory().getContainerSize();i++) {
            ItemStack stack=bot.getInventory().getItem(i);
            if(food(stack)>0) foodStock+=stack.getCount();
            if(stack.getItem() instanceof net.minecraft.world.item.BlockItem) materials+=stack.getCount();
        }
        double threat=0;
        for(var entity:bot.level().getEntitiesOfClass(net.minecraft.world.entity.LivingEntity.class,
                bot.getBoundingBox().inflate(12),e -> e instanceof net.minecraft.world.entity.monster.Enemy && e.isAlive())) {
            if(bot.distanceTo(entity)<=12 && bot.hasLineOfSight(entity)) threat=Math.max(threat,1-bot.distanceTo(entity)/12.);
        }
        BlockHitResult roof=bot.level().clip(new ClipContext(start,start.add(0,6,0),ClipContext.Block.COLLIDER,ClipContext.Fluid.NONE,bot));
        int walls=0;
        for(Vec3 direction:new Vec3[]{new Vec3(3,0,0),new Vec3(-3,0,0),new Vec3(0,0,3),new Vec3(0,0,-3)})
            if(bot.level().clip(new ClipContext(start,start.add(direction),ClipContext.Block.COLLIDER,ClipContext.Fluid.NONE,bot)).getType()!=HitResult.Type.MISS) walls++;
        double cover=roof.getType()==HitResult.Type.MISS?0:.5+.5*walls/4.;
        world.addProperty("food_stock",foodStock);world.addProperty("materials",materials);
        world.addProperty("threat",threat);world.addProperty("cover",cover);
        world.addProperty("danger",Math.max(threat,bot.level().getSkyDarken()>5||bot.level().isRaining()?.6:0));
        Foraging.sense(bot,world,walls,roof.getType()!=HitResult.Type.MISS,threat);
        world.addProperty("placed",placed);world.addProperty("broken",broken);world.addProperty("ticks",ticks);
        JsonArray position=new JsonArray();position.add(bot.getX());position.add(bot.getY());position.add(bot.getZ());world.add("position",position);
        JsonArray hotbar=new JsonArray();for(int i=0;i<9;i++) hotbar.add(bot.getInventory().getItem(i).isEmpty()?0:1);world.add("hotbar",hotbar);
        o.add("world",world);
        JsonObject vision=EyeFrames.snapshot(bot.getId());
        if(vision!=null) o.add("vision",vision);
        return o;
    }
}
