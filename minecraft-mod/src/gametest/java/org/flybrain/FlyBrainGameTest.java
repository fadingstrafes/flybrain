package org.flybrain;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.fabricmc.fabric.api.client.gametest.v1.context.TestSingleplayerContext;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.entity.Pose;

public final class FlyBrainGameTest implements FabricClientGameTest {
    private static void check(boolean condition,String message) { if(!condition) throw new AssertionError(message); }
    @Override public void runTest(ClientGameTestContext context) {
        try(TestSingleplayerContext world=context.worldBuilder().create()) {
            context.runOnClient(mc -> mc.getConnection().sendCommand("flybrain spawn"));
            world.getServer().waitFor(server -> server.getPlayerList().getPlayer(FlyBrainMod.ID)!=null);
            world.getServer().runOnServer(server -> {
                FlyBrainMod.running=false;
                var bot=server.getPlayerList().getPlayer(FlyBrainMod.ID);
                check(bot.getInventory().getItem(0).is(Items.BREAD),"Starter bread must be available");
                check(bot.getInventory().getItem(0).getCount()==4,"Starter kit is four bread");
                check(bot.level().getBlockState(bot.blockPosition().offset(4,0,0)).is(net.minecraft.world.level.block.Blocks.SWEET_BERRY_BUSH),
                    "Spawn should add renewable foraging plants on suitable ground");
                bot.getFoodData().setFoodLevel(10);
                FlyBrainMod.apply("use");
            });
            context.waitTicks(40);
            world.getServer().runOnServer(server -> {
                FlyBrainMod.running=false;
                var bot=server.getPlayerList().getPlayer(FlyBrainMod.ID);
                FlyBrainMod.apply("idle");
                check(bot.getFoodData().getFoodLevel()>10,"Using held food must restore hunger");
                check(bot.getInventory().getItem(0).getCount()<4,"Eating must consume starter food");
                check(server.getPlayerList().getPlayerCount()==2,"Agent must be independent of observer");
                check(!bot.isCreative() && bot.getHealth()==20,"Agent needs survival health");
                bot.getInventory().setItem(9,new ItemStack(Items.OAK_LOG));
                FlyBrainMod.apply("inventory");
                for(int i=0;i<9;i++) FlyBrainMod.apply("slot_next");
                FlyBrainMod.apply("slot_click");
                for(int i=0;i<8;i++) FlyBrainMod.apply("slot_previous");
                FlyBrainMod.apply("slot_click");
                FlyBrainMod.apply("slot_previous");
                FlyBrainMod.apply("slot_click");
                check(bot.containerMenu.getCarried().is(Items.OAK_PLANKS),"Primitive slots must permit crafting");
                check(bot.containerMenu.getCarried().getCount()==4,"Crafting must return four planks");
                FlyBrainMod.apply("inventory");
                FlyBrainMod.apply("crawl");
            });
            context.waitTicks(5);
            world.getServer().runOnServer(server -> {
                var bot=server.getPlayerList().getPlayer(FlyBrainMod.ID);
                check(bot.getPose()==Pose.SWIMMING,"Crawling pose must persist");
                FlyBrainMod.apply("stand");
                FlyBrainMod.apply("takeoff");
                check(bot.getAbilities().flying,"Takeoff must enable flight");
                FlyBrainMod.apply("land");
                check(!bot.getAbilities().flying,"Landing must disable flight");
                float health=bot.getHealth();
                bot.hurtServer(bot.level(),bot.damageSources().generic(),2);
                check(bot.getHealth()<health,"Agent must take actual damage");
                bot.getInventory().setItem(0,new ItemStack(Items.DIRT,1));
                FlyBrainMod.apply("hotbar_0");
                bot.setYRot(0); bot.setXRot(45);
                FlyBrainMod.apply("use");
            });
            context.waitTicks(5);
            world.getServer().runOnServer(server -> {
                var bot=server.getPlayerList().getPlayer(FlyBrainMod.ID);
                FlyBrainMod.apply("idle");
                check(bot.level().getBlockState(bot.blockPosition().offset(0,0,2)).is(net.minecraft.world.level.block.Blocks.DIRT),
                    "Use must place a block in the world");
                check(bot.getMainHandItem().isEmpty(),"Building must consume the agent's own inventory");
                check(FlyBrainMod.placed==1,"Placement must reach game-event telemetry");
                bot.setXRot(30);
                FlyBrainMod.apply("attack");
            });
            context.waitTicks(35);
            world.getServer().runOnServer(server -> {
                var bot=server.getPlayerList().getPlayer(FlyBrainMod.ID);
                FlyBrainMod.apply("idle");
                check(bot.level().getBlockState(bot.blockPosition().offset(0,0,2)).isAir(),
                    "Attack must break the placed block in survival");
                check(FlyBrainMod.broken>=1,"Breaking must reach game-event telemetry");
                FlyBrainMod.apply("inventory");
                FlyBrainMod.apply("forward");
                FlyBrainMod.apply("idle");
                check(FlyBrainMod.observe().getAsJsonArray("senses").get(9).getAsDouble()==0,
                    "Movement must exit inventory instead of being silently blocked");
                var roof=bot.blockPosition().above(3);
                bot.level().setBlockAndUpdate(roof,net.minecraft.world.level.block.Blocks.STONE.defaultBlockState());
                check(FlyBrainMod.observe().getAsJsonObject("world").get("cover").getAsDouble()>=.5,
                    "Local roof must provide a cover cue");
                bot.level().removeBlock(roof,false);
                var berries=bot.blockPosition().offset(0,0,2);
                bot.level().setBlockAndUpdate(berries,net.minecraft.world.level.block.Blocks.SWEET_BERRY_BUSH.defaultBlockState()
                    .setValue(net.minecraft.world.level.block.SweetBerryBushBlock.AGE,3));
                check(FlyBrainMod.observe().getAsJsonObject("world").get("food_visible").getAsDouble()>0,
                    "Ripe visible berries must be identified as forageable food");
                bot.setXRot(30);
                FlyBrainMod.apply("use");
            });
            context.waitTicks(5);
            world.getServer().runOnServer(server -> {
                var bot=server.getPlayerList().getPlayer(FlyBrainMod.ID);
                FlyBrainMod.apply("idle");
                check(bot.level().getBlockState(bot.blockPosition().offset(0,0,2))
                    .getValue(net.minecraft.world.level.block.SweetBerryBushBlock.AGE)==1,
                    "Using a ripe bush must harvest it and leave a regrowing plant");
                bot.setXRot(0);
            });
            context.runOnClient(mc -> {
                var bot=mc.level.getPlayerByUUID(FlyBrainMod.ID);
                check(bot!=null,"Agent must be visible to observer client");
                // Simulate a server metadata update restoring normal visibility.
                bot.setInvisible(false);
                check(!mc.getEntityRenderDispatcher().shouldRender(bot,null,0,0,0,0),
                    "Human avatar must stay hidden even when visibility is reset");
                mc.player.setYRot(bot.getYRot());
                mc.setCameraEntity(bot);
                mc.options.setCameraType(net.minecraft.client.CameraType.THIRD_PERSON_FRONT);
            });
            context.waitTicks(20);
            context.runOnClient(mc -> {
                check(FlyEyes.error==null,"Eye renderer failed: "+FlyEyes.error);
                var frame=EyeFrames.snapshot(mc.level.getPlayerByUUID(FlyBrainMod.ID).getId());
                check(frame!=null,"Actual eye images must be captured independently of observer");
                byte[] rgb=java.util.Base64.getDecoder().decode(frame.get("rgb").getAsString());
                check(rgb.length==2*96*64*3,"Both eyes must contain RGB pixels");
                var image=new java.awt.image.BufferedImage(192,64,java.awt.image.BufferedImage.TYPE_INT_RGB);
                int low=255,high=0;
                for(int eye=0;eye<2;eye++) for(int y=0;y<64;y++) for(int x=0;x<96;x++) {
                    int i=(eye*96*64+y*96+x)*3;
                    int r=rgb[i]&255,g=rgb[i+1]&255,b=rgb[i+2]&255;
                    low=Math.min(low,r);high=Math.max(high,r);
                    image.setRGB(eye*96+x,y,(r<<16)|(g<<8)|b);
                }
                check(high-low>20,"Eye images must contain world detail, not a blank target");
                check((rgb[(5*96+48)*3+2]&255)>40,"Daytime eye view must include rendered sky");
                try {
                    java.nio.file.Files.createDirectories(java.nio.file.Path.of("screenshots"));
                    javax.imageio.ImageIO.write(image,"png",new java.io.File("screenshots/flybrain-eyes.png"));
                } catch(java.io.IOException error) {throw new RuntimeException(error);}
            });
            context.takeScreenshot("flybrain-independent-agent");
            world.getServer().runOnServer(server -> FlyBrainMod.apply("crawl"));
            context.waitTicks(10);
            context.takeScreenshot("flybrain-crawling");
            world.getServer().runOnServer(server -> FlyBrainMod.apply("takeoff"));
            context.waitTicks(10);
            context.takeScreenshot("flybrain-flight");
            world.getServer().runOnServer(server -> {
                var bot=server.getPlayerList().getPlayer(FlyBrainMod.ID);
                bot.die(bot.damageSources().generic());
                check(FlyBrainMod.died,"Death must reach the learning bridge before Carpet resets health");
            });
            context.runOnClient(mc -> mc.setCameraEntity(mc.player));
        }
    }
}
