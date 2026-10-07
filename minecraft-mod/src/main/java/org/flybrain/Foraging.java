package org.flybrain;

import com.google.gson.JsonObject;
import net.minecraft.core.BlockPos;
import net.minecraft.core.component.DataComponents;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.level.LightLayer;
import net.minecraft.world.level.block.*;
import net.minecraft.world.phys.*;

/** Explicit food categories and local safety cues; no navigation or building plan. */
public final class Foraging {
    public static int seedPatch(ServerPlayer player) {
        int planted=0;
        var world=player.level();var origin=player.blockPosition();
        for(int[] offset:new int[][]{{4,0},{-4,0},{0,4},{0,-4},{4,4},{-4,4},{4,-4},{-4,-4}}) {
            for(int dy=2;dy>=-3;dy--) {
                BlockPos pos=origin.offset(offset[0],dy,offset[1]);
                var ground=world.getBlockState(pos.below());
                if(world.getBlockState(pos).isAir() && (ground.is(Blocks.GRASS_BLOCK)||ground.is(Blocks.DIRT)||ground.is(Blocks.COARSE_DIRT))) {
                    world.setBlockAndUpdate(pos,Blocks.SWEET_BERRY_BUSH.defaultBlockState().setValue(SweetBerryBushBlock.AGE,3));
                    planted++;break;
                }
            }
        }
        return planted;
    }
    public static void starter(ServerPlayer player) {
        player.getInventory().add(new ItemStack(Items.BREAD,4));
    }
    public static boolean harvestable(net.minecraft.world.level.block.state.BlockState state) {
        return (state.is(Blocks.SWEET_BERRY_BUSH)&&state.getValue(SweetBerryBushBlock.AGE)>=2)
            || (state.getBlock() instanceof CropBlock crop && crop.isMaxAge(state)
                && (state.is(Blocks.CARROTS)||state.is(Blocks.POTATOES)||state.is(Blocks.BEETROOTS)))
            || state.is(Blocks.MELON);
    }
    private static boolean visible(ServerPlayer player,Vec3 target,BlockPos block) {
        Vec3 delta=target.subtract(player.getEyePosition());
        if(delta.length()>8 || delta.normalize().dot(player.getLookAngle())<-.45) return false;
        var hit=player.level().clip(new ClipContext(player.getEyePosition(),target,ClipContext.Block.COLLIDER,ClipContext.Fluid.NONE,player));
        return hit.getType()==HitResult.Type.MISS || (block!=null && hit.getBlockPos().equals(block));
    }
    public static void sense(ServerPlayer player,JsonObject world,int walls,boolean roof,double threat) {
        Vec3 nearest=null;double distance=9;
        for(var item:player.level().getEntitiesOfClass(ItemEntity.class,player.getBoundingBox().inflate(8),e -> e.getItem().get(DataComponents.FOOD)!=null)) {
            double d=player.getEyePosition().distanceTo(item.position());
            if(d<distance && visible(player,item.position(),null)) {distance=d;nearest=item.position();}
        }
        BlockPos origin=player.blockPosition();
        for(BlockPos pos:BlockPos.betweenClosed(origin.offset(-7,-3,-7),origin.offset(7,3,7))) {
            Vec3 center=Vec3.atCenterOf(pos);
            double d=player.getEyePosition().distanceTo(center);
            if(d<distance && harvestable(player.level().getBlockState(pos)) && visible(player,center,pos)) {distance=d;nearest=center;}
        }
        double left=0,right=0;
        if(nearest!=null) {
            Vec3 direction=nearest.subtract(player.position()).normalize();
            Vec3 sideways=Vec3.directionFromRotation(0,player.getYRot()+90);
            double bearing=direction.dot(sideways);
            left=Math.max(0,-bearing);right=Math.max(0,bearing);
        }
        boolean exit=false,hazard=player.isOnFire()||player.isUnderWater();
        for(BlockPos p:BlockPos.betweenClosed(origin.offset(-1,-1,-1),origin.offset(1,1,1))) {
            var state=player.level().getBlockState(p);
            hazard|=state.is(Blocks.LAVA)||state.is(Blocks.FIRE)||state.is(Blocks.SOUL_FIRE)||state.is(Blocks.MAGMA_BLOCK);
        }
        for(int[] dir:new int[][]{{1,0},{-1,0},{0,1},{0,-1}}) {
            BlockPos p=origin.offset(dir[0],0,dir[1]);
            if(player.level().getBlockState(p).getCollisionShape(player.level(),p).isEmpty()
                && player.level().getBlockState(p.above()).getCollisionShape(player.level(),p.above()).isEmpty()
                && !player.level().getBlockState(p.below()).getCollisionShape(player.level(),p.below()).isEmpty()) exit=true;
        }
        double light=player.level().getBrightness(LightLayer.BLOCK,origin)/15.;
        exit &= walls<4;
        double quality=roof?Math.max(0,Math.min(1,.35+.1*Math.min(walls,3)+(exit?.15:0)+.2*light-(hazard?.5:0)-.4*threat)):0;
        world.addProperty("cover",quality);
        world.addProperty("food_visible",nearest==null?0:Math.max(0,1-distance/8));
        world.addProperty("food_left",left);world.addProperty("food_right",right);
        world.addProperty("shelter_exit",exit?1:0);world.addProperty("shelter_light",light);world.addProperty("shelter_hazard",hazard?1:0);
    }
}
