package org.flybrain;

import com.mojang.math.Transformation;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.Display;
import org.flybrain.mixin.*;
import org.joml.Vector3f;
import org.joml.Quaternionf;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import java.util.ArrayList;
import java.util.List;

/** Stylized segmented fly. Animation is cosmetic, not physiological biomechanics. */
public final class FlyModel {
    public static final java.util.Set<java.util.UUID> VISUAL_IDS=java.util.concurrent.ConcurrentHashMap.newKeySet();
    private record Part(Display.BlockDisplay entity, Vector3f center, Vector3f size,
                        Quaternionf rotation, int wing, int leg) {}
    private final List<Part> parts = new ArrayList<>();
    private final ServerLevel world;
    private float gait;

    public FlyModel(ServerPlayer player) {
        world=player.level();
        // Layered silhouettes soften the rectangular thorax, head and abdomen.
        box("gray_concrete",0,0,0,.43f,.47f,.40f);
        box("black_concrete",0,.02f,-.02f,.33f,.56f,.44f);
        box("gray_concrete",0,-.35f,-.07f,.43f,.26f,.40f);
        box("black_concrete",0,-.49f,-.10f,.37f,.12f,.35f);
        box("gray_concrete",0,-.60f,-.11f,.29f,.16f,.29f);
        box("black_concrete",0,-.72f,-.11f,.17f,.10f,.19f);
        box("black_concrete",0,.43f,.05f,.39f,.30f,.32f);
        box("gray_concrete",0,.58f,.05f,.26f,.08f,.23f);
        box("black_concrete",0,.35f,.23f,.09f,.16f,.10f);
        for(int side:new int[]{-1,1}) {
            box("red_concrete",side*.20f,.46f,.13f,.18f,.27f,.24f);
            box("red_terracotta",side*.25f,.45f,.18f,.10f,.18f,.18f);
            box("orange_terracotta",side*.19f,.53f,.255f,.065f,.065f,.025f);
            limb(new Vector3f(side*.09f,.57f,.12f),new Vector3f(side*.17f,.75f,.18f),.028f,-1);
            box("black_concrete",side*.17f,.76f,.18f,.055f,.055f,.055f);
            // Wings lie along the back when standing and horizontally in flight.
            Quaternionf wingAngle=new Quaternionf().rotateZ(side*.40f);
            part("light_gray_stained_glass",new Vector3f(side*.43f,-.18f,-.29f),
                new Vector3f(.40f,.87f,.025f),wingAngle,side,-1);
            part("white_stained_glass",new Vector3f(side*.54f,-.51f,-.29f),
                new Vector3f(.28f,.23f,.025f),wingAngle,side,-1);
            part("gray_concrete",new Vector3f(side*.43f,-.18f,-.31f),
                new Vector3f(.018f,.78f,.018f),wingAngle,side,-1);
            for(int leg=0;leg<3;leg++) {
                float rootY=.15f-leg*.22f;
                Vector3f root=new Vector3f(side*.19f,rootY,.08f);
                Vector3f knee=new Vector3f(side*(.47f-leg*.035f),rootY-.16f,.17f+(1-leg)*.12f);
                Vector3f foot=new Vector3f(side*(.61f-leg*.06f),rootY-.39f,.30f+(1-leg)*.20f);
                int group=leg+(side>0?3:0);
                limb(root,knee,.043f,group);
                limb(knee,foot,.032f,group);
            }
        }
        update(player,0,false);
    }
    public boolean isIn(ServerLevel level) { return world==level; }
    private void box(String material,float x,float y,float z,float sx,float sy,float sz) {
        part(material,new Vector3f(x,y,z),new Vector3f(sx,sy,sz),new Quaternionf(),0,-1);
    }
    private void limb(Vector3f start,Vector3f end,float width,int leg) {
        Vector3f delta=new Vector3f(end).sub(start);
        part("black_concrete",new Vector3f(start).add(end).mul(.5f),
            new Vector3f(width,delta.length(),width),new Quaternionf().rotationTo(new Vector3f(0,1,0),delta.normalize()),0,leg);
    }
    private void part(String material,Vector3f center,Vector3f size,Quaternionf rotation,int wing,int leg) {
        Display.BlockDisplay entity=new Display.BlockDisplay(BuiltInRegistries.ENTITY_TYPE.getValue(Identifier.withDefaultNamespace("block_display")),world);
        ((BlockDisplayAccess)entity).flybrain$block(BuiltInRegistries.BLOCK.getValue(Identifier.withDefaultNamespace(material)).defaultBlockState());
        ((DisplayAccess)entity).flybrain$interpolate(1);
        entity.addTag("flybrain_visual");
        VISUAL_IDS.add(entity.getUUID());
        parts.add(new Part(entity,center,size,rotation,wing,leg));
        // Place before spawning: avoids briefly drawing pieces at the world origin.
    }
    public void update(ServerPlayer player,int ticks,boolean crawl) {
        boolean flight=player.getAbilities().flying;
        float speed=(float)player.getDeltaMovement().horizontalDistance();
        gait+=Math.min(speed*5,.65f);
        Quaternionf body=new Quaternionf().rotateY((float)Math.toRadians(-player.getYRot()));
        if(crawl||flight) body.rotateX((float)Math.PI/2);
        for(Part part:parts) {
            Vector3f position=new Vector3f(part.center);
            Quaternionf animation=new Quaternionf();
            if(part.wing!=0) {
                float angle=part.wing*(flight?(float)Math.sin(ticks*2.1)*.70f:.07f);
                animation.rotateY(angle);
                Vector3f hinge=new Vector3f(part.wing*.18f,.12f,-.23f);
                animation.transform(position.sub(hinge)).add(hinge);
            }
            if(part.leg>=0) {
                float phase=gait+(part.leg%2==0?0:(float)Math.PI);
                float swing=flight?.30f:(float)Math.sin(phase)*Math.min(speed*2,.16f);
                position.z+=swing;
                if(flight) position.x*=.8f;
            }
            body.transform(position).add(0,crawl?.43f:flight?.85f:.95f,0);
            Quaternionf orientation=new Quaternionf(body).mul(animation).mul(part.rotation);
            Vector3f half=orientation.transform(new Vector3f(part.size).mul(.5f));
            ((DisplayAccess)part.entity).flybrain$transform(new Transformation(position.sub(half),orientation,part.size,new Quaternionf()));
            part.entity.setPos(player.getX(),player.getY(),player.getZ());
            if(ticks==0) world.addFreshEntity(part.entity);
        }
    }
    public void remove() { for(Part p:parts) {VISUAL_IDS.remove(p.entity.getUUID());p.entity.discard();} parts.clear(); }
}
