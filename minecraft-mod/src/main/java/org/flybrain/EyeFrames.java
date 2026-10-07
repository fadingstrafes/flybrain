package org.flybrain;
import com.google.gson.JsonObject;
import java.util.Base64;
/** Immutable image handoff from render thread to integrated server. */
public final class EyeFrames {
    private record Frame(int entity,long time,long sequence,String rgb) {}
    private static volatile Frame latest;
    public static void publish(int entity,long sequence,byte[] rgb) {
        latest=new Frame(entity,System.nanoTime(),sequence,Base64.getEncoder().encodeToString(rgb));
    }
    public static void clear() {latest=null;}
    public static JsonObject snapshot(int entity) {
        Frame f=latest;
        if(f==null || f.entity!=entity || System.nanoTime()-f.time>2_000_000_000L) return null;
        JsonObject o=new JsonObject();o.addProperty("width",96);o.addProperty("height",64);
        o.addProperty("frame",f.sequence);o.addProperty("rgb",f.rgb);
        o.addProperty("age_ms",(System.nanoTime()-f.time)/1_000_000);
        return o;
    }
}
