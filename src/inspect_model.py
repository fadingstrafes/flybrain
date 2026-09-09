import trimesh

path = "assets/fly.glb"

scene = trimesh.load(path, force="scene")

print("=" * 80)
print("3D FLY MODEL")
print("=" * 80)

print(f"\nGeometry objects: {len(scene.geometry)}")

print("\nOBJECTS")
print("-" * 80)

for name, mesh in scene.geometry.items():

    print(f"\n{name}")

    print(f"  vertices:  {len(mesh.vertices):,}")
    print(f"  faces:     {len(mesh.faces):,}")

    material = getattr(
        mesh.visual,
        "material",
        None,
    )

    if material is not None:

        print(
            f"  material:  "
            f"{getattr(material, 'name', None)}"
        )

        image = getattr(
            material,
            "image",
            None,
        )

        print(
            f"  texture:   "
            f"{'YES' if image is not None else 'no'}"
        )


print("\nSCENE GRAPH")
print("-" * 80)

for node_name in scene.graph.nodes:
    print(node_name)


print("\nBOUNDS")
print("-" * 80)

print(scene.bounds)

print("\nSIZE")
print("-" * 80)

print(scene.extents)
