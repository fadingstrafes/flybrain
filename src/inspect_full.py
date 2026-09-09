from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc

files = sorted(
    Path("data/raw").glob("*.feather")
)

for path in files:
    print()
    print("=" * 80)
    print(path.name)
    print("=" * 80)

    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)

        print("Record batches:", reader.num_record_batches)

        print("\nColumns:")
        for field in reader.schema:
            print(
                f"  {field.name:<35} "
                f"{field.type}"
            )
