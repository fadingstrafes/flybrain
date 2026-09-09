import sys
import navis

from navis.interfaces import neuprint

client = neuprint.Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
)

body_ids = [int(x) for x in sys.argv[1:]]

neurons = neuprint.fetch_skeletons(
    body_ids,
    client=client,
)

print(neurons)

fig = navis.plot3d(
    neurons,
    backend="plotly",
)

fig.show()
