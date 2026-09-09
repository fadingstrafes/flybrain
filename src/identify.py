import os
from neuprint import Client, NeuronCriteria, fetch_neurons

client = Client(
    "neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=os.environ["NEUPRINT_APPLICATION_CREDENTIALS"],
)

body_ids = [
    522822,
    15884,
    513052,
    10141,
    162471,
    25047,
    805181,
    800721,
    22728,
    800632,
    809623,
    802194,
    808666,
    802700,
    803722,
    23566,
    920112,
    809413,
    904849,
]

criteria = NeuronCriteria(bodyId=body_ids)

neurons, roi_info = fetch_neurons(criteria)

columns = [
    "bodyId",
    "type",
    "instance",
    "pre",
    "post",
]

print(
    neurons[columns]
    .sort_values("bodyId")
    .to_string(index=False)
)
