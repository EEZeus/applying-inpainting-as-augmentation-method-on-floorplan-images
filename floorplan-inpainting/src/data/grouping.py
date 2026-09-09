"""Stable project/building grouping for the audited 60-image dataset."""

PROJECTS = {
    "P01": {"name": "Alstria 2015 Haus1+Haus2", "image_ids": [1, 2, 3, 4, 5]},
    "P02": {"name": "Feuerwehrplan 2020", "image_ids": list(range(6, 16))},
    "P03": {"name": "Alstria building 2007", "image_ids": list(range(16, 27))},
    "P04": {"name": "Feuerwehrpläne 2021-03", "image_ids": [27, 28, 29]},
    "P05": {"name": "Alstria Haus1 Laufkarten 2007", "image_ids": [30, 31, 43, 44]},
    "P06": {"name": "Hamburg 2005", "image_ids": [32, 33]},
    "P07": {"name": "Taunusquartier 2021", "image_ids": list(range(34, 42))},
    "P08": {"name": "Übersichtsplan 2024", "image_ids": [42]},
    "P10": {"name": "FWLK 36", "image_ids": [45]},
    "P11": {"name": "Feuerwehrlaufkarten 10109-10114", "image_ids": [46, 47, 48]},
    "P12": {"name": "Objekt 1415 Momentum Düsseldorf 2018", "image_ids": list(range(49, 60))},
    "P13": {"name": "Wilhelminenstraße Darmstadt", "image_ids": [60]},
}


def image_to_project():
    result = {}
    for project, details in PROJECTS.items():
        for image_id in details["image_ids"]:
            if image_id in result:
                raise ValueError(f"Image {image_id} assigned to multiple projects")
            result[image_id] = project
    return result


def validate_groups(image_ids):
    mapping = image_to_project()
    image_ids = set(image_ids)
    missing = sorted(image_ids - mapping.keys())
    extra = sorted(mapping.keys() - image_ids)
    if missing or extra:
        raise ValueError(f"Project map mismatch: missing={missing}, extra={extra}")

