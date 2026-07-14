from gliner2 import GLiNER2

extractor = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")

text = "En 1887, la Société suisse du Grutlis'est accrue de 40 sections; l'association compte actuellement 12,000 membres."

labels = ["person", "location", "organization", "time", "production"]

entities = extractor.extract_entities(text, labels, include_confidence=True, threshold=0.3)

print(entities)