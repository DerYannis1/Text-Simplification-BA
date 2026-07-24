from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Modell laden
model = SentenceTransformer("BAAI/bge-m3")  # Modell ggf. anpassen

# Zwei Artikel
article_1 = """
Priester oder Priesterin ist eine Bezeichnung für Spezialisten für religiöse Fragen, die den Kultus verwalten sowie Lehre und Tradition bewahren. Im Gegensatz zu bestimmten religiösen Charismatikern wie etwa Schamanen, Sehern oder Propheten erhalten sie eine Art von Ausbildung und Einsetzung in ihr Amt, dem die Mitglieder der betreffenden Religionsgemeinschaft Heiligkeit zuschreiben. Priestertum wird oft nach außen über Besonderheiten der Lebensweise und der Kleidung kenntlich gemacht.

"""

article_2 = """
"Priester kennt man heute vor allem aus der katholischen Kirche . Das griechische Wort „presbyteros“ bedeutet Ältester. In vielen Gemeinschaften bestimmten früher die ältesten Männer die Richtung, weil man ihnen am meisten Erfahrung und Geschick zutraute. Das Wort Priester braucht man aber in einem religiösen Zusammenhang.
Priester sind so etwas wie Vermittler zwischen Gott und den Menschen. Die heutigen Juden und die Moslems kennen keine solchen Vermittler. Der Imam im Islam ist kein Priester, sondern ein Gelehrter des Koran ."

"""

# Document Embeddings berechnen
embedding_1 = model.encode(article_1, normalize_embeddings=True)
embedding_2 = model.encode(article_2, normalize_embeddings=True)

# Cosinus-Similarity
similarity = cosine_similarity(
    embedding_1.reshape(1, -1),
    embedding_2.reshape(1, -1)
)[0][0]

print(f"Cosine Similarity: {similarity:.4f}")