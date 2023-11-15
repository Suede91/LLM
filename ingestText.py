from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma

VECTORSTORE_CHROMADB_PATH = 'vectorstore/chromadb'
transformer_model = "all-MiniLM-L12-v2"

model_embeddings = SentenceTransformerEmbeddings(model_name=transformer_model)
vectorstore = Chroma(persist_directory=VECTORSTORE_CHROMADB_PATH, embedding_function=model_embeddings)
vectorstore.add_texts([
    # Quelles sont les dates d'application du règlement mutualiste Unéo-Référence Unéo-International mentionnées dans le document?
    "Le numéro SIREN de la mutuelle Unéo est 503 380 081. (Pages: 1,2)",
    # Quel est le numéro SIREN de la mutuelle Unéo mentionné dans le document?
    "Le règlement mutualiste Unéo-Référence Unéo-International est applicable à compter du 7 et 8 juin 2023. (Pages: 1,2)",
    # Pouvez-vous citer deux articles qui traitent des conditions d'adhésion à la mutuelle Unéo?
    "Les articles qui traitent des conditions d'adhésion à la mutuelle Unéo sont l'Article M. 1 – Adhésion à la mutuelle Unéo et l'Article M. 7 – Adhésions particulières. (Pages: 1,2 et 10)",
    # Quelles sont les différentes garanties santé de base proposées par Unéo-Référence?
    "Les garanties santé de base Unéo-Référence proposées sont : Utile, Naturelle, Essentielle, et Optimale. (Page 5)",
    # Quelles sont les différentes garanties santé de base proposées par Unéo-International?
    "Les garanties santé de base Unéo-International proposées sont : Unéo Monde-Initiale, Unéo Monde-Globale, Unéo Monde-Intégrale, et Optimonde 2*. (Page 5)",
    # Quelles sont les conditions pour souscrire à la garantie santé de base Unéo-International?
    "Les conditions pour souscrire à la garantie santé de base Unéo-International sont réservées aux personnes mentionnées à l'article 7 des Statuts, qui n'ont pas fait valoir leurs droits à retraite et qui résident ou sont affectés à l'étranger, hors France métropolitaine et certains territoires d'outre-mer. (Page 7)",
    # Quelles sont les prestations pour la garantie santé de base Unéo-Référence à compter du 1er janvier 2023?
    "Les prestations pour la garantie santé de base Unéo-Référence à compter du 1er janvier 2023 incluent des remboursements pour l'optique, avec des montants spécifiques pour différents types de verres et montures. (Pages 33, 34)",
    # Quelles sont les prestations pour la garantie santé de base Unéo-International "Unéo Monde-Initiale"?
    "Les prestations pour la garantie santé de base Unéo-International \"Unéo Monde-Initiale\" incluent des remboursements à 100% de la dépense pour des consultations, visites, actes de laboratoire, pharmacie, vaccins, et autres, dans la limite de certains plafonds annuels. (Pages 40, 41)",
    # Quelles sont les conditions de modification des garanties santé de base Unéo-Référence et Unéo-International?
    "La souscription à une garantie santé de base Unéo-Référence ou Unéo-International est acquise pour une année civile et se renouvelle par tacite reconduction d'année en année. Le changement de garantie est admis une fois par an, au 1er janvier de l'année suivante, avec un préavis de deux mois. Des exceptions permettent un changement de garantie dans un délai de douze mois à compter de la survenance d'événements spécifiques comme une modification de la situation familiale (mariage, divorce, séparation). (Pages 7, 9)",
])