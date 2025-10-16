from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import utils

model_rooter = OllamaLLM(model="gemma:7b")


template_rooter = """
Tu es un routeur de moteur de recommandation.
Lis la phrase de l'utilisateur et réponds en UNE SEULE LIGNE.

Fonctions possibles :
- recommend_by_user
- recommend_by_movie, <titre exact>
- simple_recommend

Règles strictes :
1) SI la requête contient un mot indiquant un compte (ex. "Letterboxd", "mon compte", "mes goûts", "mes notes", "profil", "compte"), RÉPONDS EXACTEMENT "recommend_by_user".
2) SI la requête cite un film, réponds "recommend_by_movie, <titre exact>" en reprenant le titre TEL QUEL DANS LA PHRASE (pas de traduction, pas de correction).
3) SINON réponds "simple_recommend".
4) AUCUNE explication, AUCUNE parenthèse, AUCUN texte supplémentaire.

Exemples :
Q: "En te basant sur mon Letterboxd recommande moi un film que je pourrais aimer"
A: recommend_by_user

Q: "Recommande-moi des films comme Orange mécanique"
A: recommend_by_movie, Orange mécanique

Q: "Quel film regarder ce soir ?"
A: simple_recommend

Q (mauvais exemple du modèle, à proscrire) :
Q: "En te basant sur mon Letterboxd recommande moi un film que je pourrais aimer"
NE PAS RÉPONDRE: recommend_by_movie, La Noire

Maintenant :
Q: {user_query}
A:
"""

prompt = ChatPromptTemplate.from_template(template_rooter)

chain = prompt | model_rooter

def get_function(message: str):

    response = chain.invoke({"user_query": message,})
    return response
