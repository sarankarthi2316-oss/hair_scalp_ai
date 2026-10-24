import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Curated Clinical Knowledge Base
DOCUMENTS = [
    {
        "id": "doc_aa",
        "condition": "alopecia-areata",
        "text": (
            "Alopecia Areata: An autoimmune disorder causing circular, patchy"
            " hair loss. Standard care involves anti-inflammatory scalp"
            " regimens, avoiding high-tension hairstyles, ensuring adequate"
            " Vitamin D and Zinc intake, and prompt consultation with a"
            " dermatologist for corticosteroid evaluation."
        ),
    },
    {
        "id": "doc_aga",
        "condition": "androgenic-alopecia",
        "text": (
            "Androgenic Alopecia (Pattern Hair Loss): Progressive hair"
            " thinning influenced by dihydrotestosterone (DHT) and genetics."
            " Clinically managed with topical Minoxidil, low-level laser"
            " therapy, and DHT-blocking botanical solutions (e.g., Saw"
            " Palmetto, Rosemary Oil). Requires long-term consistency."
        ),
    },
    {
        "id": "doc_sd",
        "condition": "seborrheic-dermatitis",
        "text": (
            "Seborrheic Dermatitis / Dandruff: Malassezia yeast overgrowth"
            " causing greasy, yellow or white flakes and itching. Use active"
            " clarifying agents like Ketoconazole 2%, Zinc Pyrithione, or"
            " Salicylic Acid shampoo 2-3 times per week. Avoid heavy coconut"
            " or mineral oils that feed the yeast."
        ),
    },
    {
        "id": "doc_pso",
        "condition": "psoriasis",
        "text": (
            "Psoriasis: Autoimmune condition creating thick, silvery scalp"
            " plaques and erythema. Requires gentle descaling with salicylic"
            " acid, coal tar, or prescription topical steroids. Avoid manual"
            " peeling or harsh scratching to prevent the Koebner phenomenon."
        ),
    },
    {
        "id": "doc_te",
        "condition": "telogen-effluvium",
        "text": (
            "Telogen Effluvium (Stress/Nutritional Hair Shedding): Sudden"
            " diffuse hair loss triggered by physiological stress, illness, or"
            " nutrient deficiencies (Iron/Ferritin, Biotin, Protein,"
            " Vitamin D). Focus on stress management, balanced protein"
            " intake, and gentle sulfate-free cleansing."
        ),
    },
    {
        "id": "doc_foll",
        "condition": "folliculitis",
        "text": (
            "Folliculitis & Scalp Infections: Bacterial or fungal"
            " inflammation of hair follicles causing pimples or tender bumps."
            " Keep scalp clean with antibacterial/antifungal washes. Avoid"
            " sharing combs and heavy pore-clogging pomades."
        ),
    },
    {
        "id": "doc_norwood",
        "condition": "advanced-baldness",
        "text": (
            "Norwood Scale Types 4-7 (Advanced Thinning): Scalp skin is more"
            " exposed to UV radiation. Apply broad-spectrum SPF 30+ to the"
            " scalp daily. Use volumizing peptides and consult a hair"
            " restoration specialist for advanced medical therapies."
        ),
    },
]

# 2. Vectorize Documents with TF-IDF Vector Database
corpus = [doc["text"] for doc in DOCUMENTS]
vectorizer = TfidfVectorizer(stop_words="english")
doc_vectors = vectorizer.fit_transform(corpus)


def retrieve_guidelines(condition, density, nutrition, top_k=2):
  """Vector search: Calculates cosine similarity between user query and document vectors"""
  query = f"{condition} {density} {nutrition} routine treatment"
  query_vec = vectorizer.transform([query])
  similarities = cosine_similarity(query_vec, doc_vectors)[0]

  top_indices = np.argsort(similarities)[::-1][:top_k]
  matched_docs = [DOCUMENTS[i]["text"] for i in top_indices]
  return "\n\n".join(matched_docs)


def generate_routine(
    disease, density, risk_score, nutrition, stress, medical_condition
):
  """Generates routine using vector retrieval + Gemini LLM (or structured fallback)"""
  retrieved_context = retrieve_guidelines(disease, density, nutrition)

  api_key = os.environ.get("GEMINI_API_KEY")

  # If API key is configured, use Gemini
  if api_key:
    try:
      from google import genai

      client = genai.Client(api_key=api_key)

      prompt = f"""
            You are a professional AI Trichology and Hair Care Specialist.
            Based on the following multi-modal diagnostics and clinical knowledge, create a comprehensive, personalized hair care and scalp recovery plan.

            User Diagnostics:
            - Detected Scalp Condition: {disease}
            - Detected Baldness / Density Grade: {density}
            - Systemic Hair Loss Risk Score: {risk_score}%
            - Reported Deficiencies: {nutrition}
            - Stress Level: {stress}
            - Prior Medical Conditions: {medical_condition}

            Retrieved Clinical Guidelines:
            {retrieved_context}

            Please structure your response in clean Markdown with these sections:
            ### 1. Diagnostic Summary
            ### 2. Weekly Scalp & Hair Care Schedule (Wash days, treatment days, rest days)
            ### 3. Recommended Active Ingredients & What to Avoid
            ### 4. Dietary & Nutritional Strategy
            ### 5. Clinical Disclaimer (Remind the user to verify with a board-certified dermatologist)
            """
      response = client.models.generate_content(
          model="gemini-2.5-flash", contents=prompt
      )
      return response.text
    except Exception as e:
      print(f"LLM call failed ({e}), using fallback.")

  # Clean fallback using retrieved clinical guidelines
  return f"""
### 1. Diagnostic Summary
- **Primary Scalp Condition**: {disease.replace('-', ' ').title()}
- **Hair Density Assessment**: {density.replace('_', ' ').title()}
- **Systemic Risk Score**: {risk_score}%
- **Identified Deficiencies**: {nutrition}
- **Stress Level**: {stress}

### 2. Clinical Reference Routine
{retrieved_context}

### 3. Recommended Next Steps
- Incorporate gentle, targeted scalp cleansers suited for {disease.replace('-', ' ')}.
- Address dietary intake of {nutrition} through whole foods or doctor-recommended supplementation.
- Maintain a consistent hair wash schedule (2-3 times weekly with lukewarm water).

*Disclaimer: This analysis is an AI-assisted screening tool and does not substitute for formal medical evaluation by a certified dermatologist.*
"""